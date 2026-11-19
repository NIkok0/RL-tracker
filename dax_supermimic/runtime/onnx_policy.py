"""Optional single-sample ONNX Runtime and TensorRT Actor backends."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def _metadata(artifact, checkpoint_sha256, hist, goal_dim, action_contract=None):
    path = Path(artifact).with_suffix(Path(artifact).suffix + '.json')
    data = json.loads(path.read_text())
    if action_contract is None:
        from ..actions import ACTION_CONTRACT_ID
        action_contract = ACTION_CONTRACT_ID
    if data.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('ONNX_CHECKPOINT_MISMATCH')
    if data.get('action_contract') != action_contract:
        raise ValueError('ONNX_ACTION_CONTRACT_MISMATCH')
    if data.get('input_shapes') != {'obs_hist': [1, hist, 66], 'goal': [1, goal_dim]}:
        raise ValueError('ONNX_INPUT_CONTRACT_MISMATCH')
    if data.get('output_shape') != [1, 18] or data.get('precision') != 'fp32':
        raise ValueError('ONNX_OUTPUT_CONTRACT_MISMATCH')
    if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != data.get('artifact_sha256'):
        raise ValueError('ONNX_ARTIFACT_HASH_MISMATCH')
    return data


class ONNXRuntimePolicy:
    def __init__(self, model, checkpoint_sha256, hist, goal_dim, device, action_contract, providers=None):
        self.path = Path(model).resolve()
        self.metadata = _metadata(self.path, checkpoint_sha256, hist, goal_dim, action_contract)
        import onnxruntime as ort
        chosen = providers or (['CUDAExecutionProvider','CPUExecutionProvider']
                               if torch.device(device).type=='cuda' else ['CPUExecutionProvider'])
        available = ort.get_available_providers()
        chosen = [p for p in chosen if p in available]
        if not chosen:
            raise RuntimeError('No requested ONNX Runtime provider is available')
        gpu_requested=torch.device(device).type == 'cuda'
        if gpu_requested:
            if not any(p in chosen for p in ('CUDAExecutionProvider','TensorrtExecutionProvider')):
                raise RuntimeError('Requested CUDA ONNX execution but GPU provider is unavailable')
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(self.path),sess_options=options,providers=chosen)
        if gpu_requested and not any(p in self.session.get_providers() for p in ('CUDAExecutionProvider','TensorrtExecutionProvider')):
            raise RuntimeError('ONNX Runtime silently fell back to CPU')
        self.device = torch.device(device)

    def __call__(self, history, goal):
        inputs = {'obs_hist': history.detach().cpu().numpy(),
                  'goal': goal.detach().cpu().numpy()}
        result = self.session.run(['raw_action'], inputs)[0]
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)


def _import_tensorrt():
    try:
        return importlib.import_module('tensorrt')
    except ModuleNotFoundError:
        for path in ('/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages'):
            if path not in sys.path:
                sys.path.append(path)
        return importlib.import_module('tensorrt')


class TensorRTPolicy:
    """FP32 static engine using torch-owned CUDA buffers and stream."""
    def __init__(self, engine, checkpoint_sha256, hist, goal_dim, device, action_contract):
        self.path = Path(engine).resolve()
        self.metadata = _metadata(self.path, checkpoint_sha256, hist, goal_dim, action_contract)
        if self.metadata.get('engine_backend') != 'tensorrt-fp32':
            raise ValueError('TENSORRT_ENGINE_METADATA_MISMATCH')
        if torch.device(device).type != 'cuda':
            raise ValueError('TensorRT backend requires a CUDA device')
        self.device = torch.device(device)
        trt = _import_tensorrt()
        if self.metadata.get('tensorrt_version') != trt.__version__:
            raise ValueError('TENSORRT_VERSION_MISMATCH: rebuild engine on this target')
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT failed to deserialize the engine')
        self.context = self.engine.create_execution_context()
        self.input_names = ('obs_hist', 'goal')
        self.output_name = 'raw_action'
        for name, shape in (('obs_hist', (1, hist, 66)), ('goal', (1, goal_dim)),
                            ('raw_action', (1, 18))):
            if tuple(self.engine.get_tensor_shape(name)) != shape:
                raise ValueError(f'TensorRT tensor shape mismatch: {name}')
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f'TensorRT tensor dtype mismatch: {name}')
        self.output = torch.empty((1, 18), dtype=torch.float32, device=self.device)

    def __call__(self, history, goal):
        history = history.contiguous()
        goal = goal.contiguous()
        if history.device != self.device or goal.device != self.device:
            raise ValueError('TensorRT inputs must already be on the configured CUDA device')
        if list(history.shape) != self.metadata['input_shapes']['obs_hist'] or list(goal.shape) != self.metadata['input_shapes']['goal']:
            raise ValueError('TensorRT input shape differs from exported contract')
        if history.dtype != torch.float32 or goal.dtype != torch.float32:
            raise ValueError('TensorRT inputs must be FP32')
        self.context.set_tensor_address('obs_hist', history.data_ptr())
        self.context.set_tensor_address('goal', goal.data_ptr())
        self.context.set_tensor_address('raw_action', self.output.data_ptr())
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError('TensorRT enqueueV3 failed')
        return self.output


def build_tensorrt_engine(onnx_model, engine_path, workspace_bytes=1 << 29):
    """Build on the target device; TensorRT plans are hardware/version specific."""
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    model = Path(onnx_model)
    if not parser.parse(model.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError('TensorRT ONNX parse failed: ' + '\n'.join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if hasattr(trt.BuilderFlag, 'TF32'):
        config.clear_flag(trt.BuilderFlag.TF32)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    target = Path(engine_path)
    if target.exists():
        raise FileExistsError('Refusing to overwrite TensorRT engine')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(serialized))
    source_meta = json.loads(model.with_suffix(model.suffix + '.json').read_text())
    meta = dict(source_meta, artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source_onnx_sha256=source_meta['artifact_sha256'],
                engine_backend='tensorrt-fp32', tensorrt_version=trt.__version__)
    target.with_suffix(target.suffix + '.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta
