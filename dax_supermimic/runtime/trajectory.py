"""Fixed-base recorded trajectory inference, without simulator or hardware dispatch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..actions import (ACTION_CONTRACT_ID, ACTION_RUNTIME_SCHEMA, checkpoint_runtime_action,
                       project_action)
from ..motion import CONTROL_DT, _gradient
from ..networks import Policy
from ..robot import JOINT_NAMES, ROBOT_CONFIG
from ..rotations import mat_to_6d


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}, got {result.shape}')
    return result


class JointTrajectory:
    """50 Hz, radians, fixed identity root; no implicit resampling or clipping."""

    def __init__(self, q):
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 18 or len(q) < 2:
            raise ValueError('Reference must have at least two frames and 18 joint columns')
        self.q = torch.from_numpy(array(q, q.shape, 'reference'))
        self.qd = _gradient(self.q, CONTROL_DT)

    def frame_window(self, start=0, stop=None):
        """Select reference frames without recomputing source finite differences."""
        stop = len(self.q) if stop is None else stop
        if not 0 <= start < stop <= len(self.q):
            raise ValueError('Reference frame window outside source range')
        result = object.__new__(JointTrajectory)
        result.q = self.q[start:stop].clone()
        result.qd = self.qd[start:stop].clone()
        return result

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as data:
                if list(data['joint_names'].astype(str)) != JOINT_NAMES:
                    raise ValueError('joint_names must match robot order exactly')
                if str(data['units'].item()) != 'rad' or not np.isclose(float(data['fps'].item()), 50):
                    raise ValueError('Expected radians and 50 Hz')
                return cls(data['reference_q'])
        raw = np.loadtxt(path, delimiter=',', ndmin=2)
        if raw.shape[1] == 25:
            if not np.allclose(raw[:, :7], [0, 0, 0, 0, 0, 0, 1], atol=1e-6):
                raise ValueError('Only fixed origin / identity xyzw root is supported')
            raw = raw[:, 7:]
        return cls(raw)

    def goal(self, frame, frames, device='cpu'):
        if not 0 <= frame < len(self.q):
            raise ValueError('Reference frame out of range')
        future = torch.arange(frame + 1, frame + frames + 1).clamp(max=len(self.q) - 1)
        # Same 51 features as MotionLib.goal for the fixed identity root.
        static = torch.cat((torch.zeros(3), mat_to_6d(torch.eye(3)), torch.zeros(6)))
        return torch.cat((static.expand(frames, -1), self.q[future], self.qd[future] * .1), -1).reshape(1, -1).to(device)


class TrajectoryActor:
    """One stream per instance. Caller supplies actual previous applied action."""

    def __init__(self, checkpoint, device='cpu', *, backend='torch', artifact=None):
        self.device = torch.device(device)
        if backend not in ('torch','onnxruntime','tensorrt'):
            raise ValueError('backend must be torch, onnxruntime or tensorrt')
        self.backend=backend
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        transform = checkpoint_runtime_action(ck, allow_portable=True)
        self.runtime_action = transform
        self.action_contract = ACTION_RUNTIME_SCHEMA if 'runtime_action' in ck else ACTION_CONTRACT_ID
        self.K, self.H = int(ck['args']['goal_frames']), int(ck['args']['hist'])
        if self.K < 1 or self.H < 1:
            raise ValueError('Invalid checkpoint history/goal dimensions')
        if backend=='torch':
            # Deliberately FP32 baseline. Avoid the training module's global CUDA autocast.
            from .. import networks
            if networks.AMP_DTYPE is not None:
                raise RuntimeError('Use an isolated inference process with networks.AMP_DTYPE=None')
            self.actor = Policy(66, 51 * self.K, hist=self.H).to(self.device).eval()
            self.actor.load_state_dict(ck['pi'], strict=True)
        else:
            if artifact is None:
                raise ValueError('ONNX/TensorRT backend requires --artifact')
            from .onnx_policy import ONNXRuntimePolicy,TensorRTPolicy
            cls=ONNXRuntimePolicy if backend=='onnxruntime' else TensorRTPolicy
            self.actor=cls(artifact,self.checkpoint_sha256,self.H,51*self.K,self.device,self.action_contract)
        self.iteration = int(ck['iteration'])
        self.default = torch.tensor(transform['default_joint_position_rad'], device=self.device)
        self.scale = torch.tensor(transform['scale_rad'], device=self.device)
        spec = json.loads(ROBOT_CONFIG.read_text())['action_interface']['joints']
        self.lo = np.array([spec[n]['position_min_rad'] for n in JOINT_NAMES])
        self.hi = np.array([spec[n]['position_max_rad'] for n in JOINT_NAMES])
        self.reset()

    def reset(self):
        self.history = None

    @torch.inference_mode()
    def step(self, q, qdot, applied_action_previous, goal):
        tensor = lambda v, n: torch.as_tensor(array(v, (18,), n), device=self.device)
        q, qd, prev = tensor(q, 'q'), tensor(qdot, 'qdot'), tensor(applied_action_previous, 'previous action')
        if bool((prev.abs() > 20).any()):
            raise ValueError('Previous applied normalized action exceeds action contract')
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if goal.shape != (1, 51 * self.K) or not bool(torch.isfinite(goal).all()):
            raise ValueError('Invalid future goal')
        # Fixed root; zero joint offsets. Matches noise-free MDP.obs.
        obs = torch.cat((q - self.default, qd, q.new_zeros(3), q.new_tensor([0, 0, -1]), prev,
                         mat_to_6d(torch.eye(3, device=self.device)))).reshape(1, 1, 66)
        history = obs.repeat(1, self.H, 1) if self.history is None else torch.cat((self.history[:, 1:], obs), 1)
        raw = self.actor(history, goal)
        action = project_action(raw)
        target = self.default + self.scale * action[0]
        if not bool(torch.isfinite(raw).all() and torch.isfinite(target).all()):
            raise ValueError('Nonfinite Actor output')
        self.history = history
        target = target.cpu().numpy()
        return dict(raw_action=raw[0].cpu().numpy(), action=action[0].cpu().numpy(),
                    joint_position_target=target, within_position_limits=bool(((target >= self.lo) & (target <= self.hi)).all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True, help='50 Hz radians: headerless 18/25-column CSV, or documented NPZ')
    parser.add_argument('--observation-mode', required=True, choices=['recorded', 'reference-proxy'])
    parser.add_argument('--observations', help='NPZ containing q, qdot, applied_action_previous, joint_names, units, fps')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend',choices=['torch','onnxruntime','tensorrt'],default='torch')
    parser.add_argument('--artifact',help='ONNX model for onnxruntime, TensorRT plan for tensorrt')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', required=True, help='New output directory; existing paths are refused')
    args = parser.parse_args()
    if args.threads < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('threads and max-frames must be positive')
    out = Path(args.out)
    if out.exists():
        parser.error('Output directory already exists')
    torch.set_num_threads(args.threads)
    ref = JointTrajectory.load(args.reference)
    n = min(len(ref.q), args.max_frames or len(ref.q))
    if args.observation_mode == 'recorded':
        if not args.observations:
            parser.error('recorded mode requires --observations')
        with np.load(args.observations, allow_pickle=False) as data:
            if list(data['joint_names'].astype(str)) != JOINT_NAMES or str(data['units'].item()) != 'rad' or not np.isclose(float(data['fps'].item()), 50):
                raise ValueError('Observation joint order / units / fps mismatch')
            measurements = {k: array(data[k], (len(ref.q), 18), k) for k in ('q', 'qdot', 'applied_action_previous')}
    elif args.observations:
        parser.error('reference-proxy mode does not accept observations')
    actor = TrajectoryActor(args.checkpoint, args.device,backend=args.backend,artifact=args.artifact)
    sync = lambda: torch.cuda.synchronize(actor.device) if actor.device.type == 'cuda' else None
    # Warm up without advancing the stream used for output.
    for _ in range(5):
        actor.step(ref.q[0].numpy(), ref.qd[0].numpy(), np.zeros(18), ref.goal(0, actor.K))
    sync()
    actor.reset()
    results, latency = [], []
    for frame in range(n):
        if args.observation_mode == 'recorded':
            q, qd, previous = (measurements[k][frame] for k in ('q', 'qdot', 'applied_action_previous'))
        else:
            # Explicit synthetic inputs; predictions are never presented as applied feedback.
            q, qd, previous = ref.q[frame].numpy(), ref.qd[frame].numpy(), np.zeros(18)
        sync()
        start = time.perf_counter()
        result = actor.step(q, qd, previous, ref.goal(frame, actor.K, actor.device))
        sync()
        latency.append((time.perf_counter() - start) * 1000)
        results.append(result)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'predictions.npz', **{k: np.asarray([r[k] for r in results]) for k in results[0]},
                        time_s=np.arange(n) * CONTROL_DT, inference_ms=np.array(latency), joint_names=JOINT_NAMES)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = dict(schema='dax_trajectory_inference_v1', scope='offline proposals only; no simulation or hardware execution',
                   checkpoint=str(actor.checkpoint), checkpoint_sha256=actor.checkpoint_sha256, iteration=actor.iteration,
                   reference=str(Path(args.reference).resolve()), reference_sha256=sha(args.reference),
                   observation_mode=args.observation_mode, observations_sha256=sha(args.observations) if args.observations else None,
                   action_contract=actor.action_contract, device=str(actor.device), precision='fp32', threads=args.threads,
                   torch_version=torch.__version__,backend=args.backend,frames=n, reference_frames=len(ref.q), fps=50,
                   hist=actor.H, goal_frames=actor.K, history_initialization='repeat first observation',
                   tail='clamp future frame index; preserve last-frame reference velocity as MotionLib does',
                   proxy_previous_action='zeros each frame' if args.observation_mode == 'reference-proxy' else None,
                   within_position_limits_frames=sum(r['within_position_limits'] for r in results),
                   latency_ms=dict(mean=float(np.mean(latency)), p50=float(np.percentile(latency, 50)),
                                   p95=float(np.percentile(latency, 95)), max=float(np.max(latency))),
                   backend_providers=getattr(getattr(actor.actor,'session',None),'get_providers',lambda:None)(),
                   timing_scope='goal construction + observation/history + selected backend + output transfer; 5 warmup calls excluded; no real-time pacing')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
