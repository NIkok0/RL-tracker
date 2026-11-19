"""Versioned DAX action projection, exploration units and checkpoint compatibility."""
from __future__ import annotations

import hashlib
import json
import math

import torch

from .robot import JOINT_NAMES, ROBOT_CONFIG, ROOT

_CONFIG_BYTES = ROBOT_CONFIG.read_bytes()
_CONFIG = json.loads(_CONFIG_BYTES)
_SPEC = _CONFIG['action_interface']
ACTION_CONTRACT_ID = _SPEC['profile'] + ':' + hashlib.sha256(_CONFIG_BYTES).hexdigest()
ACTION_RUNTIME_SCHEMA = 'dax_joint_range_action_runtime_v1'
ACTION_CLIP = float(_SPEC['clip_abs'])


def validate_action_interface():
    if _SPEC['schema'] != 'dax_action_interface_v1' or ACTION_CLIP != 20.0:
        raise ValueError('Unsupported DAX action interface or clipping boundary')
    if _SPEC['coverage_action_abs'] != 10.0 or set(_SPEC['joints']) != set(JOINT_NAMES):
        raise ValueError('Invalid action coverage or named joints')
    model = ROOT / _SPEC['joint_limit_model']
    if hashlib.sha256(model.read_bytes()).hexdigest() != _SPEC['joint_limit_model_sha256']:
        raise ValueError('Action interface joint-limit model hash differs')
    for motor in _CONFIG['motors']:
        name = motor['joint']
        row = _SPEC['joints'][name]
        old = .25 * motor['effort_limit_nm'] / (motor['armature_kg_m2'] * (2 * math.pi * 10)**2)
        default = _CONFIG['default_joint_position_rad'][name]
        lo, hi = row['position_min_rad'], row['position_max_rad']
        expected = max(old, max(abs(lo-default), abs(hi-default)) / 10.)
        if not lo <= default <= hi:
            raise ValueError('Default position is outside joint limits: ' + name)
        for actual, target in ((row['upstream_scale_rad'], old), (row['scale_rad'], expected),
                               (row['noise_ratio'], old / expected)):
            if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, target, rel_tol=1e-10):
                raise ValueError('Invalid DAX action parameter: ' + name)
    return ACTION_CONTRACT_ID


def project_action(action: torch.Tensor) -> torch.Tensor:
    """Applied action shared by collect, imagination, evaluation and replay."""
    return torch.clamp(action, -ACTION_CLIP, ACTION_CLIP)


def exploration_noise_ratio(device: str) -> torch.Tensor:
    """Scale dimensionless noise so its unclipped position/PD excitation is preserved."""
    return torch.tensor([_SPEC['joints'][name]['noise_ratio'] for name in JOINT_NAMES],
                        dtype=torch.float32, device=device)


def require_action_contract(checkpoint: dict):
    actual = checkpoint.get('action_contract')
    if actual != ACTION_CONTRACT_ID:
        raise ValueError(f'ACTION_CONTRACT_MISMATCH: expected {ACTION_CONTRACT_ID}, found {actual}; '
                         'start a fresh model/normalizers/replay for this action interface')


def runtime_action_metadata():
    """Checkpoint-owned action transforms for portable inference runtimes."""
    return {
        'schema': ACTION_RUNTIME_SCHEMA,
        'joint_names': list(JOINT_NAMES),
        'clip_abs': ACTION_CLIP,
        'default_joint_position_rad': [float(_CONFIG['default_joint_position_rad'][n]) for n in JOINT_NAMES],
        'scale_rad': [float(_SPEC['joints'][n]['scale_rad']) for n in JOINT_NAMES],
        'source_action_contract': ACTION_CONTRACT_ID,
    }


def checkpoint_runtime_action(checkpoint: dict, *, allow_portable=False):
    """Return the checkpoint's output transform; legacy checkpoints remain profile-bound."""
    spec = checkpoint.get('runtime_action')
    if spec is None:
        require_action_contract(checkpoint)
        return runtime_action_metadata()
    if not allow_portable:
        require_action_contract(checkpoint)
    if not isinstance(spec, dict):
        raise ValueError('Invalid checkpoint runtime action metadata')
    if spec.get('schema') != ACTION_RUNTIME_SCHEMA or spec.get('joint_names') != JOINT_NAMES:
        raise ValueError('ACTION_RUNTIME_SCHEMA_MISMATCH')
    if spec.get('clip_abs') != ACTION_CLIP:
        raise ValueError('ACTION_RUNTIME_CLIP_MISMATCH')
    if (not isinstance(spec.get('source_action_contract'), str) or
            spec.get('source_action_contract') != checkpoint.get('action_contract')):
        raise ValueError('ACTION_RUNTIME_PROVENANCE_MISSING')
    for key in ('default_joint_position_rad', 'scale_rad'):
        values = spec.get(key)
        if (not isinstance(values, list) or len(values) != len(JOINT_NAMES) or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            raise ValueError(f'Invalid checkpoint runtime action {key}')
    if any(v <= 0 for v in spec['scale_rad']):
        raise ValueError('Checkpoint runtime action scales must be positive')
    return spec


validate_action_interface()
