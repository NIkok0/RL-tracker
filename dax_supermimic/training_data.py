"""Explicit mixed-source data and fixed-mount adapters; original learner stays intact."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_geometry(asset_path):
    import dax_supermimic.robot as robot
    asset = json.loads(Path(asset_path).read_text())
    for name, expected in asset['hashes'].items():
        if sha(ROOT/name) != expected: raise ValueError('Geometry hash mismatch: '+name)
    robot.MJCF_PATH = asset['mjcf']; robot.USD_PATH = asset['usd']; robot.ROBOT_CFG = None
    return asset






def verify_runtime_geometry(g):
    import torch
    from dax_supermimic.robot import TRACKED_BODIES
    from dax_supermimic.rotations import quat_to_mat
    env = g['env']; env.refresh()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Geometry admission is FP32; restore the original training TF32 setting immediately.
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            p, r, _, _ = g['kin'].forward(*env.state())
            ids = env.robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0]
            poses = env.robot.data.body_link_pose_w.torch[:, ids]
            actual_p = poses[..., :3]-env.scene.env_origins[:, None, :]
            actual_r = quat_to_mat(torch.cat((poses[..., 6:7], poses[..., 3:6]), -1))
            pe, re = float((p-actual_p).abs().max()), float((r-actual_r).abs().max())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    if pe > 1e-4 or re > 1e-4 or len(env.robot.joint_names) != 18:
        raise ValueError(f'Fixed mount FK/USD mismatch {pe}, {re}')
    return {'passed': True, 'envs': env.N, 'position_max_component_error_m': pe,
            'rotation_max_component_error': re, 'controlled_dofs': 18, 'fixed_base': bool(env.robot.is_fixed_base),
            'check_precision': 'FP32 without TF32', 'training_tf32_restored': torch.backends.cuda.matmul.allow_tf32 == original_tf32}
