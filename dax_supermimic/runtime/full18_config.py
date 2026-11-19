"""Validated, auditable runtime settings for bounded full18 execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from ..robot import JOINT_NAMES, ROBOT_CONFIG, ROOT


def command_velocity_limits(cfg):
    """Effective named caps, including any lower joint or model ceiling."""
    joint_limits = cfg.get('joint_velocity_limits_rad_s', cfg.get('joint_urdf_velocity_limits_rad_s'))
    return {name: min(cfg['limits']['max_command_velocity_rad_s'], joint_limits[name],
                      cfg['training_motor_velocity_limits_rad_s'][name]) for name in JOINT_NAMES}


def load_full18_config(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    cfg = json.loads(raw)
    if cfg.get('schema') not in ('dax_full18_execution_v1', 'dax_full18_execution_v2'):
        raise ValueError('Unsupported full18 execution config schema')
    p1 = cfg['schema'] == 'dax_full18_execution_v2'
    required = {'schema', 'actor_hz', 'publish_hz', 'actor_frames', 'control_cpu', 'backend',
                'position_limit_source', 'position_limits', 'joint_urdf_velocity_limits_rad_s',
                'held_joint_urdf_velocity_limits_rad_s', 'joint_velocity_limit_source',
                'training_motor_velocity_limits_rad_s', 'training_robot_profile_sha256',
                'training_motor_velocity_source', 'lower_waist_training_rad',
                'trajectory_point_horizon_ms', 'final_hold_ms', 'feedback_start_timeout_s',
                'controller_query_timeout_s', 'transport_startup_settle_ms',
                'abort_hold_settle_ms', 'limits', 'startup'}
    if p1:
        required -= {'joint_urdf_velocity_limits_rad_s', 'held_joint_urdf_velocity_limits_rad_s'}
        required |= {'joint_velocity_limits_rad_s', 'held_joint_velocity_limits_rad_s',
                     'robot_control_baseline_id', 'geometry_asset', 'geometry_asset_sha256'}
    if set(cfg) != required:
        raise ValueError('Unexpected or missing full18 execution config keys')
    if cfg['actor_hz'] != 50 or cfg['publish_hz'] not in (50, 100):
        raise ValueError('actor_hz must be 50 and publish_hz must be 50 or 100')
    if cfg['publish_hz'] % cfg['actor_hz']:
        raise ValueError('publish_hz must be an integer multiple of actor_hz')
    if isinstance(cfg['actor_frames'], bool) or not isinstance(cfg['actor_frames'], int) or not 1 <= cfg['actor_frames'] <= 24000:
        raise ValueError('actor_frames must be an integer in [1, 24000]')
    if isinstance(cfg['control_cpu'], bool) or not isinstance(cfg['control_cpu'], int) or cfg['control_cpu'] < 0:
        raise ValueError('control_cpu must be a nonnegative integer')
    if cfg['backend'] not in ('torch', 'onnxruntime', 'tensorrt'):
        raise ValueError('Unsupported full18 inference backend')
    position_limits = cfg['position_limits']
    if set(position_limits) != set(JOINT_NAMES):
        raise ValueError('position_limits must define every joint in the trained 18-joint order')
    for joint, bounds in position_limits.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f'Invalid position range for {joint}')
        lo, hi = bounds
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds) or lo >= hi:
            raise ValueError(f'Invalid position range for {joint}')
    joint_velocity = cfg['joint_velocity_limits_rad_s' if p1 else 'joint_urdf_velocity_limits_rad_s']
    if set(joint_velocity) != set(JOINT_NAMES):
        raise ValueError('joint_urdf_velocity_limits_rad_s must define every Actor joint')
    for joint, value in joint_velocity.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 40:
            raise ValueError(f'Invalid URDF velocity limit for {joint}')
    motor_velocity = cfg['training_motor_velocity_limits_rad_s']
    if set(motor_velocity) != set(JOINT_NAMES):
        raise ValueError('training_motor_velocity_limits_rad_s must define every Actor joint')
    for joint, value in motor_velocity.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 40:
            raise ValueError(f'Invalid training motor velocity limit for {joint}')
    if not isinstance(cfg['training_robot_profile_sha256'], str) or len(cfg['training_robot_profile_sha256']) != 64:
        raise ValueError('training_robot_profile_sha256 must be a SHA256 hex digest')
    try:
        int(cfg['training_robot_profile_sha256'], 16)
    except ValueError as exc:
        raise ValueError('training_robot_profile_sha256 must be hexadecimal') from exc
    if not isinstance(cfg['training_motor_velocity_source'], str) or not cfg['training_motor_velocity_source'].strip():
        raise ValueError('training_motor_velocity_source must describe the source and units')
    training_profile_bytes = ROBOT_CONFIG.read_bytes()
    if hashlib.sha256(training_profile_bytes).hexdigest() != cfg['training_robot_profile_sha256']:
        raise ValueError('Training robot profile hash differs from configured motor velocity limits')
    training_profile = json.loads(training_profile_bytes)
    profile_motor_velocity = {row['joint']: row['velocity_limit_rad_s'] for row in training_profile['motors']}
    if set(profile_motor_velocity) != set(JOINT_NAMES) or any(
            not math.isclose(motor_velocity[joint], profile_motor_velocity[joint], rel_tol=0, abs_tol=1e-12)
            for joint in JOINT_NAMES):
        raise ValueError('Configured training motor velocity limits differ from the hashed robot profile')
    held_velocity = cfg['held_joint_velocity_limits_rad_s' if p1 else 'held_joint_urdf_velocity_limits_rad_s']
    if set(held_velocity) != {'waist_joint0', 'waist_joint1'}:
        raise ValueError('Held lower-waist URDF velocity limits must define waist_joint0/1')
    for joint, value in held_velocity.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 40:
            raise ValueError(f'Invalid held-joint URDF velocity limit for {joint}')
    if not isinstance(cfg['joint_velocity_limit_source'], str) or not cfg['joint_velocity_limit_source'].strip():
        raise ValueError('joint_velocity_limit_source must describe the source and units')
    mount = cfg['lower_waist_training_rad']
    if not isinstance(mount, list) or len(mount) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 2.0
            for v in mount):
        raise ValueError('lower_waist_training_rad must contain two finite angles within +/-2 rad')
    if p1:
        if not isinstance(cfg['robot_control_baseline_id'], str) or not cfg['robot_control_baseline_id'].strip():
            raise ValueError('robot_control_baseline_id is required')
        if not isinstance(cfg['geometry_asset'], str) or not cfg['geometry_asset'].strip():
            raise ValueError('geometry_asset is required')
        asset_raw = (ROOT / cfg['geometry_asset']).read_bytes()
        if hashlib.sha256(asset_raw).hexdigest() != cfg['geometry_asset_sha256']:
            raise ValueError('P1 geometry asset hash differs')
        asset = json.loads(asset_raw)
        if (asset['robot_config_sha256'] != cfg['training_robot_profile_sha256'] or
                asset['joint_names'] != JOINT_NAMES):
            raise ValueError('P1 geometry/profile/order binding differs')
        fixed = dict(zip(asset['fixed_joint_names'], asset['fixed_rad']))
        if any(not math.isclose(mount[i], fixed[n], rel_tol=0, abs_tol=1e-12)
               for i, n in enumerate(('waist_joint0', 'waist_joint1'))):
            raise ValueError('P1 lower waist mount differs from geometry')
        for name in JOINT_NAMES:
            row = training_profile['action_interface']['joints'][name]
            expected = [row['position_min_rad'], row['position_max_rad']]
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                   for a, b in zip(position_limits[name], expected)):
                raise ValueError('P1 execution position limits differ from training profile: ' + name)
    for key, lo, hi in [('trajectory_point_horizon_ms', 10, 100), ('final_hold_ms', 100, 5000)]:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
    for key, lo, hi in [('feedback_start_timeout_s', 1, 30),
                        ('controller_query_timeout_s', 1, 10),
                        ('transport_startup_settle_ms', 0, 2000),
                        ('abort_hold_settle_ms', 0, 2000)]:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be in [{lo}, {hi}]')
    limits = cfg['limits']
    startup = cfg['startup']
    limit_keys = {
        'max_command_velocity_rad_s', 'max_command_acceleration_rad_s2',
        'max_feedback_velocity_rad_s', 'max_tracking_gap_rad',
        'max_feedback_receive_age_ms', 'max_feedback_source_age_ms',
        'max_feedback_future_stamp_ms', 'max_publish_stall_ms',
        'publish_wait_ms', 'target_ttl_ms',
    }
    startup_keys = {
        'max_initial_feedback_velocity_rad_s', 'lower_waist_mount_tolerance_rad',
        'lower_waist_hold_tolerance_rad', 'initialization_timeout_s',
        'initialization_position_tolerance_rad', 'initialization_limiter_velocity_rad_s',
        'initialization_feedback_velocity_rad_s', 'initialization_settled_frames',
    }
    if set(limits) != limit_keys or set(startup) != startup_keys:
        raise ValueError('Unexpected or missing full18 limit/startup config keys')

    def number(obj, key, lo, hi, *, inclusive_lo=False):
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if (value < lo if inclusive_lo else value <= lo) or value > hi:
            op = '[' if inclusive_lo else '('
            raise ValueError(f'{key} must be in {op}{lo}, {hi}]')
        return float(value)

    # v2 operating limits are positive finite configuration, not hard-coded
    # historical trial caps. Velocity still respects the per-joint ceilings.
    number(limits, 'max_command_velocity_rad_s', 0, math.inf if p1 else 0.12)
    number(limits, 'max_command_acceleration_rad_s2', 0, math.inf if p1 else 0.25)
    number(limits, 'max_feedback_velocity_rad_s', 0, math.inf if p1 else 3.0)
    number(limits, 'max_tracking_gap_rad', 0, 0.2)
    number(limits, 'max_feedback_receive_age_ms', 0, 1000)
    number(limits, 'max_feedback_source_age_ms', 0, 1000)
    number(limits, 'max_feedback_future_stamp_ms', 0, 100)
    number(limits, 'max_publish_stall_ms', 20, 200, inclusive_lo=True)
    number(limits, 'publish_wait_ms', 0, 40, inclusive_lo=True)
    number(limits, 'target_ttl_ms', 0, 100)
    if limits['publish_wait_ms'] >= limits['target_ttl_ms']:
        raise ValueError('publish_wait_ms must be less than target_ttl_ms')
    number(startup, 'max_initial_feedback_velocity_rad_s', 0, 0.5)
    number(startup, 'lower_waist_mount_tolerance_rad', 0, 0.1)
    number(startup, 'lower_waist_hold_tolerance_rad', 0, 0.1)
    number(startup, 'initialization_timeout_s', 0, 120)
    number(startup, 'initialization_position_tolerance_rad', 0, 0.1)
    number(startup, 'initialization_limiter_velocity_rad_s', 0, 0.12)
    number(startup, 'initialization_feedback_velocity_rad_s', 0, 0.5)
    settled = startup['initialization_settled_frames']
    if isinstance(settled, bool) or not isinstance(settled, int) or not 1 <= settled <= 250:
        raise ValueError('initialization_settled_frames must be an integer in [1, 250]')

    cfg['_config_path'] = str(path)
    cfg['_config_sha256'] = hashlib.sha256(raw).hexdigest()
    return cfg
