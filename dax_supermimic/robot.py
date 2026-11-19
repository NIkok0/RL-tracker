"""DAX robot description for the otherwise upstream SuperMimic algorithm.

Motor data and the versioned joint-range action interface are explicit inputs.
Constructing physical drives fails when per-joint motor values are absent.
No legacy command limiter, gravity compensation or servo gains are imported.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_robot_config_env = os.environ.get('DAX_ROBOT_CONFIG', 'config/dax_supermimic_robot_v3_163_linkerhand_20261117.json')
ROBOT_CONFIG = (ROOT / _robot_config_env).resolve()
_CONFIG = json.loads(ROBOT_CONFIG.read_text())
JOINT_NAMES = list(_CONFIG['joint_names'])
NUM_J = len(JOINT_NAMES)
TRACKED_BODIES = list(_CONFIG['tracked_bodies'])
FIXED_BASE = bool(_CONFIG['fixed_base'])
ROOT_BODY = _CONFIG['root_body']
TORSO_BODY = _CONFIG['torso_body']
URDF_PATH = str(ROOT / _CONFIG['body_urdf'])
MJCF_PATH = str(ROOT / _CONFIG['kinematics_mjcf'])
USD_PATH = str(ROOT / _CONFIG['usd_path'])
MOTION_FPS = 50
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0
# A lazy value: only load_isaac_robot_cfg constructs an Isaac Lab object.
ROBOT_CFG = None


class RobotParametersMissing(ValueError):
    """A robot-specific input needed by the upstream actuator model is absent."""


def load_drives(path=ROBOT_CONFIG, *, require_complete=False):
    """Read explicit robot motors; never infer rotor inertia from body mass."""
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'dax_supermimic_robot_v3':
        raise ValueError('Unsupported DAX SuperMimic robot profile')
    if config.get('joint_names') != JOINT_NAMES:
        raise ValueError('Robot motor order differs from the named data/kinematics order')
    motors = config.get('motors', [])
    if [row.get('joint') for row in motors] != JOINT_NAMES:
        raise ValueError('Robot profile must contain one ordered motor for every joint')
    missing = []
    for row in motors:
        for key in ('effort_limit_nm', 'velocity_limit_rad_s'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {key} for {row["joint"]}')
        value = row.get('armature_kg_m2')
        if value is None:
            missing.append(row['joint'])
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid motor reflected armature for '+row['joint'])
    if require_complete and missing:
        raise RobotParametersMissing(
            'MISSING_MOTOR_ARMATURE: supply per-joint reflected motor inertia in '
            + str(path) + '; no neutral-body-inertia substitution is allowed. Missing: '
            + ', '.join(missing))
    return config


def motor_of(joint_name: str) -> tuple[float, float, float]:
    config = load_drives(require_complete=True)
    row = next((item for item in config['motors'] if item['joint'] == joint_name), None)
    if row is None:
        raise KeyError(joint_name)
    return row['effort_limit_nm'], row['velocity_limit_rad_s'], row['armature_kg_m2']


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


def upstream_action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)


def action_scale(joint_name: str) -> float:
    """Robot task coverage, independent of the unchanged motor PD gains."""
    return float(_CONFIG['action_interface']['joints'][joint_name]['scale_rad'])


def default_joint_pos_list() -> list[float]:
    """Named robot zero pose, replacing G1's crouched default posture."""
    return [float(_CONFIG['default_joint_position_rad'][name]) for name in JOINT_NAMES]


def load_isaac_robot_cfg():
    """Build upstream-style implicit PD only after explicit drive inputs exist."""
    global ROBOT_CFG
    config = load_drives(require_complete=True)
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
    except ModuleNotFoundError as error:
        raise RuntimeError('ISAACLAB_RUNTIME_REQUIRED: install the pinned upstream Isaac Lab 3 kit-less/Newton environment') from error
    if not Path(USD_PATH).is_file():
        raise FileNotFoundError('DAX_USD_REQUIRED: convert the DAX robot asset with the pinned upstream offline importer: '+USD_PATH)
    acts = {}
    for row in config['motors']:
        name, arm = row['joint'], row['armature_kg_m2']
        acts[name] = ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=row['effort_limit_nm'],
            velocity_limit_sim=row['velocity_limit_rad_s'], stiffness=_kp(arm),
            damping=_kd(arm), armature=arm)
    ROBOT_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH, variants={'Physics': 'mujoco'}, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            # The converted MJCF already has a fixed root in the Newton model.
            # The legacy fix_root_link helper relocates a referenced USD schema
            # and fails on this importer output; do not author a second root.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(config['root_position_m']),
            joint_pos=dict(config['default_joint_position_rad']), joint_vel={'.*': 0.0}),
        soft_joint_pos_limit_factor=0.9, actuators=acts)
    return ROBOT_CFG
