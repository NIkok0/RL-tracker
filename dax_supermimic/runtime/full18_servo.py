"""Full-servo wire contract for the DAX head/waist servo controller."""
import numpy as np
from ..robot import JOINT_NAMES

HEAD_WAIST_JOINTS = ('head_joint0', 'head_joint1', 'waist_joint0', 'waist_joint1', 'waist_joint2', 'waist_joint3')
HEAD_WAIST_TOPIC = '/head_waist_servo_controller/commands'
DAX_HEAD_WAIST_TYPE = 'daxbot_control/HeadWaistServoController'


def limited_position_step(limiter, target=None):
    """Slew a target and expose which one-tick requests exceeded configured caps.

    A finite oversized request is normal input: the position limiter consumes it
    and returns a bounded command. Flags describe the hypothetical one-tick
    request, not measured robot velocity or an inference failure.
    """
    previous_q, previous_v = limiter.q.copy(), limiter.v.copy()
    if target is None:
        requested_v = np.zeros_like(previous_v)
    else:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != previous_q.shape or not np.isfinite(target).all():
            raise ValueError('Expected finite position target matching the limiter')
        requested_v = (target - previous_q) / limiter.dt
    requested_acc = (np.clip(requested_v, -limiter.vmax, limiter.vmax) - previous_v) / limiter.dt
    q = limiter.step(target)
    velocity = (q - previous_q) / limiter.dt
    acceleration = (velocity - previous_v) / limiter.dt
    return q, velocity, acceleration, {
        'velocity_limited_indices': np.flatnonzero(abs(requested_v) > limiter.vmax + 1e-9).tolist(),
        'acceleration_limited_indices': np.flatnonzero(abs(requested_acc) > limiter.amax + 1e-9).tolist(),
    }


def head_waist_target(q, lower_hold):
    q = np.asarray(q, dtype=float)
    lower_hold = np.asarray(lower_hold, dtype=float)
    if q.shape != (18,) or lower_hold.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(lower_hold).all():
        raise ValueError('Expected finite 18D Actor target and 2D lower-waist hold')
    values = dict(zip(JOINT_NAMES, q))
    values.update(zip(('waist_joint0', 'waist_joint1'), lower_hold))
    return [float(values[n]) for n in HEAD_WAIST_JOINTS]


def clamp_feedback_velocity(qd, cap):
    qd = np.asarray(qd, dtype=np.float64)
    if qd.ndim != 1 or not np.isfinite(qd).all():
        raise ValueError('Feedback velocity must be a finite vector')
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('Feedback velocity cap must be finite and positive')
    clipped = np.clip(qd, -cap, cap)
    return clipped, np.flatnonzero(clipped != qd)


def plan_reference_frames(actor_frames, reference_frames, first_invalid_frame=None):
    if actor_frames < 1 or reference_frames < 1:
        return 0
    candidates=[actor_frames,reference_frames]
    if first_invalid_frame is not None:candidates.append(first_invalid_frame)
    return min(candidates)


def validate_servo_controllers(state):
    expected = {f'{s}_arm_servo_controller': [f'{s}_arm_joint{i}/position' for i in range(7)] for s in ('left', 'right')}
    for name, interfaces in expected.items():
        if (name not in state or state[name].state != 'active' or
                state[name].type != 'position_controllers/JointGroupPositionController' or
                state[name].required_command_interfaces != interfaces):
            raise RuntimeError(name+' servo state/order mismatch')
    upper = state.get('head_waist_servo_controller')
    required = [n+'/position' for n in HEAD_WAIST_JOINTS] + ['waist_joint0/servo_active']
    if (upper is None or upper.state != 'active' or upper.type != DAX_HEAD_WAIST_TYPE or
            upper.required_command_interfaces != required):
        raise RuntimeError('head_waist_servo_controller type/interface mismatch')
    for name in ('left_arm_controller', 'right_arm_controller', 'waist_controller', 'waist_rotation_controller', 'head_controller'):
        if name not in state or state[name].state != 'inactive':
            raise RuntimeError(name+' must be inactive for full servo')
