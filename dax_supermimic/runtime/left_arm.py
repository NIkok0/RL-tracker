"""Left-arm message boundary; never publishes or invents applied-action feedback."""
from __future__ import annotations

import numpy as np

from ..robot import JOINT_NAMES

LEFT_NAMES = tuple(f'left_arm_joint{i}' for i in range(7))
LEFT_INDICES = tuple(JOINT_NAMES.index(name) for name in LEFT_NAMES)
LEFT_TOPIC = '/left_arm_servo_controller/commands'
LEFT_TYPE = 'std_msgs/msg/Float64MultiArray'


def ordered_feedback(message):
    names = list(message['name'])
    if len(set(names)) != len(names):
        raise ValueError('Duplicate joint names')
    q = np.asarray(message['position'], dtype=np.float32)
    qd = np.asarray(message['velocity'], dtype=np.float32)
    if q.shape != (len(names),) or qd.shape != (len(names),):
        raise ValueError('Complete position and velocity arrays required')
    if not all(name in names for name in JOINT_NAMES):
        raise ValueError('Feedback is missing model joints')
    index = [names.index(name) for name in JOINT_NAMES]
    q, qd = q[index], qd[index]
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError('Nonfinite model joint feedback')
    return q, qd


def left_command(target):
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (18,) or not np.isfinite(target).all():
        raise ValueError('Expected finite 18-joint Actor position target')
    return {'topic': LEFT_TOPIC, 'type': LEFT_TYPE,
            'message': {'layout': {'dim': [], 'data_offset': 0},
                        'data': target[list(LEFT_INDICES)].tolist()}}
