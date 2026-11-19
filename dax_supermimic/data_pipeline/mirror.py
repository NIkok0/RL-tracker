"""Pure left/right reflection for physical-v3 joint vectors."""
from __future__ import annotations

import numpy as np

JOINT_NAMES = (
    'waist_joint2', 'waist_joint3',
    *(f'left_arm_joint{i}' for i in range(7)),
    *(f'right_arm_joint{i}' for i in range(7)),
    'head_joint0', 'head_joint1',
)

# Each target joint takes the named source joint multiplied by this sign.
MIRROR_SOURCE = (
    'waist_joint2', 'waist_joint3',
    *(f'right_arm_joint{i}' for i in range(7)),
    *(f'left_arm_joint{i}' for i in range(7)),
    'head_joint0', 'head_joint1',
)
MIRROR_SIGN = (1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1)
MIRROR_INDEX = tuple(JOINT_NAMES.index(name) for name in MIRROR_SOURCE)


def mirror_q(q: np.ndarray) -> np.ndarray:
    """Reflect one or more 18-joint vectors across the base-frame y=0 plane."""
    values = np.asarray(q)
    if values.ndim == 0 or values.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f'Expected the final dimension to contain {len(JOINT_NAMES)} joints')
    signs = np.asarray(MIRROR_SIGN, dtype=values.dtype)
    return values[..., MIRROR_INDEX] * signs
