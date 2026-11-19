"""Waist/head trajectory message construction; no transport or motor writes."""
import numpy as np
from ..robot import JOINT_NAMES

GROUPS={
    'waist':('waist_joint0','waist_joint1','waist_joint2'),
    'waist_rotation':('waist_joint3',),
    'head':('head_joint0','head_joint1'),
}


def check_mount(measured_lower, expected_lower, tolerance_rad=.02):
    actual=np.asarray(measured_lower,dtype=float)
    expected=np.asarray(expected_lower,dtype=float)
    if actual.shape!=(2,) or expected.shape!=(2,) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Two finite lower-waist angles required')
    if not np.isfinite(tolerance_rad) or tolerance_rad<=0:
        raise ValueError('Positive finite mount tolerance required')
    if np.max(np.abs(actual-expected))>tolerance_rad:
        raise ValueError('Lower-waist mounting differs from checkpoint geometry')


def trajectory_commands(target, lower_hold, limits, duration_s=.05):
    """Map upper18 targets to JTC messages, explicitly holding lower waist.

    position limits must include the two non-Actor lower joints. A duration is
    an interpolation horizon, not a proof of velocity/acceleration bounds.
    Caller must rate-limit, check measured state and verify controllers first.
    """
    q=np.asarray(target,dtype=float);lower=np.asarray(lower_hold,dtype=float)
    if q.shape!=(18,) or lower.shape!=(2,) or not np.isfinite(q).all() or not np.isfinite(lower).all():
        raise ValueError('Finite upper18 target and lower-waist hold required')
    if not np.isfinite(duration_s) or not .01<=duration_s<=.1:
        raise ValueError('Trajectory horizon must be between 10 and 100 ms')
    values=dict(zip(JOINT_NAMES,q));values.update(zip(('waist_joint0','waist_joint1'),lower))
    # Validate all groups before constructing any command, so no partial output.
    for names in GROUPS.values():
        for name in names:
            row=limits[name]
            if not row['position_min_rad']<=values[name]<=row['position_max_rad']:
                raise ValueError('Upper-body target outside execution limits: '+name)
    ns=round(duration_s*1e9)
    return {group:{'topic':f'/{group}_controller/joint_trajectory',
        'type':'trajectory_msgs/msg/JointTrajectory',
        'message':{'header':{'stamp':{'sec':0,'nanosec':0},'frame_id':''},
            'joint_names':list(names),'points':[{'positions':[float(values[n]) for n in names],
                'velocities':[],'accelerations':[],'effort':[],
                'time_from_start':{'sec':ns//1_000_000_000,'nanosec':ns%1_000_000_000}}]}}
        for group,names in GROUPS.items()}
