"""Recover local motion clocks from unique command/path/scheduler matches.

This recovers relative steps inside a primitive, not the complete episode clock.
The 4ms step is a documented source assumption independently checked against
stored planned velocities. It is not a recorded timestamp measurement.
"""
from __future__ import annotations

import io
import pickle

import numpy as np


def load_planned_paths(data):
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Planned path input exceeds the intake limit')

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            from numpy._core.multiarray import _reconstruct
            allowed = {('numpy.core.multiarray', '_reconstruct'): _reconstruct,
                       ('numpy._core.multiarray', '_reconstruct'): _reconstruct,
                       ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
            if (module, name) not in allowed:
                raise ValueError('Unapproved planned-path pickle global: ' + module + '.' + name)
            return allowed[module, name]

    paths = Restricted(io.BytesIO(data)).load()
    if not isinstance(paths, dict) or set(paths) != {'left_joint_path', 'right_joint_path'}:
        raise ValueError('Unknown planned-path structure')
    for values in paths.values():
        if not isinstance(values, list) or len(values) > 256:
            raise ValueError('Invalid planned-path count')
        for path in values:
            if set(path) != {'status', 'position', 'velocity'} or path['status'] != 'Success':
                raise ValueError('Planned-path fields or result are not supported')
            q, qd = path['position'], path['velocity']
            if (not isinstance(q, np.ndarray) or q.dtype.kind != 'f' or q.ndim != 2
                    or q.shape[1] != 6 or not 3 <= len(q) <= 100000
                    or qd.shape != q.shape or qd.dtype.kind != 'f'
                    or not np.isfinite(q).all() or not np.isfinite(qd).all()):
                raise ValueError('Invalid planned position/velocity arrays')
    return paths


def scheduler_pairs(left_length, right_length):
    """Exact index scheduler from RoboTwin together_move_to_pose (no simulation)."""
    if min(left_length, right_length) < 1:
        raise ValueError('Both path lengths must be positive')
    left = right = 0
    applied_left = applied_right = -1
    pairs = []
    while left < left_length or right < right_length:
        if left < left_length and (right >= right_length or left / left_length <= right / right_length):
            applied_left = left
            left += 1
        if right < right_length and (left >= left_length or right / right_length <= left / left_length):
            applied_right = right
            right += 1
        pairs.append((applied_left, applied_right))
    return pairs


def velocity_clock_check(path, dt=.004):
    q, qd = np.asarray(path['position'], dtype=float), np.asarray(path['velocity'], dtype=float)
    dq = (q[2:] - q[:-2]) / 2
    v = qd[1:-1]
    energy = float(np.linalg.norm(v))
    residual = float(np.linalg.norm(dq / dt - v) / max(energy, 1e-12))
    active = np.abs(v) > .05
    median_dt = float(np.median(dq[active] / v[active])) if active.any() else None
    return {'documented_dt_s': dt, 'relative_derivative_residual': residual,
            'median_dq_over_velocity_s': median_dt,
            'passed': energy > .1 and residual <= .10
                      and (median_dt is None or abs(median_dt - dt) <= .0002)}


def unique_path_matches(command, paths, tolerance=1e-7):
    command = np.asarray(command)
    if command.ndim != 2 or command.shape[1] != 6 or not np.isfinite(command).all():
        raise ValueError('Invalid source command matrix')
    if not paths:
        return [None] * len(command)
    values = np.concatenate([p['position'] for p in paths])
    locations = [(segment, i) for segment, path in enumerate(paths) for i in range(len(path['position']))]
    output = []
    for frame in command:
        matches = np.flatnonzero(np.max(np.abs(values - frame), axis=1) <= tolerance)
        output.append(locations[int(matches[0])] if len(matches) == 1 else None)
    return output


def recover_bilateral_clock(commands, poses, grippers, paths, dt=.004):
    """Return only contiguous, open-gripper, uniquely timed bilateral segments."""
    commands, poses, grippers = map(np.asarray, (commands, poses, grippers))
    n = len(commands)
    if (commands.shape != (n, 2, 6) or poses.shape != (n, 2, 7)
            or grippers.shape != (n, 2) or not all(np.isfinite(a).all() for a in (commands, poses, grippers))
            or not .003999 <= dt <= .004001):
        raise ValueError('Motion arrays or documented clock are invalid')
    left = unique_path_matches(commands[:, 0], paths['left_joint_path'])
    right = unique_path_matches(commands[:, 1], paths['right_joint_path'])
    schedules, checks = {}, {}
    frame_records = []
    for frame, (lmatch, rmatch) in enumerate(zip(left, right)):
        if lmatch is None or rmatch is None or np.any(grippers[frame] < .8):
            frame_records.append(None)
            continue
        li, lstep = lmatch
        ri, rstep = rmatch
        key = (li, ri)
        if key not in schedules:
            lp, rp = paths['left_joint_path'][li], paths['right_joint_path'][ri]
            schedule = scheduler_pairs(len(lp['position']), len(rp['position']))
            schedules[key] = {pair: index + 1 for index, pair in enumerate(schedule)}
            checks[key] = [velocity_clock_check(lp, dt), velocity_clock_check(rp, dt)]
        step = schedules[key].get((lstep, rstep))
        if step is None or not all(check['passed'] for check in checks[key]):
            frame_records.append(None)
        else:
            frame_records.append((key, step))
    segments = []
    start = 0
    while start < len(frame_records):
        if frame_records[start] is None:
            start += 1
            continue
        key, _ = frame_records[start]
        end = start + 1
        while (end < len(frame_records) and frame_records[end] is not None
               and frame_records[end][0] == key
               and frame_records[end][1] > frame_records[end - 1][1]):
            end += 1
        steps = np.asarray([x[1] for x in frame_records[start:end]], dtype=np.int64)
        if len(steps) >= 4 and (steps[-1] - steps[0]) * dt >= .4:
            segments.append({'capture_frame_range': [start, end], 'planned_path_ids': list(key),
                             'simulation_steps_local': steps,
                             'time_s': (steps - steps[0]) * dt, 'poses': poses[start:end],
                             'clock_checks': checks[key],
                             'clock_method': 'unique_drive_target_match_to_saved_paths_and_documented_bilateral_scheduler',
                             'clock_scope': 'relative motion primitive; episode gaps and holds not recovered',
                             'dt_source': 'documented source simulator default, independently supported by saved dq/velocity; historical exporter revision not supplied'})
        start = end
    return segments


def recover_arm_independent_clock(commands, poses, grippers, paths, dt=.004):
    """Recover single-arm primitive clocks; preserve source frame order and arm holds.

    Per-arm move arrays are matched independently. A locally held arm is copied
    from the recorded simulator pose. Cross-primitive wall time is not inferred.
    """
    commands, poses, grippers = map(np.asarray, (commands, poses, grippers))
    n = len(commands)
    if (commands.shape != (n, 2, 6) or poses.shape != (n, 2, 7)
            or grippers.shape != (n, 2) or not all(np.isfinite(a).all() for a in (commands, poses, grippers))
            or not .003999 <= dt <= .004001):
        raise ValueError('Motion arrays or documented clock are invalid')
    matches = [unique_path_matches(commands[:, arm], paths[side + '_joint_path'])
               for arm, side in enumerate(('left', 'right'))]
    velocities = [{i: velocity_clock_check(p, dt) for i, p in enumerate(paths[side + '_joint_path'])}
                  for side in ('left', 'right')]
    records = []
    for frame in range(n):
        if np.any(grippers[frame] < .8):
            records.append(None)
            continue
        active = [arm for arm in range(2) if matches[arm][frame] is not None
                  and velocities[arm][matches[arm][frame][0]]['passed']]
        if len(active) != 1:
            records.append(None)
            continue
        arm = active[0]
        plan_id, local_step = matches[arm][frame]
        records.append((arm, plan_id, local_step))
    segments = []
    start = 0
    while start < n:
        if records[start] is None:
            start += 1
            continue
        arm, plan_id, _ = records[start]
        end = start + 1
        while (end < n and records[end] is not None
               and records[end][0] == arm and records[end][1] == plan_id
               and records[end][2] > records[end - 1][2]):
            end += 1
        side = ('left_joint_path', 'right_joint_path')[arm]
        path = paths[side][plan_id]
        steps = np.asarray([row[2] for row in records[start:end]], dtype=np.int64)
        if len(steps) >= 4 and (steps[-1] - steps[0]) * dt >= .4:
            segments.append({'capture_frame_range': [start, end], 'active_arm': ('left', 'right')[arm],
                             'planned_path_id': int(plan_id), 'simulation_steps_local': steps,
                             'time_s': (steps - steps[0]) * dt, 'poses': poses[start:end],
                             'active_joint_path': path['position'][steps].copy(),
                             'clock_check': velocities[arm][plan_id],
                             'clock_method': 'unique_active_arm_drive_target_match_to_saved_path',
                             'clock_scope': 'relative single-arm primitive; stationary arm follows simulator pose; no inter-primitive gap or episode absolute time recovered',
                             'dt_source': 'documented source simulator default, independently supported by saved dq/velocity'})
        start = end
    return segments
