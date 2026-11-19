"""Fixed-waist bilateral motion retargeting with an explicit no-collision policy."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from .robotwin_source import sha256
from ..robot_sim.elbow import ElbowConstraint


def source_fk(urdf, config, q, arm):
    """Geometry-only source FK, before RoboTwin's configurable tool-axis matrix."""
    tree = ET.parse(urdf).getroot()
    by_child = {j.find('child').get('link'): j for j in tree.findall('joint')}
    values = dict(zip(config['arm_joints_name'][arm], q))
    root = config['robot_pose'][0]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([*root[4:7], root[3]]).as_matrix()
    transform[:3, 3] = root[:3]
    child, chain = config['move_group'][arm], []
    while child in by_child:
        joint = by_child[child]
        chain.append(joint)
        child = joint.find('parent').get('link')
    for joint in reversed(chain):
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        offset = np.eye(4)
        offset[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        offset[:3, 3] = xyz
        transform = transform @ offset
        if joint.get('type') in ('revolute', 'continuous'):
            axis = np.fromstring(joint.find('axis').get('xyz', '1 0 0'), sep=' ')
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * values.get(joint.get('name'), 0.)).as_matrix()
            transform = transform @ rotation
    return transform


def resample_source(time_s, poses, output_time_s):
    t, poses, output = np.asarray(time_s), np.asarray(poses), np.asarray(output_time_s)
    if t.ndim != 1 or poses.shape != (len(t), 2, 7) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid source time/pose arrays')
    if output[0] < t[0] - 1e-9 or output[-1] > t[-1] + 1e-9:
        raise ValueError('Cannot extrapolate source motion')
    positions = np.stack([PchipInterpolator(t, poses[:, arm, :3], axis=0)(output) for arm in range(2)], axis=1)
    rotations = np.stack([Slerp(t, Rotation.from_quat(poses[:, arm, [4, 5, 6, 3]]))(output).as_matrix()
                          for arm in range(2)], axis=1)
    return positions, rotations


def map_targets(positions, rotations, center, scale, rotation, tool_rotation, active_arm=None):
    """Map bilateral targets together, or center a single active wrist independently."""
    origin = positions[:, active_arm].mean(axis=0) if active_arm is not None else positions.mean(axis=(0, 1))
    p = np.einsum('ij,taj->tai', rotation, positions - origin) * scale + center
    r = rotation @ rotations @ tool_rotation
    return p, r, origin


class DaxRetarget:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root), cfg
        self.profile = json.loads((self.root / cfg['robot_profile']).read_text())
        self.names = self.profile['joint_names']
        self.m = mujoco.MjModel.from_xml_path(str(self.root / self.profile['kinematics_mjcf']))
        self.d = mujoco.MjData(self.m)
        self.ids = np.asarray([self.m.joint(n).qposadr[0] for n in self.names])
        self.dofs = np.asarray([self.m.joint(n).dofadr[0] for n in self.names])
        self.wrists = [self.m.body(s + '_arm_link6').id for s in ('left', 'right')]
        self.elbows = np.asarray([[self.m.body(s + f'_arm_link{i}').id for i in (0, 3, 6)] for s in ('left', 'right')])
        joints = self.profile['action_interface']['joints']
        self.low = np.asarray([joints[n]['position_min_rad'] for n in self.names])
        self.high = np.asarray([joints[n]['position_max_rad'] for n in self.names])
        self.caps = np.asarray([m['velocity_limit_rad_s'] for m in self.profile['motors']])
        self.fixed = np.asarray([self.profile['default_joint_position_rad'][n] for n in self.names])
        self.active = np.arange(2, 16)
        self.nominal = self.fixed.copy()
        if not cfg.get('use_profile_default_nominal', False):
            self.nominal[2:9] = [-.5, .4, 0, -1.2, 0, .3, 0]
            self.nominal[9:16] = [-.5, -.4, 0, -1.2, 0, .3, 0]

    def initial_seed_bank(self, active_arm=None):
        seeds = [self.nominal.copy()]
        variants = [
            {3: -.45, 5: -.65, 10: .35, 12: .55},
            {3: .40, 5: .60, 10: -.35, 12: -.55},
            {2: .55, 5: -.35, 9: -.55, 12: .35},
            {2: -.55, 5: .35, 9: .55, 12: -.35},
            {6: -.45, 7: .25, 13: .40, 14: -.20},
            {6: .45, 7: -.25, 13: -.40, 14: .20},
            {3: -.30, 5: .65, 10: -.30, 12: -.65},
        ]
        for changes in variants:
            if active_arm == 0:
                changes = {i: v for i, v in changes.items() if 2 <= i <= 8}
            elif active_arm == 1:
                changes = {i: v for i, v in changes.items() if 9 <= i <= 15}
            q = self.nominal.copy()
            for index, delta in changes.items():
                q[index] = np.clip(q[index] + delta, self.low[index] + .01, self.high[index] - .01)
            seeds.append(q)
        unique = []
        for q in seeds:
            if all(np.max(np.abs(q - prior)) > .05 for prior in unique):
                unique.append(q)
        return unique

    def fk(self, q):
        self.d.qpos[self.ids] = q
        mujoco.mj_forward(self.m, self.d)
        return self.d.xpos[self.wrists].copy(), self.d.xmat[self.wrists].reshape(2, 3, 3).copy()

    def elbow_check(self):
        metrics = [ElbowConstraint.metrics(points) for points in self.d.xpos[self.elbows]]
        angles = [r['swivel_from_down_deg'] for r in metrics if r.get('swivel_from_down_deg') is not None]
        guard = self.cfg['elbow_guard_deg']
        if any(abs(angle - 150) <= guard for angle in angles):
            return False, 'unknown_elbow_guard', max(angles, default=0.)
        return all(angle <= 150 for angle in angles), 'elbow_up', max(angles, default=0.)

    def ik(self, p, r, initial, iterations, active_arm=None):
        q = initial.copy()
        q[[0, 1, 16, 17]] = self.fixed[[0, 1, 16, 17]]
        if active_arm is not None:
            held = 1 - active_arm
            q[2 + held * 7:9 + held * 7] = self.nominal[2 + held * 7:9 + held * 7]
        for _ in range(iterations):
            ap, ar = self.fk(q)
            ep = p - ap
            er = Rotation.from_matrix(r @ ar.transpose(0, 2, 1)).as_rotvec()
            if np.linalg.norm(ep, axis=1).max() < .0003 and np.linalg.norm(er, axis=1).max() < np.deg2rad(.2):
                break
            for arm, bid in enumerate(self.wrists):
                if active_arm is not None and arm != active_arm:
                    continue
                indices = np.arange(2 + arm * 7, 9 + arm * 7)
                jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jacBody(self.m, self.d, jp, jr, bid)
                jac = np.concatenate((10 * jp[:, self.dofs[indices]], jr[:, self.dofs[indices]]))
                error = np.r_[10 * ep[arm], er[arm]]
                reg = 1e-5
                dq = np.linalg.solve(jac.T @ jac + reg * np.eye(7), jac.T @ error + reg * (initial[indices] - q[indices]))
                dq *= min(1., .15 / max(np.max(np.abs(dq)), 1e-12))
                q[indices] = np.clip(q[indices] + dq, self.low[indices] + 1e-7, self.high[indices] - 1e-7)
        ap, ar = self.fk(q)
        pe = np.linalg.norm(ap - p, axis=1)
        oe = np.rad2deg(Rotation.from_matrix(r @ ar.transpose(0, 2, 1)).magnitude())
        elbow, reason, swivel = self.elbow_check()
        success = bool(np.max(pe) <= .005 and np.max(oe) <= 3 and elbow)
        return q, success, {'position_error_m': pe.tolist(), 'orientation_error_deg': oe.tolist(),
                            'elbow_passed': bool(elbow), 'max_swivel_deg': swivel,
                            'reason': None if success else reason if not elbow else 'IK_tolerance'}

    def reference_check(self, q, actual_state=False):
        report = {'method': 'joint_step_elbow_no_self_collision_v1',
                  'self_collision': {'enabled': False, 'decision_source': 'user_2026-11-17'},
                  'environment_collision': 'not_certified',
                  'environment_scope': 'DAX robot to static support/table objects is not modelled by this standalone checker',
                  'passed': True, 'status': 'accepted',
                  'checked_samples': 0, 'degenerate_arm_samples': 0, 'max_swivel_deg': 0.}
        if not np.isfinite(q).all() or q.ndim != 2 or q.shape[1] != 18 or len(q) < 2:
            return {**report, 'passed': False, 'status': 'unknown', 'reason': 'invalid_q'}
        fixed_error = np.max(np.abs(q[:, [0, 1, 16, 17]] - self.fixed[[0, 1, 16, 17]]))
        tolerance = 1e-5 if actual_state else 0.
        if ((not actual_state and fixed_error > 1e-7) or np.any(q < self.low-tolerance)
                or np.any(q > self.high+tolerance)):
            return {**report, 'passed': False, 'status': 'rejected', 'reason': 'joint_limits_or_fixed_joints'}
        steps = np.maximum(1, np.ceil(np.max(np.abs(np.diff(q, axis=0)), axis=1) / .01).astype(int))
        if 1 + steps.sum() > 200000:
            return {**report, 'passed': False, 'status': 'unknown', 'reason': 'sample_budget'}
        for i in range(len(q)):
            states = [q[0]] if i == 0 else [q[i-1] + (q[i] - q[i-1]) * k / steps[i-1] for k in range(1, steps[i-1]+1)]
            for state in states:
                self.fk(state)
                ok, reason, angle = self.elbow_check()
                report['checked_samples'] += 1
                report['degenerate_arm_samples'] += sum(ElbowConstraint.metrics(points).get('swivel_from_down_deg') is None
                                                         for points in self.d.xpos[self.elbows])
                report['max_swivel_deg'] = max(report['max_swivel_deg'], angle)
                if not ok:
                    return {**report, 'passed': False, 'status': 'unknown' if reason.startswith('unknown') else 'rejected', 'reason': reason, 'frame': i}
        return report

    def solve_motion(self, time_s, poses, motion_id, initial=None, active_arm=None):
        duration = float(time_s[-1])
        n = int(np.ceil(duration * 50)) + 1
        source_times = np.linspace(0, duration, n)
        positions, rotations = resample_source(time_s, poses, source_times)
        # Source world +Y forward / -X left to DAX +X forward / +Y left.
        base_rotation = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        # Virtual motion tool: DAX wrist -Z follows source tool +X, wrist +X follows source +Y.
        tool = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
        seed = self.nominal if initial is None else np.asarray(initial, dtype=float)
        variants = (self.cfg.get('active_arm_placement_candidates', self.cfg['placement_candidates'])
                    if active_arm is not None else self.cfg['placement_candidates'])
        best, attempts = None, []
        for index, variant in enumerate(variants):
            rotation = Rotation.from_euler('z', variant['yaw_deg'], degrees=True).as_matrix() @ base_rotation
            target_p, target_r, center = map_targets(
                positions, rotations, np.asarray(variant['center_m']), variant['scale'], rotation, tool,
                active_arm=active_arm)
            if active_arm is not None:
                held_p, held_r = self.fk(self.nominal)
                held_arm = 1 - active_arm
                target_p[:, held_arm] = held_p[held_arm]
                target_r[:, held_arm] = held_r[held_arm]
            q, ok, first = self.ik(target_p[0], target_r[0], seed, self.cfg['initial_ik_iterations'], active_arm)
            qpath, diagnostics = [q.copy()], [first]
            if ok:
                for p, r in zip(target_p[1:], target_r[1:]):
                    previous = q.copy()
                    q, ok, d = self.ik(p, r, q, self.cfg['continuous_ik_iterations'], active_arm)
                    diagnostics.append(d)
                    if np.max(np.abs(q - previous)) > .25:
                        ok = False
                        d['reason'] = 'IK_branch_jump'
                    if not ok:
                        break
                    qpath.append(q.copy())
            attempts.append({'variant': variant, 'solved_frames': len(qpath), 'total_frames': n,
                             'passed_IK': ok, 'failure': diagnostics[-1] if not ok else None})
            if not ok:
                continue
            qpath = np.asarray(qpath)
            reference = self.reference_check(qpath)
            if not reference['passed']:
                attempts[-1]['reference_failure'] = reference
                continue
            # Uniform stretching preserves every solved joint knot and both-arm synchronization.
            speed_ratio = float(np.max(np.abs(np.diff(qpath, axis=0)) / self.caps[None, :]) / .02)
            new_duration = max((n-1)*.02, (n-1)*.02*speed_ratio*1.001)
            ticks = int(np.ceil(new_duration/.02 - 1e-10))
            if ticks*.02 > 2*duration + 1e-9:
                attempts[-1]['retime_rejection'] = {'required_duration_s': ticks*.02, 'source_duration_s': duration}
                continue
            output_phase = np.linspace(0, 1, ticks+1)
            qout = np.stack([np.interp(output_phase, np.linspace(0, 1, n), qpath[:, j]) for j in range(18)], axis=1)
            if np.any(np.abs(np.diff(qout, axis=0)) / .02 > self.caps + 1e-6):
                attempts[-1]['resampled_velocity_failure'] = True
                continue
            output_source_t = output_phase * duration
            psrc, rsrc = resample_source(time_s, poses, output_source_t)
            pt, rt, _ = map_targets(
                psrc, rsrc, np.asarray(variant['center_m']), variant['scale'], rotation, tool,
                active_arm=active_arm)
            if active_arm is not None:
                held_p, held_r = self.fk(self.nominal)
                held_arm = 1 - active_arm
                pt[:, held_arm] = held_p[held_arm]
                rt[:, held_arm] = held_r[held_arm]
            actual = [self.fk(q) for q in qout]
            fk_p, fk_r = np.asarray([a[0] for a in actual]), np.asarray([a[1] for a in actual])
            pe = np.linalg.norm(fk_p-pt,axis=-1)
            oe = np.rad2deg(Rotation.from_matrix((rt @ fk_r.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude()).reshape(-1,2)
            if pe.max() > .005 or oe.max() > 3:
                attempts[-1]['resampled_tolerance_failure'] = [float(pe.max()), float(oe.max())]
                continue
            final_screen = self.reference_check(qout)
            if not final_screen['passed']:
                attempts[-1]['resampled_reference_failure'] = final_screen
                continue
            return {'status': 'reference_passed', 'motion_id': motion_id,
                    'mapping': {**variant, 'rotation_matrix': rotation.tolist(), 'source_center_m': center.tolist(),
                                'virtual_tool_rotation': tool.tolist(),
                                'semantics': 'virtual motion frame at source stored endpose and DAX wrist origin; not physical gripper/TCP equivalence'},
                    'source_duration_s': duration, 'motion_duration_s': ticks*.02,
                    'duration_ratio': ticks*.02/duration, 'frames': len(qout),
                    'position_error_max_m': pe.max(axis=0).tolist(), 'orientation_error_max_deg': oe.max(axis=0).tolist(),
                    'velocity_peak_rad_s': (np.abs(np.diff(qout,axis=0))/.02).max(axis=0).tolist(),
                    'reference_screen': final_screen, 'attempts': attempts,
                    '_arrays': {'reference_q': qout, 'source_time_s': output_source_t,
                                'target_wrist_p': pt, 'target_wrist_R': rt,
                                'reference_wrist_p': fk_p, 'reference_wrist_R': fk_r}}
        return {'status': 'unknown_no_solution_within_budget', 'motion_id': motion_id, 'source_duration_s': duration,
                'attempts': attempts, 'reason': 'No complete motion met all fixed mapping/IK/elbow/timing gates within frozen search budget'}

    def solve_motion_multistart(self, time_s, poses, motion_id, active_arm=None, seeds=None):
        candidates = list(seeds) if seeds is not None else self.initial_seed_bank(active_arm)
        search = []
        for seed_index, seed in enumerate(candidates):
            result = self.solve_motion(time_s, poses, motion_id, initial=seed, active_arm=active_arm)
            search.append({'seed_index': seed_index, 'seed_q': np.asarray(seed).tolist(),
                           'status': result['status'], 'attempts': result.get('attempts', [])})
            if result['status'] == 'reference_passed':
                result['initialization_search'] = search
                result['initial_seed_index'] = seed_index
                return result
        return {'status': 'unknown_no_solution_within_budget', 'motion_id': motion_id,
                'source_duration_s': float(np.asarray(time_s)[-1]),
                'reason': 'All frozen initial configurations and common placement candidates failed',
                'initialization_search': search}
