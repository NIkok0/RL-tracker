"""Upstream body pose/velocity FK with DAX named joints and fixed-link roots.

The recursion is unchanged for moving revolute links. DAX's fixed root/ancestors
are retained explicitly, since upstream G1 assumed a free root and moving parents.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import JOINT_NAMES, MJCF_PATH, NUM_J, ROOT_BODY, TRACKED_BODIES
from .rotations import axis_angle_to_mat, quat_to_mat


class DaxKinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.device = device
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        if root < 1:
            raise ValueError('DAX root body is absent from the compiled MJCF')
        jname_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(root, m.nbody):
            # Include only descendants of the physical root, retaining fixed links.
            ancestor = b
            while ancestor and ancestor != root:
                ancestor = int(m.body_parentid[ancestor])
            if ancestor != root:
                continue
            mj_to_local[b] = len(bodies)
            bodies.append(m.body(b).name)
            if b == root:
                parent.append(-1); joint_idx.append(-1)
                axes.append(np.array([1., 0., 0.])); offs_p.append(np.zeros(3)); offs_R.append(np.array([1., 0., 0., 0.]))
                continue
            parent.append(mj_to_local[int(m.body_parentid[b])])
            count = int(m.body_jntnum[b])
            if count > 1:
                raise ValueError('DAX adapter expects at most one revolute joint per body')
            if count:
                jid = int(m.body_jntadr[b])
                if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('DAX upper18 only supports revolute joints')
                if np.linalg.norm(m.jnt_pos[jid]) > 1e-9:
                    raise ValueError('DAX MJCF joint pivot must coincide with the child body origin')
                name = m.joint(jid).name
                if name not in jname_to_idx:
                    raise ValueError('Unexpected active joint in DAX geometry: '+str(name))
                joint_idx.append(jname_to_idx[name]); axes.append(m.jnt_axis[jid].copy())
            else:
                joint_idx.append(-1); axes.append(np.array([1., 0., 0.]))
            offs_p.append(m.body_pos[b].copy()); offs_R.append(m.body_quat[b].copy())
        self.body_names, self.parent, self.joint_idx = bodies, parent, joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)
        lo, hi = np.zeros(NUM_J), np.zeros(NUM_J)
        found = set()
        for j in range(m.njnt):
            name = m.joint(j).name
            if name in jname_to_idx:
                lo[jname_to_idx[name]], hi[jname_to_idx[name]] = m.jnt_range[j]
                found.add(name)
        if found != set(JOINT_NAMES):
            raise ValueError('Compiled MJCF is missing DAX joints')
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """World link-origin poses and linear/angular velocities, as upstream."""
        P, Rs, V, W = ([None] * self.num_bodies for _ in range(4))
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            if j < 0:
                Ri, wi = R_off, W[par]
            else:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                wi = W[par] + axis_w * qd[..., j:j+1]
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (torch.stack([P[i] for i in idx], -2), torch.stack([Rs[i] for i in idx], -3),
                torch.stack([V[i] for i in idx], -2), torch.stack([W[i] for i in idx], -2))
