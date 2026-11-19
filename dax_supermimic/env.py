"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

import math

import torch
import warp as wp

from .actions import project_action

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import (JOINT_NAMES, NUM_J, FIXED_BASE, TORSO_BODY, action_scale,
                    load_drives, load_isaac_robot_cfg)
from .rotations import mat_to_quat, quat_to_mat, so3_exp


def _isaac_runtime():
    # Fail on missing motor data before importing the optional physics runtime.
    load_drives(require_complete=True)
    try:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        from newton import ModelFlags
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass, instantiate, replace
    except ModuleNotFoundError as error:
        raise RuntimeError('ISAACLAB_RUNTIME_REQUIRED: use the pinned upstream Isaac Lab 3 kit-less/Newton environment') from error

    @configclass
    class TrackingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot = replace(load_isaac_robot_cfg(), prim_path="{ENV_REGEX_NS}/Robot")
    return (sim_utils, MJWarpSolverCfg, NewtonCfg, SimulationContext,
            instantiate, TrackingSceneCfg, ModelFlags)


def make_sim(device: str, args):
    sim_utils, MJWarpSolverCfg, NewtonCfg, *_ = _isaac_runtime()
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics)


class TrackingSim:
    """Batched DAX motion-tracking simulator; upstream step/reward/curriculum.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, reset_sampling_mode: str = "original",
                 reset_bin_weight_mode: str = "none", reset_bin_weight_value: float = 1.0):
        if reset_sampling_mode not in ("original", "uniform"):
            raise ValueError("reset_sampling_mode must be original or uniform")
        self.reset_sampling_mode = reset_sampling_mode
        if reset_bin_weight_mode not in ("none", "raw_teleop", "fast_motion", "source_type"):
            raise ValueError("Unknown reset_bin_weight_mode")
        if not math.isfinite(reset_bin_weight_value) or reset_bin_weight_value <= 0:
            raise ValueError("reset_bin_weight_value must be finite and positive")
        if reset_bin_weight_mode != "none" and reset_sampling_mode != "original":
            raise ValueError("Bin weighting requires original adaptive sampling")
        if reset_bin_weight_mode != getattr(motion, "reset_bin_weight_mode", "none"):
            raise ValueError("MotionLib reset bin weights do not match environment mode")
        self.reset_bin_weight_mode = reset_bin_weight_mode
        _, _, _, SimulationContext, instantiate, TrackingSceneCfg, ModelFlags = _isaac_runtime()
        self._ModelFlags = ModelFlags
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True))
        self.sim.reset()
        self.robot = self.scene["robot"]
        if FIXED_BASE and (not self.robot.is_fixed_base or self.robot.num_base_dofs != 0):
            raise ValueError('DAX fixed-base asset unexpectedly has floating root degrees of freedom')
        # USD tree order may put the head before the arms. Preserve the named
        # data/policy order rather than silently permuting DAX joint columns.
        actual_names = list(self.robot.joint_names)
        if len(actual_names) != NUM_J or set(actual_names) != set(JOINT_NAMES):
            raise ValueError('DAX USD joint names differ from the robot/data contract: '+repr(actual_names))
        self.joint_ids = [actual_names.index(name) for name in JOINT_NAMES]
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0, self.joint_ids].clone()
        self.alpha = torch.tensor([action_scale(n) for n in JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies(TORSO_BODY)[0][0]

        # ---- sim context xi = [friction, named joint offsets, torso com offset(3)] ----
        self.xi = torch.zeros(self.N, NUM_J + 4, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 1 + NUM_J:4 + NUM_J] = com
            default_com = self.robot.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, NUM_J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.sim.backend_manager
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if not self.eval_mode else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.sim.backend_manager.add_model_change(self._ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.torch.clone()
        vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.torch[:, self.joint_ids].clone(), d.joint_vel.torch[:, self.joint_ids].clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if getattr(self, "reset_sampling_mode", "original") == "uniform":
            probs = torch.full_like(self.bin_G, 1.0 / m.num_bins)
        else:
            soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
            if getattr(self, "reset_bin_weight_mode", "none") == "none":
                probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
            else:
                probs = 0.5 * m.reset_bin_weights / m.reset_bin_weights.sum() + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        if hasattr(m, "reset_sample_counts"):
            m.reset_sample_counts.index_add_(0, f, torch.ones_like(f))
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        if FIXED_BASE:
            # Robot-only structural adapter: fixed installation has no free-root DOF.
            p, R, v, w = (x.clone() for x in m.ref_state(frames)[:4])
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q, joint_ids=self.joint_ids, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:1 + NUM_J] = torch.empty(n, NUM_J, device=dev).uniform_(-0.01, 0.01)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        a = project_action(a)
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:1 + NUM_J] + self.alpha * a
        self.robot.set_joint_position_target_index(target=target, joint_ids=self.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
