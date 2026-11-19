"""Versioned joint-space feasibility checks and local time parameterization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator


@dataclass
class FeasibilityResult:
    position_ok: np.ndarray
    velocity_ok: np.ndarray
    torque_ok: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray
    torque_nm: np.ndarray
    position_utilization: np.ndarray
    velocity_utilization: np.ndarray
    torque_utilization: np.ndarray

    @property
    def feasible(self) -> bool:
        return bool(self.position_ok.all() and self.velocity_ok.all() and self.torque_ok.all())


class FeasibilityModel:
    def __init__(self, config: dict, root: Path):
        self.config = config
        self.root = Path(root)
        self.joints = config["joint_names"]
        self.model = mujoco.MjModel.from_xml_path(str(self.root / config["model_xml"]))
        profile = json.loads((self.root / config["robot_config"]).read_text())
        for motor in profile["motors"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, motor["joint"])
            self.model.dof_armature[self.model.jnt_dofadr[jid]] = motor["armature_kg_m2"]
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.data = mujoco.MjData(self.model)
        self.qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joints])
        self.dofs = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joints])
        execution = json.loads((self.root / config["execution_config"]).read_text())
        self.velocity_limits = np.array([min(execution["joint_urdf_velocity_limits_rad_s"][n],
            execution["training_motor_velocity_limits_rad_s"][n]) for n in self.joints])
        motors = {x["joint"]: x for x in profile["motors"]}
        self.effort_limits = np.array([motors[n]["effort_limit_nm"] for n in self.joints]) * config["effort_margin"]
        action = profile["action_interface"]["joints"]
        self.lower = np.array([action[n]["position_min_rad"] for n in self.joints])
        self.upper = np.array([action[n]["position_max_rad"] for n in self.joints])
        self.dt = 1.0 / config["sample_hz"]

    def check(self, q: np.ndarray, *, sigma: float | None = None) -> FeasibilityResult:
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(self.joints) or len(q) < 2 or not np.isfinite(q).all():
            raise ValueError("q must be finite [frames, joints] with at least two frames")
        sigma = self.config["acceleration_smoothing_sigma_frames"] if sigma is None else sigma
        q_dynamics = gaussian_filter1d(q, sigma=sigma, axis=0, mode="nearest") if sigma else q
        vel = np.gradient(q_dynamics, self.dt, axis=0, edge_order=1)
        acc = np.gradient(vel, self.dt, axis=0, edge_order=1)
        tau = np.empty_like(q)
        for i in range(len(q)):
            self.data.qpos[:] = 0
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qpos[self.qpos] = q_dynamics[i]
            self.data.qvel[self.dofs] = vel[i]
            self.data.qacc[self.dofs] = acc[i]
            mujoco.mj_inverse(self.model, self.data)
            tau[i] = self.data.qfrc_inverse[self.dofs]
        vu = np.abs(vel) / self.velocity_limits
        tu = np.abs(tau) / self.effort_limits
        return FeasibilityResult(
            (q >= self.lower).astype(bool) & (q <= self.upper), vu <= 1 + 1e-9,
            tu <= 1 + 1e-9, vel, acc, tau,
            (q - self.lower) / (self.upper - self.lower), vu, tu)

    def retime(self, q: np.ndarray, *, speed_factor: float = 1.0):
        """Return PCHIP-resampled path, duration ratio, check, output time and source time."""
        if speed_factor < 1 or speed_factor not in [1.0, *self.config["speed_variants"]]:
            raise ValueError("speed_factor must be 1 or a configured acceleration variant")
        src = np.asarray(q, dtype=float)
        if len(src) < 2:
            raise ValueError("retiming requires at least two frames")
        base_t = np.arange(len(src)) * self.dt
        target_t = np.arange(int(np.ceil((base_t[-1] / speed_factor) / self.dt)) + 1) * self.dt
        source_t = np.minimum(target_t * speed_factor, base_t[-1])
        path_spline = PchipInterpolator(base_t, src, axis=0)
        path = path_spline(source_t)
        path[0], path[-1] = src[0], src[-1]
        if speed_factor == 1 and self.check(path, sigma=0).feasible:
            return path, 1.0, self.check(path, sigma=0), target_t, source_t
        smooth = self.config["retime_smoothing_window_s"] * self.config["sample_hz"]
        for _ in range(self.config["max_retime_iterations"]):
            result = self.check(path, sigma=0)
            if result.feasible:
                break
            # Acceleration is diagnostic only. Time scaling responds to velocity and drive torque.
            required = np.max(np.maximum.reduce([np.ones_like(result.velocity_utilization),
                result.velocity_utilization, np.sqrt(result.torque_utilization)]), axis=1)
            required = np.where(required > 1.0, required * 1.01, 1.0)
            k = required.copy()
            if smooth > 0:
                k = gaussian_filter1d(k, smooth, mode="nearest")
            # Smoothing may not erase the per-frame minimum required by the hard limits.
            k = np.maximum(k, required)
            old_clock = target_t
            new_clock = np.r_[0.0, np.cumsum(np.diff(old_clock) * k[1:])]
            target_t = np.arange(int(np.ceil(new_clock[-1] / self.dt)) + 1) * self.dt
            inverse_clock = PchipInterpolator(new_clock, old_clock, extrapolate=True)
            mapped = np.clip(inverse_clock(np.minimum(target_t, new_clock[-1])), old_clock[0], old_clock[-1])
            current_path_spline = PchipInterpolator(old_clock, path, axis=0)
            path = current_path_spline(mapped)
            source_map = PchipInterpolator(old_clock, source_t, extrapolate=True)
            source_t = np.clip(source_map(mapped), 0.0, base_t[-1])
            path[0], path[-1] = src[0], src[-1]
        result = self.check(path, sigma=0)
        return path, (len(path) - 1) / (len(src) - 1), result, target_t, source_t


def legal_segments(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, *,
                   minimum_jump_rad: float = 0.1, mad_multiplier: float = 12.0):
    """Split at out-of-range rows and isolated single-frame position outliers."""
    q = np.asarray(q, dtype=float)
    position_ok = np.isfinite(q).all(axis=1) & (q >= lower).all(axis=1) & (q <= upper).all(axis=1)
    residual = np.zeros_like(q)
    residual[1:-1] = q[1:-1] - (q[:-2] + q[2:]) * 0.5
    center = np.median(np.abs(residual[1:-1]), axis=0)
    mad = np.median(np.abs(np.abs(residual[1:-1]) - center), axis=0)
    threshold = np.maximum(minimum_jump_rad, center + mad_multiplier * 1.4826 * mad)
    outlier_frame = np.zeros(len(q), dtype=bool)
    magnitude = np.abs(residual)
    local_peak = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])
    outlier_frame[1:-1] = ((magnitude[1:-1] > threshold) & local_peak).any(axis=1)
    row_ok = position_ok & ~outlier_frame
    spans = []
    short_rows = []
    start = None
    for frame, ok in enumerate(row_ok):
        if ok and start is None:
            start = frame
        if start is not None and (not ok or frame == len(row_ok) - 1):
            end = frame + 1 if ok and frame == len(row_ok) - 1 else frame
            if end - start >= 2:
                spans.append((start, end))
            else:
                short_rows.append(start)
            start = None
    jumps = []
    for frame in np.flatnonzero(outlier_frame):
        j = int(np.argmax(np.abs(residual[frame]) / threshold))
        jumps.append({"frame": int(frame), "joint_index": j,
                      "single_frame_jump_rad": float(residual[frame, j]),
                      "outlier_threshold_rad": float(threshold[j])})
    return (spans, jumps, np.flatnonzero(~position_ok).tolist(),
            np.flatnonzero(outlier_frame).tolist(), short_rows)


def legacy_1p5(q: np.ndarray, *, hz: float = 50.0) -> dict:
    q = np.asarray(q, dtype=float)
    velocity = np.gradient(q, 1 / hz, axis=0)
    acceleration = np.gradient(velocity, 1 / hz, axis=0)
    return {"velocity_rad_s": velocity, "acceleration_rad_s2": acceleration,
            "feasible": bool((np.abs(acceleration) <= 1.5).all())}


def smooth_reference(q: np.ndarray, sigma_frames: float = 2.0) -> np.ndarray:
    """Apply the configured Gaussian reference filter while preserving endpoints."""
    q = np.asarray(q, dtype=float)
    if sigma_frames < 0:
        raise ValueError("sigma_frames must be non-negative")
    if sigma_frames == 0:
        return q.copy()
    out = gaussian_filter1d(q, sigma=sigma_frames, axis=0, mode="nearest")
    out[0], out[-1] = q[0], q[-1]
    return out
