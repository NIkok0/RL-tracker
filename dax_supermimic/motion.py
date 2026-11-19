"""Reference motion library: DAX joint motion at 50 Hz; upstream reference/FK/goal algorithms."""

from __future__ import annotations

import glob
import os
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .kinematics import DaxKinematics
from .robot import JOINT_NAMES

NUM_J = len(JOINT_NAMES)
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


class MotionLib:
    def __init__(self, files: list[str], kin: DaxKinematics, device: str, src_fps: float = 50.0,
                 reset_bin_weight_mode: str = "none", reset_bin_weight_value: float = 1.0,
                 reset_bin_manifest: str | None = None, reset_bin_audit_path: str | None = None,
                 reset_source_type_weights: dict[str, float] | None = None):
        if reset_bin_weight_mode not in ("none", "raw_teleop", "fast_motion", "source_type"):
            raise ValueError("Unknown reset_bin_weight_mode")
        if not math.isfinite(reset_bin_weight_value) or reset_bin_weight_value <= 0:
            raise ValueError("reset_bin_weight_value must be finite and positive")
        self.reset_source_type_weights = dict(reset_source_type_weights or {})
        if any(not isinstance(k, str) or not math.isfinite(v) or v <= 0
               for k, v in self.reset_source_type_weights.items()):
            raise ValueError("Source type weights must be finite and positive")
        self.device = device
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:7 + NUM_J]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            # resample to 50 Hz
            T_src = raw.shape[0]
            dur = (T_src - 1) / src_fps
            ticks = dur / CONTROL_DT
            nearest_tick = np.rint(ticks)
            tick_count = nearest_tick if np.isclose(ticks, nearest_tick, rtol=0.0, atol=1e-8) else np.floor(ticks)
            n = int(tick_count) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            # Reset bins contain transition starts, so exclude the terminal frame.
            nb = int(np.ceil((lens[c] - 1) / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b
            self.frame_bin[self.clip_end[c]] = b
        if reset_bin_weight_mode != "none" or reset_bin_audit_path is not None:
            self._prepare_reset_bin_weights(files, js, bin_clip, bin_start, lens,
                                           reset_bin_weight_mode, reset_bin_weight_value,
                                           reset_bin_manifest, reset_bin_audit_path)

    def _prepare_reset_bin_weights(self, files, joints, bin_clip, bin_start, lens,
                                   mode, value, manifest_path, audit_path):
        """One-time metadata; speeds use adjacent-frame forward differences at 50 Hz."""
        if manifest_path is None:
            raise ValueError("Reset bin weighting/audit requires an explicit train manifest")
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        rows = manifest['clips']
        if any(row.get('split') != 'train' or 'test' in Path(row['file']).parts or
               'capture-d9dbf936d42e2b62b19f' in json.dumps(row) for row in rows):
            raise ValueError("Reset bin manifest contains forbidden input")
        by_path = {(manifest_path.parent / row['file']).resolve(): row for row in rows}
        if set(map(lambda p: Path(p).resolve(), files)) != set(by_path):
            raise ValueError("Reset bin manifest and loaded files differ")
        sources = [by_path[Path(p).resolve()]['source_type'] for p in files]
        arms = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(('left_arm_', 'right_arm_'))]
        if len(arms) != 14:
            raise ValueError("Reset speed classification requires 14 arm joints")
        speed = torch.cat([arm_speed_per_frame(q) for q in joints])
        raw_clip = torch.tensor([s in ('teleop', 'teleop_mirror') for s in sources])
        weights, audit = [], []
        for b, (c, start) in enumerate(zip(bin_clip, bin_start)):
            end = min(start + BIN_FRAMES, sum(lens[:c + 1]) - 1)
            maximum = float(speed[start:end].max())
            weighted = (mode == 'raw_teleop' and bool(raw_clip[c])) or (mode == 'fast_motion' and maximum > 1.0)
            weight = (getattr(self, "reset_source_type_weights", {}).get(sources[c], 1.0)
                      if mode == "source_type" else value if weighted else 1.0)
            weights.append(weight)
            audit.append((b, self.names[c], sources[c], start, end, maximum, weight))
        self.reset_source_types = sources
        self.reset_bin_weight_mode = mode
        self.reset_bin_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.reset_frame_speed = speed.to(self.device)
        self.reset_raw_clip = raw_clip.to(self.device)
        self.reset_sample_counts = torch.zeros(self.num_frames, dtype=torch.long, device=self.device)
        if audit_path is not None:
            with Path(audit_path).open('x', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('bin', 'clip', 'source_type', 'start_frame', 'end_exclusive', 'max_arm_speed_rad_s', 'weight'))
                writer.writerows(audit)

    @staticmethod
    def from_dir(path: str, kin: DaxKinematics, device: str, src_fps: float = 50.0, **kwargs) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return MotionLib(files, kin, device, src_fps=src_fps, **kwargs)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,15 + 2*NUM_J)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def arm_speed_per_frame(q: torch.Tensor) -> torch.Tensor:
    """Outgoing 50 Hz interval speed, max absolute over 14 arms; last repeats last interval."""
    arms = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(('left_arm_', 'right_arm_'))]
    if len(arms) != 14:
        raise ValueError("Speed classification requires 14 arm joints")
    speed = ((q[1:, arms] - q[:-1, arms]) / CONTROL_DT).abs().amax(-1)
    return torch.cat((speed, speed[-1:]))
