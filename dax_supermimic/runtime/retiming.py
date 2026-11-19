"""Offline smooth and uniformly retime fixed-base joint references."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_and_retime(
    reference_q,
    initial_q,
    *,
    initial_qdot=None,
    max_velocity=0.12,
    max_acceleration=0.25,
    dt=0.02,
    smoothing_sigma_frames=40.0,
    safety_factor=1.01,
):
    """Gaussian smooth and uniformly dilate a 50 Hz joint path to discrete limits.

    The first state is fixed to ``initial_q``. Limits are checked on the emitted
    sample velocities and finite-difference accelerations, including initial
    velocity transition and final stop. This is a numerical sample contract,
    not a continuous-time or hardware safety certificate.
    """
    q = np.asarray(reference_q, dtype=np.float64)
    q0 = np.asarray(initial_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 18 or len(q) < 3:
        raise ValueError('reference_q must have shape (T>=3, 18)')
    if q0.shape != (18,) or not np.isfinite(q).all() or not np.isfinite(q0).all():
        raise ValueError('finite reference_q and initial_q are required')
    qd0 = np.zeros(18) if initial_qdot is None else np.asarray(initial_qdot, dtype=np.float64)
    if qd0.shape != (18,) or not np.isfinite(qd0).all():
        raise ValueError('initial_qdot must be a finite 18-vector')
    if not np.isfinite([max_velocity, max_acceleration, dt, smoothing_sigma_frames, safety_factor]).all():
        raise ValueError('limits and smoothing parameters must be finite')
    if min(max_velocity, max_acceleration, dt, smoothing_sigma_frames) <= 0 or safety_factor < 1:
        raise ValueError('limits, dt and smoothing sigma must be positive; safety_factor must be >=1')
    if np.max(np.abs(qd0)) > max_velocity:
        raise ValueError('initial velocity already exceeds the requested limit')

    smooth = gaussian_filter1d(q, smoothing_sigma_frames, axis=0, mode='reflect')
    smooth += (q0-smooth[0])[None, :]
    source_index = np.arange(len(smooth), dtype=np.float64)

    # Decelerate measured initial motion within the same acceleration envelope,
    # then shift the smoothed path to the resulting rest pose.
    prefix = [q0.copy()]
    brake_q = q0.copy()
    brake_v = qd0.copy()
    brake_acceleration = max_acceleration/safety_factor
    while np.max(np.abs(brake_v)) > 1e-12:
        brake_v = np.sign(brake_v)*np.maximum(np.abs(brake_v)-brake_acceleration*dt, 0.0)
        brake_q = brake_q + brake_v*dt
        prefix.append(brake_q.copy())
    prefix = np.asarray(prefix)
    smooth += (brake_q-smooth[0])[None, :]

    def sample(scale):
        count = int(np.ceil((len(smooth)-1)*scale))+1
        if count > 200_000:
            raise ValueError('retimed path exceeds 200000 frames')
        index = np.minimum(np.arange(count, dtype=np.float64)/scale, len(smooth)-1)
        result = np.stack([np.interp(index, source_index, smooth[:, j]) for j in range(18)], axis=1)
        result[0] = q0
        result[-1] = smooth[-1]
        complete = np.concatenate((prefix, result[1:]), axis=0)
        velocity = np.diff(complete, axis=0)/dt
        acceleration = np.diff(np.vstack((qd0, velocity, np.zeros((1, 18)))), axis=0)/dt
        vmax = float(np.abs(np.vstack((qd0, velocity))).max())
        amax = float(np.abs(acceleration).max())
        return complete, vmax, amax

    scale = 1.0
    for _ in range(32):
        result, vmax, amax = sample(scale)
        vr = vmax/(max_velocity/safety_factor)
        ar = np.sqrt(amax/(max_acceleration/safety_factor))
        if vr <= 1.0+1e-9 and ar <= 1.0+1e-9:
            return result.astype(np.float32), {
                'smoothing': 'gaussian_filter1d',
                'smoothing_sigma_frames': float(smoothing_sigma_frames),
                'time_scale': float(scale),
                'dt_s': float(dt),
                'input_frames': int(len(q)),
                'output_frames': int(len(result)),
                'duration_s': float((len(result)-1)*dt),
                'max_sample_velocity_rad_s': vmax,
                'max_sample_acceleration_rad_s2': amax,
                'velocity_limit_rad_s': float(max_velocity),
                'acceleration_limit_rad_s2': float(max_acceleration),
                'initial_velocity_rad_s': qd0.tolist(),
                'safety_factor_requested': float(safety_factor),
                'limit_semantics': '50Hz finite-difference samples including initial transition and final stop',
            }
        scale *= max(vr, ar, 1.0)*safety_factor
    raise RuntimeError('could not satisfy sampled velocity and acceleration limits')
