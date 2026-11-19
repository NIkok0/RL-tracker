"""Full physical-trace integrity checks for validation and publication."""
from pathlib import Path
from ..common.artifacts import sha

def physical_trace_binding(folder,physics,csv):
    """Inspect the entire recorded rollout, using MotionLib's exact frame-count rule."""
    import numpy as np
    folder=Path(folder).resolve()
    path=(folder/physics.get('physical_npz','')).resolve()
    if not path.is_relative_to(folder)or not path.is_file():raise ValueError('物理轨迹缺失/越界')
    source_frames=len(np.loadtxt(csv,delimiter=',',ndmin=2))
    expected=int(np.floor(((source_frames-1)/50)/.02))+1
    with np.load(path,allow_pickle=False)as z:
        if not {'q','wrist_error_m','time_s'}<=set(z.files):raise ValueError('物理轨迹缺少状态/误差/时间轴')
        q,error,t=z['q'],z['wrist_error_m'],z['time_s']
        if q.shape!=(expected,18)or error.shape!=(expected-1,2)or t.shape!=(expected,):
            raise ValueError('物理轨迹未完整回放：帧数不符')
        if not all(np.isfinite(x).all()for x in(q,error,t)):raise ValueError('物理轨迹包含非有限值')
        if not np.allclose(t,np.arange(expected)*.02,rtol=0,atol=1e-9):raise ValueError('物理轨迹时间轴不完整')
    if physics.get('frames')!=expected-1:raise ValueError('物理回放步数记录不符')
    return {'physical_npz_sha256':sha(path),'expected_state_frames':expected,
            'observed_state_frames':expected,'observed_control_steps':expected-1,'control_dt_s':.02}
