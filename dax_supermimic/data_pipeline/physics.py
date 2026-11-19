"""Complete direct-PD physical replay; no training or hardware transport."""
from pathlib import Path
import numpy as np
from ..common.paths import ROOT
from ..common.artifacts import atomic
from .records import physical_trace_binding
from .processing import timing

def simulate_references(plan,records,folder):
    """GPU direct-PD rollout for reference-qualified candidates, no optimizer or checkpoint."""
    import argparse,torch
    from ..training_data import configure_geometry
    configure_geometry(ROOT/plan['bindings']['asset'])
    from isaaclab.app import launch_simulation
    from ..env import TrackingSim,make_sim
    from ..kinematics import DaxKinematics
    from ..motion import MotionLib
    from ..mdp import MDP
    from ..robot import default_joint_pos_list
    from ..actions import project_action
    from ..rotations import mat_to_quat
    selected=[r for r in records if r['status']=='accepted']
    if not selected:return {}
    dev='cuda:0';torch.backends.cuda.matmul.allow_tf32=False
    kin=DaxKinematics(dev);motion=MotionLib([str(folder/r['csv'])for r in selected],kin,dev)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev));cfg=make_sim(dev,argparse.Namespace())
    count=len(selected);trace={k:[]for k in ('q','wrist_error_m','action_clipped')}
    with launch_simulation(cfg,argparse.Namespace()):
        env=TrackingSim(cfg,count,motion,mdp,eval_mode=True,reset_noise=False)
        env.done_eval.zero_();env.failed_eval.zero_();env.reset(torch.arange(count,device=dev),motion.clip_start.clone());env.refresh()
        from ..training_data import verify_runtime_geometry
        atomic(folder/'physics_geometry_check.json',verify_runtime_geometry({'env':env,'kin':kin}))
        initial_q=env.state()[4].cpu().numpy().copy()
        for step in range(int(motion.clip_len.max())-1):
            with torch.no_grad():
                f=torch.minimum(motion.clip_start+step,motion.clip_end-1);env.f=f
                raw=(motion.q[f]-env.default_q-env.xi[:,1:19])/env.alpha;action=project_action(raw);env.step(action)
                state=env.state();positions=kin.forward(*state)[0]
                error=(positions[:,[6,9]]-motion.body_p[f+1][:,[6,9]]).norm(dim=-1)
                vals={'q':state[4],'wrist_error_m':error,'action_clipped':(action!=raw).any(-1)}
                for k,v in vals.items():trace[k].append(v.cpu().numpy())
            if step%100==0:
                atomic(folder/'progress.json',{'stage':'physics','step':step,'max_steps':int(motion.clip_len.max())-1,'clips':count})
                print('physics',step,'/',int(motion.clip_len.max())-1,'clips',count,flush=True)
        failed=env.failed_eval.cpu().tolist()
    arrays={k:np.stack(v)for k,v in trace.items()};directory=folder/'physical_traces';directory.mkdir(exist_ok=True);reports={}
    for i,row in enumerate(selected):
        length=int(motion.clip_len[i])-1;path=directory/(row['candidate_id']+'.npz')
        physical=np.r_[initial_q[i:i+1],arrays['q'][:length,i]]
        np.savez_compressed(path,q=physical,wrist_error_m=arrays['wrist_error_m'][:length,i],time_s=np.arange(length+1)*.02)
        err=arrays['wrist_error_m'][:length,i]
        reports[row['candidate_id']]={'frames':length,'physical_npz':str(path.relative_to(folder)),
             'finite':bool(np.isfinite(physical).all()and np.isfinite(err).all()),'upstream_failure':failed[i],
             'action_clip_steps':int(arrays['action_clipped'][:length,i].sum()),
             'left_wrist_rmse_m':float(np.sqrt(np.mean(err[:,0]**2))),
             'right_wrist_rmse_m':float(np.sqrt(np.mean(err[:,1]**2))),
             'wrist_p95_m':np.quantile(err,.95,axis=0).tolist(),
             'actual_state_timing':timing(physical),
             'tracking_quality':'diagnostic only; no task accuracy/jerk threshold selected',
             'physics_backend':'IsaacLab/Newton/MuJoCo-Warp; original PD and fixed mount'}
    for row in selected:
        m=reports[row['candidate_id']]
        m['trace_binding']=physical_trace_binding(folder,m,folder/row['csv'])
    atomic(folder/'physics_metrics.json',reports)
    return reports
