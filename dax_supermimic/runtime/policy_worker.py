"""On-demand isolated Isaac/Newton policy simulation for the local workspace.

No training, dataset writes, model updates or hardware transport. Core evaluator semantics
remain in the frozen modules; workspace noise/perturbation is explicitly diagnostic.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import traceback


def serve(pipe,geometry_asset=None):
    try:
        if geometry_asset:
            from ..training_data import configure_geometry
            from ..robot import ROOT
            configure_geometry(ROOT/geometry_asset)
        import argparse
        import numpy as np
        import torch
        from isaaclab.app import launch_simulation
        from ..env import make_sim, TrackingSim
        from ..kinematics import DaxKinematics
        from ..mdp import MDP, OBS_DIM
        from ..motion import MotionLib
        from ..robot import ROOT, NUM_J, default_joint_pos_list
        from .. import networks
        from ..networks import Policy
        from ..actions import project_action, require_action_contract
        device='cuda:0'
        sim_cfg=make_sim(device,argparse.Namespace())
        with tempfile.TemporaryDirectory(prefix='dax-workspace-') as temp, launch_simulation(sim_cfg,argparse.Namespace()):
            kin=DaxKinematics(device)
            env=None;hist=None;policy=None;checkpoint=None;last=None
            rng=torch.Generator(device=device).manual_seed(123)
            while True:
                try:request=pipe.recv()
                except EOFError:break
                try:
                    op=request['operation']
                    if op=='load':
                        checkpoint=request['checkpoint']
                        ck=torch.load(ROOT/checkpoint,map_location=device,weights_only=False)
                        expected=ck.get('args',{}).get('geometry_asset')
                        actual=str((ROOT/geometry_asset).resolve()) if geometry_asset else None
                        if expected!=actual:raise ValueError('Checkpoint and workspace fixed-mount geometry differ')
                        require_action_contract(ck)
                        a=ck['args'];K,H=int(a['goal_frames']),int(a['hist'])
                        networks.AMP_DTYPE=torch.bfloat16 if a.get('amp','none')=='bf16' else None
                        policy=Policy(OBS_DIM,(15+2*NUM_J)*K,hist=H).to(device)
                        policy.load_state_dict(ck['pi']);policy.eval()
                        path=request['clip_path']
                        if 'samples' in request:
                            q=np.asarray(request['samples'],dtype=float)
                            source=np.loadtxt(path,delimiter=',')
                            raw=np.concatenate((np.repeat(source[:1,:7],len(q),axis=0),q),axis=1)
                            path=str(Path(temp)/'authored.csv');np.savetxt(path,raw,delimiter=',')
                        motion=MotionLib([path],kin,device)
                        mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=device))
                        if env is None:env=TrackingSim(sim_cfg,1,motion,mdp,eval_mode=True)
                        else:env.motion=motion;env.mdp=mdp
                        env.done_eval.zero_();env.failed_eval.zero_();env.reset(torch.tensor([0],device=device),motion.clip_start.clone());env.refresh()
                        hist=None;last=None;step_count=0;command=env.state()[4][0].tolist();samples=[]
                        noise=0.;delay=0;dropout=0.;obs_queue=[];last_obs=None;reward=None;terms={}
                    elif env is None:raise ValueError('No simulation loaded')
                    elif op=='reset':
                        env.done_eval.zero_();env.failed_eval.zero_();env.reset(torch.tensor([0],device=device),motion.clip_start.clone());env.refresh()
                        hist=None;last=None;step_count=0;command=env.state()[4][0].tolist();samples=[];obs_queue=[];last_obs=None;reward=None;terms={}
                    elif op=='perturb':
                        values=np.asarray(request.get('velocity_delta'),dtype=float)
                        if values.shape!=(18,) or not np.isfinite(values).all() or np.abs(values).max()>2:raise ValueError('Perturbation requires 18 finite velocity deltas bounded by 2 rad/s')
                        velocity=env.state()[5]+torch.tensor(values,device=device,dtype=torch.float32)[None,:]
                        env.robot.write_joint_velocity_to_sim_index(velocity=velocity,joint_ids=env.joint_ids,env_ids=torch.tensor([0],device=device));env.refresh()
                    elif op=='step':
                        count=int(request.get('steps',1))
                        if not 1<=count<=10:raise ValueError('Step batch must be 1–10')
                        noise=float(request.get('noise_std',0));dropout=float(request.get('dropout',0));delay=int(request.get('delay_ticks',0))
                        if not np.isfinite([noise,dropout]).all() or not 0<=noise<=.1 or not 0<=dropout<=.5 or not 0<=delay<=10:raise ValueError('Noise/delay/dropout outside debug bounds')
                        samples=[]
                        with torch.no_grad():
                            for _ in range(count):
                                if bool(env.done_eval[0]):break
                                state=env.state();obs=mdp.obs(state,env.a_prev,env.xi,env.f)
                                if noise:obs=obs+torch.randn(obs.shape,device=device,generator=rng)*noise
                                obs_queue.append(obs.clone());obs_queue=obs_queue[-11:]
                                obs=obs_queue[max(0,len(obs_queue)-1-delay)]
                                if last_obs is not None and float(torch.rand((),device=device,generator=rng))<dropout:obs=last_obs
                                last_obs=obs.clone()
                                hist=obs.unsqueeze(1).repeat(1,H,1) if hist is None else torch.cat((hist[:,1:],obs.unsqueeze(1)),1)
                                action=project_action(policy(hist,motion.goal(env.f,K)))
                                command=(env.default_q+env.xi[:,1:1+NUM_J]+env.alpha*action)[0].tolist()
                                out=env.step(action);step_count+=1
                                env.f=torch.minimum(env.f,motion.clip_end-1)
                                reward=float(out['r'][0]);terms={k:float(v[0]) for k,v in out['terms'].items()}
                                f=min(step_count,int(motion.clip_len[0])-1)
                                samples.append({'time_s':step_count*.02,'q':env.state()[4][0].tolist(),'reference_q':motion.q[f].tolist(),'command':command,'reward':reward})
                    state=env.state();f=min(step_count,int(motion.clip_len[0])-1)
                    pipe.send({'state':{'q':state[4][0].tolist(),'qdot':state[5][0].tolist(),'command':command,
                                      'reference_q':motion.q[f].tolist(),'frame':f,'frames':int(motion.clip_len[0]),
                                      'time_s':step_count*.02,'done':bool(env.done_eval[0]),'failed':bool(env.failed_eval[0]),
                                      'checkpoint':checkpoint,'reward':reward,'reward_terms':terms,
                                      'backend':'IsaacLab / Newton / MuJoCo-Warp','diagnostic_observation':{'noise_std':noise,'delay_ticks':delay,'dropout':dropout}},
                               'samples':samples if op=='step' else []})
                except Exception as exc:pipe.send({'error':f'{type(exc).__name__}: {exc}'})
    except Exception as exc:
        traceback.print_exc()
        try:pipe.send({'error':f'Simulation initialization failed: {type(exc).__name__}: {exc}'})
        except (BrokenPipeError,EOFError,OSError):pass
    finally:pipe.close()
