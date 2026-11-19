"""Workspace services composing geometry, catalogs, generation jobs and policy sessions."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import mujoco
from ..common.paths import ROOT
from .reference_preview import ReferencePreview
from ..robot import JOINT_NAMES,TRACKED_BODIES
from ..data_pipeline.pipeline import Pipeline
from .jobs import GenerationJobs
from .catalog import GeneratedCatalog
from ..runtime.session import PolicySession

EDIT_BODIES = ['left_arm_link6','right_arm_link6','left_arm_link3','right_arm_link3','waist_Link3','head_Link1']

def finite_array(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite array of shape {shape}')
    return a


class Workbench(ReferencePreview):
    def __init__(self, directory):
        manifest=json.loads((Path(directory)/'manifest.json').read_text())
        self.geometry_asset=manifest.get('geometry_asset')
        asset=json.loads((ROOT/self.geometry_asset).read_text()) if self.geometry_asset else None
        if asset:
            from ..training_data import sha
            for path,expected in asset['hashes'].items():
                if sha(ROOT/path)!=expected:raise ValueError('Workspace asset hash mismatch: '+path)
        super().__init__(directory,mjcf_path=asset['mjcf'] if asset else None)
        joints=[self.model.joint(n).id for n in JOINT_NAMES]
        self.limits=self.model.jnt_range[joints].copy()
        self.body_ids=[self.model.body(n).id for n in EDIT_BODIES]
        self.generation_pipeline=Pipeline()
        self.policy_session=PolicySession(self)
        self.generation_jobs=GenerationJobs(self.generation_pipeline,self.policy_session.is_running)
        self.generated_catalog=GeneratedCatalog(self.generation_pipeline,self.geometry)

    def geometry(self, q):
        q=finite_array(q,(18,))
        data=mujoco.MjData(self.model)
        data.qpos[self.qids]=q
        mujoco.mj_forward(self.model,data)
        contacts=[{'a':self.model.geom(int(c.geom1)).name,'b':self.model.geom(int(c.geom2)).name,'depth_mm':float(max(0,-c.dist)*1000)} for c in data.contact if c.dist < -0.001]
        return {'q':q.tolist(),'geom_positions':data.geom_xpos[self.geoms].tolist(),
                'geom_rotations':data.geom_xmat[self.geoms].reshape(-1,3,3).tolist(),
                'bodies':[{'name':n,'position':data.xpos[b].tolist(),'rotation':data.xmat[b].reshape(3,3).tolist()} for n,b in zip(EDIT_BODIES,self.body_ids)],
                'outside_limits':np.flatnonzero((q<self.limits[:,0])|(q>self.limits[:,1])).tolist(),
                'contacts':contacts,'scope':'geometric FK only'}

    def state(self, name, frame):
        values=self.clip(name)
        frame=max(0,min(int(frame),len(values)-1))
        return dict(self.geometry(values[frame,7:]),clip=name,frame=frame,frames=len(values),time_s=frame*.02)

    def solve(self, payload):
        """Explicit offline pose-authoring solve; never a policy control fallback."""
        from scipy.optimize import least_squares
        q=finite_array(payload['q'],(18,))
        name=payload['body']
        if name not in EDIT_BODIES:raise ValueError('Unknown tracked body')
        target=finite_array(payload['position'],(3,))
        rotation=payload.get('rotation')
        if rotation is not None:
            rotation=finite_array(rotation,(3,3))
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<0:
                raise ValueError('Rotation must be a proper orthonormal matrix')
        bid=self.model.body(name).id
        data=mujoco.MjData(self.model)
        seed=np.clip(q,self.limits[:,0],self.limits[:,1])
        def residual(x):
            data.qpos[self.qids]=x;mujoco.mj_forward(self.model,data)
            terms=[(data.xpos[bid]-target)*10,(x-seed)*0.015]
            if rotation is not None:terms.append((data.xmat[bid].reshape(3,3)-rotation).ravel()*0.5)
            return np.concatenate(terms)
        result=least_squares(residual,seed,bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=100)
        state=self.geometry(result.x)
        body=next(b for b in state['bodies'] if b['name']==name)
        state['solve']={'position_error_mm':float(np.linalg.norm(np.array(body['position'])-target)*1000),'converged':bool(result.success),
                        'scope':'offline reference pose authoring; no dynamics admission'}
        return state

    def checkpoints(self):
        run=ROOT/'runs/dax_physical_v3_teleop_8h_20261111'
        return [str((run/name).relative_to(ROOT)) for name in ('last.pt','budget_final.pt') if (run/name).is_file()]

    @lru_cache(maxsize=3)
    def feedback_clip(self, name):
        row=self.entries[name]
        if row.get('source_type')=='synthetic':raise ValueError('合成参考没有采集反馈')
        row=row.get('source_clip',row)
        path=(ROOT/row['source_payload']).resolve()
        if not path.is_relative_to((ROOT/'data').resolve()):raise ValueError('Invalid source path')
        with np.load(path,allow_pickle=False) as data:q=data['state_q'].copy()
        if len(q)!=row['frames']:raise ValueError('Feedback frame count differs from reference; alignment unavailable')
        return q

    def feedback(self, name, frame):
        q=self.feedback_clip(name)
        return self.geometry(q[max(0,min(int(frame),len(q)-1))])

    def trajectory(self,payload):
        if payload.get('schema')!='dax_workspace_motion_v1':raise ValueError('Unsupported workspace motion schema')
        if payload.get('joint_names')!=JOINT_NAMES:raise ValueError('Joint order differs from robot')
        rows=payload.get('keyframes',[])
        if not 2<=len(rows)<=500:raise ValueError('Use 2–500 keyframes')
        times=np.asarray([row['time_s'] for row in rows],dtype=float)
        qs=finite_array([row['q'] for row in rows],(len(rows),18))
        if not np.isfinite(times).all() or times[0]!=0 or np.any(np.diff(times)<.02) or times[-1]>300:
            raise ValueError('Times must start at 0, increase by ≥20 ms and end within 300 s')
        if np.any(qs<self.limits[:,0]) or np.any(qs>self.limits[:,1]):raise ValueError('Edited keyframe exceeds joint limits')
        duration=round(float(times[-1])*50)/50
        if abs(duration-times[-1])>1e-6:raise ValueError('Final time must lie on a 20 ms boundary')
        t=np.arange(round(duration*50)+1)*.02
        q=np.stack([np.interp(t,times,qs[:,j]) for j in range(18)],axis=1)
        return {'schema':'dax_workspace_samples_v1','dt_s':.02,'frames':len(q),'duration_s':duration,'q':q.tolist(),
                'max_joint_speed_rad_s':float(np.abs(np.diff(q,axis=0)/.02).max()),
                'scope':'reference interpolation; no dynamic feasibility acceptance'}

    def generation_start(self,payload):
        return self.generation_jobs.start(payload)

    def generated(self,clip=None,frame=0,job=None):
        return self.generated_catalog.get(clip,frame,job)

    def simulation(self,payload):
        return self.policy_session.request(payload)
