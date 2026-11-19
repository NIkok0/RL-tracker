"""Policy simulation session and process isolation, independent of HTTP routing."""
import multiprocessing as mp
import uuid
from threading import Lock
import numpy as np

class PolicySession:
    def __init__(self,workspace):
        self.workspace=workspace
        self.sim_lock=Lock()
        self.process=None
        self.pipe=None
        self.session_id=None

    def is_running(self):
        return bool(self.process and self.process.is_alive())

    def request(self,payload):
        operation=payload.get('operation')
        if operation not in {'load','step','reset','perturb','close','status'}:raise ValueError('Invalid simulation operation')
        with self.sim_lock:
            if operation=='close':
                if self.process:
                    self.process.terminate();self.process.join(5)
                    self.pipe.close();self.process=None;self.pipe=None
                return {'running':False}
            if operation=='status':return {'running':bool(self.process and self.process.is_alive())}
            if operation not in {'load','status','close'} and payload.get('session_id') != self.session_id:
                raise ValueError('Simulation session changed; reload before controlling it')
            if operation=='load':
                if payload.get('checkpoint') not in self.workspace.checkpoints():raise ValueError('Choose a current v2 checkpoint')
                if self.workspace.geometry_asset:
                    if self.process:
                        self.process.terminate();self.process.join(5);self.pipe.close();self.process=None;self.pipe=None
                clip=payload.get('clip')
                if clip not in self.workspace.entries:raise ValueError('Choose a train/val reference clip')
                payload=dict(payload,clip_path=str(self.workspace.directory/self.workspace.entries[clip]['file']))
                if 'motion' in payload:payload['samples']=self.workspace.trajectory(payload.pop('motion'))['q']
            if self.process is None or not self.process.is_alive():
                if operation!='load':raise ValueError('Load a checkpoint and reference first')
                from .policy_worker import serve
                context=mp.get_context('spawn');self.pipe,child=context.Pipe()
                self.process=context.Process(target=serve,args=(child,self.workspace.geometry_asset),daemon=True);self.process.start();child.close()
            self.pipe.send(payload)
            if not self.pipe.poll(120 if operation=='load' else 30):
                self.process.terminate();self.process.join(5);self.pipe.close();self.process=None;self.pipe=None
                raise ValueError('Simulation timed out; worker stopped')
            response=self.pipe.recv()
            if 'error' in response:raise ValueError(response['error'])
            if operation=='load':self.session_id=uuid.uuid4().hex
            response['session_id']=self.session_id
            state=response.get('state')
            if state:
                response['geometry']=self.workspace.geometry(state['q'])
                response['reference_geometry']=self.workspace.geometry(state['reference_q'])
                p=np.asarray([b['position'] for b in response['geometry']['bodies']]);t=np.asarray([b['position'] for b in response['reference_geometry']['bodies']])
                response['position_error_mm']=(np.linalg.norm(p-t,axis=1)*1000).tolist()

            return response
