"""Generated-reference and recorded-physics catalog for the workspace."""
import json
from pathlib import Path
import numpy as np
from ..common.paths import ROOT

class GeneratedCatalog:
    def __init__(self,pipeline,geometry):
        self.pipeline=pipeline
        self.geometry=geometry
        self.cache=None

    def get(self, clip=None, frame=0, job=None):
        folder=ROOT/'output/dax_synthetic_5h_20261103'
        if self.cache is None or self.cache[0]!=job:
            if job:
                directory=self.pipeline._dir(job)
                release=directory/'release';published=json.loads((release/'manifest.json').read_text())
                rows=published['clips'];traces=[];references=[];errors=[];entries=[];metrics=[]
                sources={r['clip_id']:r for r in json.loads((ROOT/'data/dax_synthetic_5h_v1_20261103/manifest.json').read_text())['clips']}
                for i,row in enumerate(rows):
                    trace_root=release if published.get('schema')=='dax_generated_release_v2'else directory
                    if published.get('schema')=='dax_generated_release_v2':
                        from ..data_pipeline.records import physical_trace_binding
                        if physical_trace_binding(release,row['physics'],release/row['csv'])!=row['physics']['trace_binding']:
                            raise ValueError('发布物理轨迹完整性绑定不匹配')
                    with np.load(trace_root/row['physics']['physical_npz']) as z:
                        actual=z['q'].copy();error=np.r_[np.zeros((1,2)),z['wrist_error_m']]
                    reference=np.loadtxt(release/row['csv'],delimiter=',')[:,7:];length=min(len(actual),len(reference))
                    traces.append(actual[:length]);references.append(reference[:length]);errors.append(error[:length])
                    source=sources[row['parent_clip']]
                    entries.append({'clip_id':row['candidate_id'],'label':source['label']+' · '+str(row['joint_perturbation_rad'])+'rad / '+str(row['speed_factor'])+'×',
                        'category':source['category'],'variant':i,'frames':length,'duration_s':(length-1)*.02,
                        'max_speed_rad_s':row['timing']['max_speed_rad_s'],'max_acceleration_rad_s2':row['timing']['max_acceleration_rad_s2']})
                    metrics.append(dict(row['physics'],clip_id=row['candidate_id']))
                maximum=max(len(x)for x in traces);count=len(rows)
                arrays={'q':np.zeros((maximum,count,18),dtype=np.float32),'qref':np.zeros((maximum,count,18),dtype=np.float32),
                      'alive':np.zeros((maximum,count),bool),'wrist_error_m':np.zeros((maximum,count,2),dtype=np.float32)}
                for i,(actual,reference,error)in enumerate(zip(traces,references,errors)):
                    n=len(actual);arrays['q'][:n,i]=actual;arrays['qref'][:n,i]=reference;arrays['alive'][:n,i]=True;arrays['wrist_error_m'][:n,i]=error
                manifest={'clips':entries,'job_id':job,'source_clips':900,'screening_profile':published.get('screening_profile'),'training_included':False}
                self.cache=job,manifest,{'clips':metrics},arrays
            else:
                manifest=json.loads((ROOT/'data/dax_synthetic_5h_v1_20261103/manifest.json').read_text())
                metrics=json.loads((folder/'physics_metrics.json').read_text())
                with np.load(folder/'direct_pd_trajectories.npz') as packed:
                    arrays={key:packed[key] for key in ('q','qref','alive','wrist_error_m')}
                self.cache=None,manifest,metrics,arrays
        _,manifest,metrics,z=self.cache
        if clip is None:return {'manifest':manifest,'metrics':metrics}
        if not 0<=clip<len(manifest['clips']):raise ValueError('Unknown generated clip')
        length=int(z['alive'][:,clip].sum());frame=max(0,min(frame,length-1))
        return {'reference':self.geometry(z['qref'][frame,clip]),'physical':self.geometry(z['q'][frame,clip]),
                'frame':frame,'frames':length,'time_s':(frame+1)*.02,'wrist_error_cm':(z['wrist_error_m'][frame,clip]*100).tolist()}
