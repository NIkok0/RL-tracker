"""Retimed-reference screening, full replay, actual-state checks and receipts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from ..common.paths import ROOT
from ..common.artifacts import atomic
from .pipeline import REQUIRED_GATES
from .processing import make_reference_files
from .physics import simulate_references
from .quality import screen_polyline,report_status,pair_diagnostics
from .evidence import write_receipts
from . import workers as _workers
from .workers import worker_init

def physical_gate_worker(task):
    cid,path=task;checker,elbow=_workers._context
    if Path(path).suffix=='.csv':q=np.loadtxt(path,delimiter=',',ndmin=2)[:,7:]
    else:
        with np.load(path)as z:q=z['q'].copy()
    return cid,screen_polyline(q,checker,elbow,_workers._screening,_workers._pair_policy)


def validate_adapter(plan,previous,folder,workers=6):
    records=make_reference_files(plan,previous,folder)
    import multiprocessing as mp
    reference_tasks=[(r['candidate_id'],str(folder/r['csv']))for r in records if r['status']=='accepted']
    reference_reports={}
    if reference_tasks:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,
             initargs=(str(ROOT/plan['bindings']['asset']),plan['bindings']['joint_names'],plan['config']['geometry_screening']))as pool:
            futures=[pool.submit(physical_gate_worker,t)for t in reference_tasks]
            for future in as_completed(futures):
                cid,report=future.result();reference_reports[cid]=report
                if len(reference_reports)%20==0 or len(reference_reports)==len(reference_tasks):
                    atomic(folder/'progress.json',{'stage':'resampled_geometry','checked':len(reference_reports),'clips':len(reference_tasks)})
        for row in records:
            if row['candidate_id']not in reference_reports:continue
            report=reference_reports[row['candidate_id']];row['resampled_geometry']=report
            if not report['passed']:
                row.update(status=report_status(report),reason='resampled_'+report['reason'],failure_category=report['failure_category'])
    atomic(folder/'resampled_geometry.json',reference_reports)
    # Inventory describes the final reference-qualified set actually sent to physics.
    atomic(folder/'candidate_inventory.json',{'candidates':records,'generated_candidates':len(records),
           'note':'reference screening complete; all accepted references undergo full physics replay'})
    metrics=simulate_references(plan,records,folder)
    import multiprocessing as mp
    tasks=[(cid,str(folder/r['physical_npz']))for cid,r in metrics.items()]
    physical={}
    if tasks:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,
             initargs=(str(ROOT/plan['bindings']['asset']),plan['bindings']['joint_names'],plan['config']['geometry_screening']))as pool:
            futures=[pool.submit(physical_gate_worker,t)for t in tasks]
            for future in as_completed(futures):
                cid,report=future.result();physical[cid]=report
                if len(physical)%20==0 or len(physical)==len(tasks):
                    atomic(folder/'progress.json',{'stage':'actual_state_geometry','checked':len(physical),'clips':len(tasks),
                          'passed':sum(r['passed']for r in physical.values())})
                    print('physical geometry',len(physical),'/',len(tasks),'passed',sum(r['passed']for r in physical.values()),flush=True)
    for row in records:
        if row['status']!='accepted':continue
        m=metrics[row['candidate_id']];gate=physical[row['candidate_id']];row['physics']=m;row['physical_geometry']=gate
        if not m['finite']or m['action_clip_steps']or m['upstream_failure']or not gate['passed']:
            if not m['finite']or m['action_clip_steps']or m['upstream_failure']:
                row.update(status='rejected',reason='rollout_failure',failure_category='confirmed_violation')
            else:
                row.update(status=report_status(gate),reason='actual_state_'+gate['reason'],failure_category=gate['failure_category'])
            continue
        detail={'reference_geometry':row['geometry_report'],'resampled_geometry':row['resampled_geometry'],
             'resampling':'uniform-time linear interpolation + endpoint + optional constant tail hold; resulting path screened again',
             'timing':row['timing'],'physical_state_geometry':gate,'physics':m,
             'screening_profile':plan['config']['screening_profile'],'tracking_criterion':'finite complete direct-PD rollout, no action clipping, no upstream failure; accuracy is diagnostic'}
        row['evidence']=write_receipts(folder,plan,row,row['csv'],REQUIRED_GATES,detail,'validate')
    atomic(folder/'physical_geometry.json',physical)
    atomic(folder/'validation_pair_diagnostics.json',pair_diagnostics(records,['resampled_geometry','physical_geometry']))
    return {'candidates':records,'generated_candidates':len(records),'physics_simulated':len(metrics),
         'admitted':sum(r['status']=='accepted'for r in records),'screening_profile':plan['config']['screening_profile']}
