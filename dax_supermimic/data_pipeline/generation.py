"""Smooth augmentation and reference geometry stage orchestration."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from ..common.paths import ROOT
from ..common.artifacts import atomic,sha
from .pipeline import REQUIRED_GATES
from .processing import timing
from .quality import screen_polyline,report_status,pair_diagnostics
from .evidence import write_receipts
from . import workers as _workers
from .workers import worker_init

def smooth_offset(frames,seed,amplitude):
    rng=np.random.default_rng(seed);u=np.linspace(0,1,frames)[:,None]
    frequencies=rng.integers(1,4,(1,18));phase=rng.uniform(-np.pi,np.pi,(1,18))
    wave=np.sin(np.pi*u)**2*(np.sin(2*np.pi*u*frequencies+phase)+.3*np.sin(2*np.pi*u+phase*.7))
    wave/=np.maximum(np.abs(wave).max(axis=0,keepdims=True),1e-12)
    wave[0]=0;wave[-1]=0
    return amplitude*wave


def geometry_worker(task):
    group,recipes,folder,source,limits=task;folder=Path(folder);checker,elbow=_workers._context
    row=recipes[0];p=Path(source)/row['parent_file']
    if sha(p)!=row['parent_sha256']:raise ValueError('Source drift')
    original=np.loadtxt(p,delimiter=',')[:,7:];amp=row['joint_perturbation_rad'];best=None
    # Keep one perturbation waveform per source/level for the two speed counterparts.
    for attempt in range(12):
        delta=smooth_offset(len(original),(row['seed']+attempt)%(2**32),amp);q=original+delta;stats=timing(q)
        best=q,delta,stats,attempt
        if stats['max_speed_rad_s']<=limits['velocity_rad_s'] and stats['max_acceleration_rad_s2']<=limits['acceleration_rad_s2']:break
    q,delta,stats,attempt=best
    directory=folder/'geometry_groups';directory.mkdir(exist_ok=True)
    path=directory/(group+'.npz');np.savez_compressed(path,q=q,delta=delta)
    report=screen_polyline(q,checker,elbow,_workers._screening,_workers._pair_policy)
    report.update(trajectory=str(path.relative_to(folder)),max_perturbation_rad=float(np.abs(delta).max()),
                  timing_at_original_speed=stats,perturbation_attempt=attempt,source_frames=len(q))
    atomic(directory/(group+'.json'),report)
    return group,report


def geometry_adapter(plan,previous,folder,workers=6):
    worker_init(str(ROOT/plan['bindings']['asset']),plan['bindings']['joint_names'],plan['config']['geometry_screening'])
    atomic(folder/'collision_pairs.json',_workers._pair_policy[1])
    grouped={}
    for r in previous['recipes']:grouped.setdefault(r['geometry_group'],[]).append(r)
    reports={};tasks=[(key,rows,str(folder),str(ROOT/'data/dax_synthetic_5h_v1_20261103'),plan['config']['limits'])for key,rows in grouped.items()]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,
          initargs=(str(ROOT/plan['bindings']['asset']),plan['bindings']['joint_names'],plan['config']['geometry_screening']))as pool:
        futures=[pool.submit(geometry_worker,t)for t in tasks]
        for future in as_completed(futures):
            key,report=future.result();reports[key]=report
            if len(reports)%20==0 or len(reports)==len(grouped):
                atomic(folder/'progress.json',{'stage':'geometry','groups_done':len(reports),'groups_total':len(grouped),
                     'geometry_groups_passed':sum(r['passed']for r in reports.values())})
                print('geometry',len(reports),'/',len(grouped),'passed',sum(r['passed']for r in reports.values()),flush=True)
    records=[]
    for recipe in previous['recipes']:
        report=reports[recipe['geometry_group']]
        r={**recipe,'status':report_status(report),'trajectory':report['trajectory'],'geometry_report':report}
        if report['passed']:r['evidence']=write_receipts(folder,plan,r,r['trajectory'],REQUIRED_GATES[:5],report,'geometry')
        else:r.update(reason=report['reason'],failure_category=report['failure_category'])
        records.append(r)
    atomic(folder/'geometry_pair_diagnostics.json',pair_diagnostics(records,['geometry_report']))
    return {'candidates':records,'geometry_groups':len(grouped),'same_waveform_across_speed_levels':True}
