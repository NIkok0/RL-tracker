"""Reference timing, speed resampling and CSV/full20 materialization."""
import numpy as np
from ..common.artifacts import atomic

def timing(q,dt=.02):
    v=np.diff(q,axis=0)/dt;a=np.diff(v,axis=0)/dt
    return {'max_speed_rad_s':float(np.abs(v).max(initial=0)),
            'max_acceleration_rad_s2':float(np.abs(a).max(initial=0))}


def resample_candidate(q,factor):
    if type(factor)not in(int,float)or not np.isfinite(factor)or not .1<=factor<=3:
        raise ValueError('Speed factor must be finite and between0.1 and3')
    if q.ndim!=2 or q.shape[1]!=18 or len(q)<2 or not np.isfinite(q).all():
        raise ValueError('Expected finite reference trajectory with at least2 frames and18 joints')
    source=np.arange(len(q))
    intervals=int(np.ceil((len(q)-1)/factor))
    times=np.minimum(np.arange(intervals+1)*factor,len(q)-1)
    out=np.stack([np.interp(times,source,q[:,j])for j in range(18)],axis=1)
    # Uniform-time linear interpolation, retaining the endpoint. Noninteger speeds can
    # change chords at source corners, so all retimed samples and segments are screened again.
    loaded=lambda n:int(np.floor(((n-1)/50)/.02))+1
    extra=0
    while loaded(len(out))%50==1:out=np.r_[out,out[-1:]];extra+=1
    return out,extra


def make_reference_files(plan,previous,folder):
    directory=folder/'candidates';directory.mkdir(exist_ok=True);records=[]
    for row in previous['candidates']:
        with np.load(folder/row['trajectory'])as z:q=z['q'].copy()
        samples,hold=resample_candidate(q,row['speed_factor']);csv=directory/(row['candidate_id']+'.csv')
        raw=np.zeros((len(samples),25));raw[:,6]=1;raw[:,7:]=samples;np.savetxt(csv,raw,delimiter=',',fmt='%.17g')
        side=directory/(row['candidate_id']+'.npz');np.savez_compressed(side,
             joint_names=np.array(['waist_joint0','waist_joint1']+plan['bindings']['joint_names']),
             reference_q=np.c_[np.tile(plan['bindings']['fixed_lower_rad'],(len(samples),1)),samples],time_s=np.arange(len(samples))*.02)
        stats=timing(samples);r={**row,'csv':str(csv.relative_to(folder)),'full20':str(side.relative_to(folder)),
             'frames':len(samples),'duration_s':(len(samples)-1)*.02,'tail_hold_frames':hold,'timing':stats,
             'reference_kind':'synthetic_reference_not_measured_feedback','evidence':{}}
        if row['status']=='accepted'and(stats['max_speed_rad_s']>plan['config']['limits']['velocity_rad_s']+1e-9 or stats['max_acceleration_rad_s2']>plan['config']['limits']['acceleration_rad_s2']+1e-9):
            r.update(status='rejected',reason='timing_limits',failure_category='confirmed_violation')
        records.append(r)
    atomic(folder/'candidate_inventory.json',{'candidates':records,'generated_candidates':len(records),
          'note':'every requested CSV exists; rejected references are quarantined, not training-qualified'})
    return records
