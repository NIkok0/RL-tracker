"""Legacy recursive screening retained for historical comparisons only."""
import numpy as np
from ..robot_sim.elbow import ElbowConstraint
from .quality import NUMERIC_GUARD_M,failed_check

def collision_failure(distance,location,**details):
    if not np.isfinite(distance):
        return failed_check('collision_nonfinite_'+location,'inspection_incomplete',**details)
    if distance<=NUMERIC_GUARD_M:
        confirmed=distance < -NUMERIC_GUARD_M
        return failed_check(('collision_'if confirmed else'collision_numerical_uncertainty_')+location,
             'confirmed_violation'if confirmed else'numerical_uncertainty',minimum_distance_m=distance,**details)
    return None


def elbow_failure(arms,location,**details):
    if all(r['passed']for r in arms):return None
    confirmed=any(r.get('score_m')is not None and r['score_m'] < -ElbowConstraint.score_guard for r in arms)
    return failed_check(('elbow_'if confirmed else'elbow_numerical_uncertainty_')+location,
          'confirmed_violation'if confirmed else'numerical_uncertainty',arms=arms,**details)


def certify_polyline(q,checker,elbow,max_nodes=100000,max_depth=12):
    """Legacy recursive checker for historical comparison; new jobs use screen_polyline."""
    for endpoint in (q[0],q[-1]):
        collision=checker.posture(endpoint);branch=elbow.posture(endpoint)
        failure=collision_failure(collision['min_distance_lower_bound_m'],'endpoint',detail=collision)
        if failure:return failure
        failure=elbow_failure(branch['arms'],'endpoint')
        if failure:return failure
    nodes=0;minimum=999.;min_score=999.
    for frame in range(len(q)-1):
        stack=[(q[frame],q[frame+1],0)]
        while stack:
            a,b,depth=stack.pop();nodes+=1
            if nodes>max_nodes:return failed_check('unknown_path_budget','inspection_incomplete',frame=frame,nodes=nodes)
            ds=checker.distances((a+b)/2);minimum=min(minimum,float(ds.min()))
            failure=collision_failure(float(ds.min()),'path',frame=frame)
            if failure:return failure
            points=checker.d.xpos[elbow.bodies].copy();metrics=[elbow.metrics(arm)for arm in points]
            failure=elbow_failure(metrics,'path',frame=frame)
            if failure:return failure
            min_score=min(min_score,min(r['score_m']for r in metrics))
            half=np.abs(b-a)/2
            if np.all(ds>checker.bounds@half+1e-6)and elbow.interval_pass(points,elbow.levers@half):continue
            if depth>=max_depth:return failed_check('unknown_path_resolution','inspection_incomplete',frame=frame,nodes=nodes)
            mid=(a+b)/2;stack.extend([(a,mid,depth+1),(mid,b,depth+1)])
    return {'passed':True,'nodes':nodes,'segments':len(q)-1,'min_sampled_distance_lower_bound_m':minimum,
            'min_sampled_elbow_down_score_m':min_score,'model_scope':'linear segments; compiled convex meshes and original48 exclusions',
            'physical_substep_scope':'when used for recorded states, checks interpolated control states, not unrecorded solver substeps'}
