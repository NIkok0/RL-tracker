"""Per-candidate constraint receipts."""
from pathlib import Path
from ..common.artifacts import atomic,sha

def write_receipts(folder,plan,row,trajectory,gates,details,stage):
    folder=Path(folder);directory=folder/'evidence';directory.mkdir(exist_ok=True);receipts={}
    for gate in gates:
        path=directory/(stage+'_'+row['candidate_id']+'_'+gate+'.json')
        atomic(path,{'gate':gate,'candidate_id':row['candidate_id'],'parent_sha256':row['parent_sha256'],
             'trajectory_sha256':sha(folder/trajectory),'asset_sha256':plan['bindings']['asset_sha256'],
             'plan_sha256':sha(folder/'plan.json'),'collision_pairs_sha256':sha(folder/'collision_pairs.json'),
             **({'physical_trace':details['physics']['trace_binding']}if gate in('actual_state_geometry','physics_tracking')else{}),
             'passed':True,'evaluator':'existing_motion_augment_v1','detail':details})
        receipts[gate]={'passed':True,'file':str(path.relative_to(folder)),'sha256':sha(path)}
    return receipts
