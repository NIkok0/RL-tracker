"""Materialize all qualified augmentations alongside an immutable baseline release."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import shutil

from ..common.paths import ROOT
from ..common.artifacts import atomic,sha
from .records import physical_trace_binding


def build_training_release(index_path,baseline_dir,output_dir,*,project_root=ROOT):
    """Publish train/val directories without changing the baseline or source quotas.

    The index must reference already accepted generation releases. This validates
    file provenance/completeness; it does not rerun geometric or physical screening.
    Failed staging directories are kept for diagnosis and are never overwritten.
    """
    index_path=Path(index_path).resolve();baseline=Path(baseline_dir).resolve()
    output=Path(output_dir).resolve();project_root=Path(project_root).resolve()
    staging=output.with_name(output.name+'.building')
    if output.exists()or staging.exists():raise FileExistsError('Preserve existing release or staging directory')
    index=json.loads(index_path.read_text());additions=index['clips']
    ids=[r['candidate_id']for r in additions]
    if not additions or len(ids)!=len(set(ids))or index.get('count')!=len(ids):raise ValueError('Invalid or duplicate augmentation index')
    if any(not re.fullmatch(r'[0-9a-f]{24}',cid)for cid in ids):raise ValueError('Unsafe candidate ID')
    manifest=json.loads((baseline/'manifest.json').read_text())
    def relative(path):
        path=Path(path).resolve()
        return str(path.relative_to(project_root))if path.is_relative_to(project_root)else str(path)
    staging.mkdir(parents=True);(staging/'train').mkdir();(staging/'val').mkdir();clips=[]
    for row in manifest['clips']:
        source=(baseline/row['file']).resolve();dest=(staging/row['file']).resolve()
        if row['split']not in('train','val')or not source.is_relative_to(baseline)or not dest.is_relative_to(staging/row['split']):
            raise ValueError('Baseline split/path mismatch')
        if sha(source)!=row['sha256']or dest.exists():raise ValueError('Baseline hash mismatch or duplicate file')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        if sha(dest)!=row['sha256']:raise ValueError('Baseline copy hash mismatch')
        clips.append({**row,'inherited_baseline':True,
                      'screening_scope':'historical baseline; not rescreened by augmentation gates'})
    releases={};release_jobs={}
    for row in additions:
        source=Path(row['csv']).resolve();release=source.parent
        if release not in releases:
            released=json.loads((release/'manifest.json').read_text())
            if released['source_bindings']['asset_sha256']!=manifest['geometry_asset_sha256']:raise ValueError('Baseline and augmentation geometry differ')
            if 'joint_names'in manifest and released['joint_names']!=manifest['joint_names']:raise ValueError('Joint order differs')
            release_jobs[release]=released['job_id']
            releases[release]={r['candidate_id']:r for r in released['clips']}
        checked=releases[release][row['candidate_id']]
        if row['validation_job_id']!=release_jobs[release]or row['parent_clip']!=checked['parent_clip']or row['duration_s']!=checked['duration_s']:
            raise ValueError('Augmentation index metadata does not match release')
        if checked['status']!='accepted'or sha(source)!=row['csv_sha256']or row['csv_sha256']!=checked['csv_sha256']:
            raise ValueError('Augmentation is not a matching accepted reference')
        if (release/checked['csv']).resolve()!=source:raise ValueError('Augmentation release path mismatch')
        binding=physical_trace_binding(release,checked['physics'],source)
        if binding!=checked['physics']['trace_binding']:raise ValueError('Physical trace binding mismatch')
        for field in('csv','full20'):
            if sha(release/checked[field])!=checked[field+'_sha256']:raise ValueError('Reference/sidecar changed')
        for receipt in checked['evidence'].values():
            if receipt['passed']is not True or sha(release/receipt['file'])!=receipt['sha256']:raise ValueError('Constraint receipt changed')
        name='aug__'+row['candidate_id'];file='train/'+name+'.csv';dest=staging/file
        if dest.exists():raise ValueError('Duplicate training filename')
        shutil.copyfile(source,dest)
        if sha(dest)!=row['csv_sha256']:raise ValueError('Augmentation copy hash mismatch')
        clips.append({'clip_id':name,'split':'train','source_type':'synthetic_augmentation','file':file,'sha256':row['csv_sha256'],
            'frames':checked['frames'],'duration_s':row['duration_s'],'source_manifest':relative(release/'manifest.json'),
            'source_manifest_sha256':sha(release/'manifest.json'),'candidate_id':row['candidate_id'],'parent_clip':row['parent_clip'],
            'validation_job_id':row['validation_job_id'],'screening_method':row['screening_method'],'inherited_baseline':False,
            'full20_reference':relative(release/checked['full20']),'physical_trace':relative(release/checked['physics']['physical_npz']),
            'physical_trace_sha256':binding['physical_npz_sha256']})
    train=[r for r in clips if r['split']=='train'];val=[r for r in clips if r['split']=='val']
    summary={'train_clips':len(train),'val_clips':len(val),'qualified_augmentations':len(additions),
       'train_by_source':dict(Counter(r['source_type']for r in train)),
       'train_duration_s':sum(r['duration_s']for r in train),'val_duration_s':sum(r['duration_s']for r in val),
       'all_files_hash_verified':len(clips),'path':str(output),'training_started':False}
    final={**manifest,'schema':'dax_mixed_reference_v2','clips':clips,'summary':summary,
        'sampling_contract':'unchanged upstream sampler; all baseline train and all qualified augmentation clips, no source ratios',
        'augmentation_training_index':relative(index_path),'augmentation_training_index_sha256':sha(index_path),
        'previous_manifest':relative(baseline/'manifest.json'),'previous_manifest_sha256':sha(baseline/'manifest.json'),
        'training_started':False}
    shutil.copyfile(baseline/'asset.json',staging/'asset.json')
    if sha(staging/'asset.json')!=manifest['geometry_asset_sha256']:raise ValueError('Geometry asset changed')
    atomic(staging/'manifest.json',final);staging.rename(output)
    return final
