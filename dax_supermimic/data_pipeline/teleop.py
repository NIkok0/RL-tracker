"""Publish DAX request joint motion in SuperMimic's reference CSV convention.

Only train/val are read by default. Feedback/state and original timestamps stay
in the hash-bound source payload; they are never substituted for requested q.
The loader computes velocities, FK and future goals with upstream algorithms.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from ..robot import JOINT_NAMES

from ..common.paths import ROOT
DEFAULT_SOURCE = ROOT / 'data/dax_wbc_teleop_v1_20261030'
DEFAULT_OUTPUT = ROOT / 'data/dax_supermimic_v1_20261102'
SCHEMA = 'dax_supermimic_motion_release_v1'
PROFILE = 'upstream_motion_tracking'
UPSTREAM_COMMIT = 'a1508d42c9d1699a52e8233ab16d08c488441ab9'
DT_NS = 20_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_member(directory: Path, name: str) -> Path:
    require(isinstance(name, str), 'Source member must have a named path')
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), 'Source path escapes release')
    require(path.is_file(), 'Missing source payload: ' + str(path))
    return path


def joint_limits():
    """Only robot adaptation: read named angular position limits from our URDF."""
    path = ROOT / 'inputs/body.identified.urdf'
    root = ET.parse(path).getroot()
    by_name = {j.get('name'): j for j in root.findall('joint')}
    limits = []
    for name in JOINT_NAMES:
        require(name in by_name, 'Joint absent from body URDF: ' + name)
        limit = by_name[name].find('limit')
        require(limit is not None, 'Joint has no URDF limits: ' + name)
        limits.append((float(limit.get('lower')), float(limit.get('upper'))))
    return path, np.asarray(limits, dtype=np.float64)


def export(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, *, splits=('train', 'val'), include_test=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    splits = tuple(dict.fromkeys(splits))
    require(splits and set(splits) <= {'train', 'val', 'test'}, 'Expected named train/val/test splits')
    require('test' not in splits or include_test, 'Test export requires explicit --include-test')
    require(not output.exists(), 'Refuse to overwrite an existing motion release')
    staging = output.with_name(output.name + '.building')
    require(not staging.exists(), 'Previous staging exists; preserve it and choose another output')
    manifest_path = source / 'manifest.json'
    src = json.loads(manifest_path.read_text())
    require(src.get('schema') == 'o6_dax_wbc_motion_v1', 'Unsupported source release')
    require(list(src.get('joint_names', [])) == list(JOINT_NAMES), 'Named source/robot joint order mismatch')
    require(src.get('state_semantics') == 'current_joint_position', 'Source feedback semantics mismatch')
    require(src.get('command_semantics') in ('requested_position', 'requested_joint_position'), 'Source request semantics mismatch')
    require(src.get('control_dt_s') == .02, 'Source must preserve the published 50 Hz grid')
    require(src.get('time', {}).get('retiming') is False, 'Retimed source is not the original motion baseline')
    require(src.get('time', {}).get('clipping') is False, 'Clipped source is not the original request baseline')
    rows = [r for r in src['clips'] if r['split'] in splits]
    require(rows and all(any(r['split'] == s for r in rows) for s in splits), 'Requested split has no clips')
    ids = [r['clip_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate clip IDs')
    urdf, limits = joint_limits()
    staging.mkdir(parents=True)
    for split in splits:
        (staging / split).mkdir()
    entries, counts, frames = [], Counter(), Counter()
    for row in rows:
        clip_id = row['clip_id']
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', clip_id) is not None, 'Unsafe clip ID')
        payload = source_member(source, row['file'])
        require(sha256(payload) == row['sha256'], 'Source payload SHA256 mismatch: ' + clip_id)
        with np.load(payload, allow_pickle=False) as arrays:
            required = {'requested_q', 'state_q', 'time_ns', 'source_state', 'source_action',
                        'source_time_ns', 'source_row', 'source_left_row', 'source_right_row',
                        'interpolation_alpha', 'recorder_time_ns'}
            require(required <= set(arrays.files), 'Missing request/state/time provenance fields')
            q = arrays['requested_q'].copy()
            state = arrays['state_q']
            time = arrays['time_ns']
            require(q.shape == (int(row['frames']), len(JOINT_NAMES)), 'Request shape mismatch')
            require(state.shape == q.shape and np.isfinite(state).all(), 'Feedback shape/finite mismatch')
            require(np.isfinite(q).all() and len(q) >= 2, 'Reference needs finite positions and two frames')
            require(time.shape == (len(q),) and time[0] == 0 and
                    np.all(np.diff(time.astype(np.int64)) == DT_NS), 'Nonuniform source control grid')
            require(np.all(np.diff(arrays['source_time_ns'].astype(np.int64)) > 0), 'Original source time must increase')
            require(arrays['source_state'].shape == arrays['source_action'].shape,
                    'Original state/action provenance shape mismatch')
            provenance_shapes = {name: list(arrays[name].shape) for name in sorted(required)}
            source_start_ns, source_end_ns = int(arrays['source_time_ns'][0]), int(arrays['source_time_ns'][-1])
        csv = np.zeros((len(q), 7 + len(JOINT_NAMES)), dtype=np.float64)
        csv[:, 6] = 1.0  # CSV quaternion is xyzw; fixed robot base is identity.
        csv[:, 7:] = q
        filename = row['split'] + '/' + clip_id + '.csv'
        csv_path = staging / filename
        np.savetxt(csv_path, csv, delimiter=',', fmt='%.9g')
        # Nine significant digits round-trip every finite IEEE float32 q.
        roundtrip = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        require(np.array_equal(roundtrip[:, 7:], q), 'CSV altered request q values')
        outside = (q < limits[:, 0]) | (q > limits[:, 1])
        per_joint = outside.sum(0)
        sidecar = {
            'schema': 'dax_supermimic_motion_provenance_v1', 'profile': PROFILE,
            'clip_id': clip_id, 'split': row['split'], 'recording_id': row['recording_id'],
            'lineage_group': row['lineage_group'], 'reference_signal': 'requested_q',
            'reference_fps': 50.0, 'joint_names': list(JOINT_NAMES),
            'csv': filename, 'csv_sha256': sha256(csv_path),
            'source_payload': relative(payload), 'source_payload_sha256': row['sha256'],
            'source_fields': provenance_shapes,
            'source_time_range_ns': [source_start_ns, source_end_ns],
            'source_joint_names': src['source_joint_names'],
            'source_clip_metadata': row,
            'root_convention': 'fixed URDF base_link at world origin; xyz_m=0; quaternion_xyzw=0,0,0,1',
            'feedback_usage': 'state_q/source_state remain independent audit fields in the hash-bound source NPZ',
            'time_usage': 'CSV 50Hz grid from source time_ns; original recorder/source times and row interpolation stay in source NPZ',
            'reference_velocity': 'computed by unchanged upstream central/endpoint differences; not measured dynamics',
            'processing': {'retiming': False, 'clipping': False, 'filtering': False,
                           'resampling_here': False, 'source_resampling': src['time']['resampling']},
            'joint_limit_evidence': {'urdf': relative(urdf), 'urdf_sha256': sha256(urdf),
                                    'outside_rows': int(outside.any(-1).sum()),
                                    'outside_values': int(outside.sum()),
                                    'outside_by_joint': {n: int(c) for n, c in zip(JOINT_NAMES, per_joint) if c}},
        }
        sidecar_name = row['split'] + '/' + clip_id + '.provenance.json'
        sidecar_path = staging / sidecar_name
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + '\n')
        entries.append({'clip_id': clip_id, 'split': row['split'], 'recording_id': row['recording_id'],
                        'lineage_group': row['lineage_group'], 'frames': len(q),
                        'duration_s': (len(q)-1)*.02, 'file': filename, 'sha256': sha256(csv_path),
                        'provenance_file': sidecar_name, 'provenance_sha256': sha256(sidecar_path),
                        'source_payload': relative(payload), 'source_payload_sha256': row['sha256'],
                        'outside_joint_limit_rows': int(outside.any(-1).sum())})
        counts[row['split']] += 1
        frames[row['split']] += len(q)
    result = {
        'schema': SCHEMA, 'profile': PROFILE, 'upstream_commit': UPSTREAM_COMMIT,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_id': output.name, 'reference_signal': 'requested_q',
        'reference_fps': 50.0, 'control_dt_s': .02, 'joint_names': list(JOINT_NAMES),
        'columns': ['root_x_m', 'root_y_m', 'root_z_m', 'root_qx', 'root_qy', 'root_qz', 'root_qw'] + list(JOINT_NAMES),
        'root_convention': 'fixed_base_link_world_origin_identity',
        'source_manifest': relative(manifest_path), 'source_manifest_sha256': sha256(manifest_path),
        'source_dataset_id': src['dataset_id'],
        'urdf': relative(urdf), 'urdf_sha256': sha256(urdf),
        'splits': list(splits), 'split_counts': dict(counts), 'split_frames': dict(frames),
        'test_payload_accessed': 'test' in splits,
        'requested_outside_joint_limit_rows': sum(x['outside_joint_limit_rows'] for x in entries),
        'processing': {'retiming': False, 'clipping': False, 'filtering': False, 'resampling_here': False},
        'loader': {'src_fps': 50.0, 'future_goal': 'upstream future q/qd/root features',
                   'reset_reference': 'same requested q/qd as motion reward and goal',
                   'feedback_source': 'hash-bound source payload only; not substituted into motion'},
        'clips': entries,
        'boundary': 'Reference export only; outside-limit requests retained/reported; no physical or policy acceptance',
    }
    (staging / 'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    staging.rename(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    p.add_argument('--include-test', action='store_true', help='Explicitly authorize reading/exporting test payloads')
    args = p.parse_args()
    result = export(args.source, args.out, splits=args.splits, include_test=args.include_test)
    print(json.dumps({k: result[k] for k in ('dataset_id', 'profile', 'split_counts', 'split_frames',
          'requested_outside_joint_limit_rows', 'test_payload_accessed')}, indent=2))


if __name__ == '__main__':
    main()
