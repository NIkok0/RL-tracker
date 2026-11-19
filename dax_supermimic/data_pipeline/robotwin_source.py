"""RoboTwin motion intake with explicit clocks and command/state semantics.

Native collection frames are not necessarily uniformly spaced. A video FPS or
save-frequency field alone is deliberately insufficient to create a motion clock.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile
import zlib

import h5py
import numpy as np


class SourceContractError(ValueError):
    """A source cannot be converted without guessing its interpretation."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RemoteZip:
    """Fetch bounded ZIP members by verified HTTP byte ranges; never extract paths."""

    def __init__(self, url, size, get=None):
        import requests
        self.url, self.size = url, int(size)
        self.get = get or requests.get
        self.receipts = []
        tail_start = max(0, self.size - 65558)
        tail = self.read_range(tail_start, self.size - 1)
        end = tail.rfind(b'PK\x05\x06')
        if end < 0:
            raise SourceContractError('ZIP end record missing')
        sig, disk, cd_disk, count, total, cd_size, cd_offset, comment_len = struct.unpack_from('<4s4H2LH', tail, end)
        if disk or cd_disk or count != total or total == 65535 or cd_size == 0xffffffff:
            raise SourceContractError('Multi-disk and ZIP64 sources require a separate audited reader')
        if cd_offset + cd_size != tail_start + end:
            raise SourceContractError('ZIP central-directory bounds do not match end record')
        if end + 22 + comment_len != len(tail):
            raise SourceContractError('ZIP end record length mismatch')
        data = bytearray(self.read_range(cd_offset, self.size - 1))
        struct.pack_into('<L', data, cd_size + 16, 0)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.members = archive.infolist()
        if len(self.members) != total:
            raise SourceContractError('ZIP member count mismatch')

    def read_range(self, start, end):
        if not 0 <= start <= end < self.size:
            raise SourceContractError('Invalid HTTP byte-range bounds')
        separator = '&' if '?' in self.url else '?'
        response = self.get(self.url + f'{separator}motion_range={start}-{end}',
                            headers={'Range': f'bytes={start}-{end}'}, timeout=60)
        response.raise_for_status()
        expected = f'bytes {start}-{end}/{self.size}'
        if (response.status_code != 206 or response.headers.get('Content-Range') != expected
                or len(response.content) != end - start + 1):
            raise SourceContractError('Server ignored or corrupted a bounded range request')
        self.receipts.append({'start': start, 'end': end, 'bytes': len(response.content),
                              'sha256': hashlib.sha256(response.content).hexdigest()})
        return response.content

    def member_bytes(self, info):
        path = PurePosixPath(info.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
            raise SourceContractError('Unsafe archive member name')
        if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SourceContractError('Encrypted or unsupported ZIP member')
        if info.file_size > 256 * 1024 * 1024 or info.compress_size > 256 * 1024 * 1024:
            raise SourceContractError('Single episode exceeds bounded intake size')
        header = self.read_range(info.header_offset, info.header_offset + 29)
        fields = struct.unpack('<4s5H3L2H', header)
        if fields[0] != b'PK\x03\x04' or fields[3] != info.compress_type:
            raise SourceContractError('ZIP local header disagrees with central record')
        offset = info.header_offset + 30 + fields[-2] + fields[-1]
        compressed = self.read_range(offset, offset + info.compress_size - 1) if info.compress_size else b''
        data = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise SourceContractError('ZIP member CRC or size mismatch')
        return data


def audit_hdf5(path, clock_contract=None):
    """Inspect only motion fields; an optional clock contract must be explicit."""
    path = Path(path)
    errors, arrays, fields = [], {}, []
    pose_paths = {}
    with h5py.File(path, 'r') as file:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                fields.append({'path': name, 'shape': list(obj.shape), 'dtype': str(obj.dtype)})
        file.visititems(visit)
        native = all(f'endpose/{side}_endpose' in file for side in ('left', 'right'))
        xpolicy = all(f'state/{side}_ee_poses' in file for side in ('left', 'right'))
        if not native and not xpolicy:
            errors.append('missing_bilateral_pose_fields')
        for side in ('left', 'right'):
            key = f'endpose/{side}_endpose' if native else f'state/{side}_ee_poses'
            if key not in file:
                continue
            pose_paths[side] = key
            values = file[key][...]
            arrays[side] = values
            if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
                errors.append(side + '_invalid_pose_shape')
            elif not np.isfinite(values).all():
                errors.append(side + '_nonfinite_pose')
            elif np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)) > 1e-3:
                errors.append(side + '_invalid_quaternion_norm')
        n = len(arrays.get('left', []))
        if 'right' in arrays and len(arrays['right']) != n:
            errors.append('bilateral_frame_count_mismatch')
        declared_frequency = None
        if 'additional_info/frequency' in file:
            declared_frequency = np.asarray(file['additional_info/frequency'][()]).tolist()
        temporal_fields = [x['path'] for x in fields
                           if re.search(r'(^|/)(timestamp[^/]*|time[^/]*|step[^/]*|frame_index|sim_step)$', x['path'], re.I)]
        time_s = None
        if clock_contract:
            key = clock_contract.get('field')
            unit = clock_contract.get('unit')
            if not key or key not in file or clock_contract.get('semantics') != 'capture_time':
                errors.append('clock_contract_field_or_semantics_unverified')
            else:
                t = np.asarray(file[key][...], dtype=np.float64)
                factors = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
                if unit not in factors:
                    errors.append('clock_contract_unit_unverified')
                elif t.shape != (n,) or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
                    errors.append('clock_not_finite_strictly_increasing_aligned')
                elif not clock_contract.get('provenance'):
                    errors.append('clock_contract_provenance_missing')
                else:
                    time_s = (t - t[0]) * factors[unit]
        if time_s is None:
            errors.append('source_capture_time_unverified')
    return {
        'schema': 'robotwin_motion_source_audit_v1', 'file': str(path), 'sha256': sha256(path),
        'format': 'native_collection' if native else 'xpolicylab' if xpolicy else 'unknown',
        'frames': n, 'motion_fields': pose_paths, 'hdf5_fields': fields,
        'temporal_fields_found': temporal_fields, 'declared_frequency_metadata': declared_frequency,
        'clock_contract': clock_contract, 'clock_verified': time_s is not None,
        'duration_s': float(time_s[-1]) if time_s is not None else None,
        'pose_semantics': 'source simulator link-pose-derived with embodiment-specific tool transform; not measured robot feedback',
        'joint_semantics': 'native joint_action and converted state joint fields may contain drive targets; never assert feedback from field name',
        'status': 'ready_for_coordinate_audit' if not errors else 'unknown',
        'unknown_reasons': sorted(set(errors)),
    }


def require_verified_clock(audit):
    if audit.get('clock_verified') is not True or audit.get('status') != 'ready_for_coordinate_audit':
        raise SourceContractError('Source capture clock is not verified; 50Hz conversion/IK/PD cannot proceed')


def listing_record(info):
    return {'name': info.filename, 'size': info.file_size, 'compressed_size': info.compress_size,
            'header_offset': info.header_offset, 'crc32': info.CRC, 'compression': info.compress_type}
