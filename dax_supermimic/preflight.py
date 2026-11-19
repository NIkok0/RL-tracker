"""Check pinned runtime prerequisites separately from DAX assets and motor inputs.

These checks execute small CUDA operations but never construct a DAX simulator,
start training, or touch hardware. A passing environment check is not a physical
acceptance result for the robot adapter.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = 'a1508d42c9d1699a52e8233ab16d08c488441ab9'
ISAACLAB_COMMIT = '46d6cc3d7b115906fad12efd372fbf69f4e3d3ed'
NEWTON_COMMIT = '71eeba16a717d4dcf0b9dd7056a7b6f863054dc3'
REQUIRED_VERSIONS = {
    'torch': '2.11.0+cu130',
    'torchvision': '0.26.0+cu130',
    'warp-lang': '1.17.0',
    'mujoco': '3.15.0',
    'mujoco-warp': '3.15.0',
    'newton': '1.8.0.dev0',
    'isaacsim-asset-isolated': '6.2.0.0',
    'usd-exchange': '3.0.1',
}
MODULES = ('torch', 'mujoco', 'mujoco_warp', 'warp', 'isaaclab',
           'isaaclab_newton', 'newton', 'pxr')


def _error(error):
    return f'{type(error).__name__}: {error}'


def _section(ready=None, **details):
    return {'ready': ready, 'status': 'not_checked' if ready is None else
            ('passed' if ready else 'blocked'), **details}


def _git(checkout, *args):
    return subprocess.run(['git', '-C', str(checkout), *args], check=True,
                          capture_output=True, text=True, timeout=15).stdout.strip()


def _isaaclab_source(module):
    """Verify the checkout actually supplying imports, rather than an expected path."""
    source = Path(module.__file__).resolve()
    for parent in source.parents:
        if (parent / '.git').exists():
            commit = _git(parent, 'rev-parse', 'HEAD')
            dirty = _git(parent, 'status', '--porcelain', '--untracked-files=no')
            return {'path': str(parent), 'commit': commit, 'tracked_changes': dirty,
                    'matches_pin': commit == ISAACLAB_COMMIT and not dirty}
    return {'path': str(source), 'matches_pin': False,
            'reason': 'Imported Isaac Lab has no verifiable source checkout'}


def _newton_source():
    distribution = importlib.metadata.distribution('newton')
    data = json.loads(distribution.read_text('direct_url.json') or '{}')
    commit = data.get('vcs_info', {}).get('commit_id')
    return {'direct_url': data, 'commit': commit, 'matches_pin': commit == NEWTON_COMMIT}


def _warp_increment(values: wp.array(dtype=wp.float32)):
    index = wp.tid()
    values[index] = values[index] + 1.0


def _environment():
    blockers = []
    discovered, loaded = {}, {}
    for name in MODULES:
        try:
            discovered[name] = importlib.util.find_spec(name) is not None
            if discovered[name]:
                loaded[name] = importlib.import_module(name)
            else:
                blockers.append({'component': 'runtime', 'reason': 'missing Python module ' + name})
        except Exception as error:
            discovered[name] = False
            blockers.append({'component': 'runtime', 'reason': f'Cannot import {name}: {_error(error)}'})

    versions = {}
    for name, required in REQUIRED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {'actual': actual, 'required': required, 'matches_pin': actual == required}
        if actual != required:
            blockers.append({'component': 'runtime_version', 'reason': f'{name} must be {required}; found {actual}'})
    if sys.version_info[:2] != (3, 12):
        blockers.append({'component': 'runtime_version', 'reason': 'Pinned Isaac Lab requires Python 3.12'})

    provenance = {}
    for name, probe in (('isaaclab', lambda: _isaaclab_source(loaded['isaaclab'])),
                        ('newton', _newton_source)):
        if name not in loaded:
            continue
        try:
            provenance[name] = probe()
            if not provenance[name]['matches_pin']:
                blockers.append({'component': 'runtime_source', 'reason': f'{name} source does not match the pinned clean revision'})
        except Exception as error:
            provenance[name] = {'matches_pin': False, 'error': _error(error)}
            blockers.append({'component': 'runtime_source', 'reason': f'Cannot verify {name} source: {_error(error)}'})

    probes = {}
    if 'torch' in loaded:
        torch = loaded['torch']
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required; no CPU fallback')
            values = torch.arange(16, dtype=torch.float32, device='cuda:0').reshape(4, 4)
            result = values @ torch.eye(4, device='cuda:0')
            if not torch.equal(result, values):
                raise RuntimeError('CUDA matrix result differs from the expected values')
            torch.cuda.synchronize()
            probes['torch_cuda'] = {'passed': True, 'device': torch.cuda.get_device_name(0),
                                    'torch_cuda_version': torch.version.cuda, 'operation': 'CUDA float32 matrix multiply'}
        except Exception as error:
            probes['torch_cuda'] = {'passed': False, 'error': _error(error)}
            blockers.append({'component': 'cuda', 'reason': 'PyTorch CUDA execution failed: ' + _error(error)})
    if 'warp' in loaded:
        global wp
        wp = loaded['warp']
        try:
            wp.init()
            values = wp.zeros(16, dtype=wp.float32, device='cuda:0')
            wp.launch(wp.kernel(_warp_increment), dim=16, inputs=[values], device='cuda:0')
            wp.synchronize_device('cuda:0')
            if not bool((values.numpy() == 1.0).all()):
                raise RuntimeError('Warp CUDA kernel result differs from the expected values')
            probes['warp_cuda'] = {'passed': True, 'device': 'cuda:0', 'operation': 'Warp CUDA increment kernel'}
        except Exception as error:
            probes['warp_cuda'] = {'passed': False, 'error': _error(error)}
            blockers.append({'component': 'cuda', 'reason': 'Warp CUDA execution failed: ' + _error(error)})

    if 'isaaclab_newton' in loaded:
        try:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
            from isaaclab.sim import SimulationCfg, SimulationContext
            from isaaclab.utils import instantiate
            from isaaclab.app import launch_simulation
            from newton import ModelFlags
            from pxr import Usd, UsdPhysics
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
            probes['upstream_api_imports'] = {'passed': True,
                'scope': 'Newton/MJWarp, SimulationContext, USD and standalone MJCF importer import only'}
        except Exception as error:
            probes['upstream_api_imports'] = {'passed': False, 'error': _error(error)}
            blockers.append({'component': 'runtime_api', 'reason': 'Upstream API imports failed: ' + _error(error)})

    section = _section(not blockers, python=sys.executable, python_version=sys.version.split()[0],
                       modules=discovered, versions=versions, provenance=provenance, probes=probes,
                       acceptance='imports, exact versions/source and CUDA operations only; DAX physical runtime remains separate')
    return section, blockers


def _asset(robot):
    path = Path(robot.USD_PATH)
    blockers = []
    details = {'path': str(path), 'fixed_base_required': robot.FIXED_BASE,
               'physical_acceptance': 'unverified; requires fixed-root and named-joint simulator read/write checks'}
    if not path.is_file():
        blockers.append({'component': 'robot_asset', 'reason': 'converted DAX USD asset is missing', 'path': str(path)})
    else:
        try:
            from pxr import Usd, UsdPhysics
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise ValueError('USD stage could not be opened')
            root = stage.GetDefaultPrim()
            if not root:
                raise ValueError('USD has no default prim')
            variants = root.GetVariantSet('Physics')
            if 'mujoco' not in variants.GetVariantNames():
                raise ValueError('USD has no mujoco Physics variant')
            variants.SetVariantSelection('mujoco')
            names = [prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
            details.update(default_prim=str(root.GetPath()), joint_names=names,
                           joint_names_match=len(names) == robot.NUM_J and set(names) == set(robot.JOINT_NAMES))
            if not details['joint_names_match']:
                raise ValueError('USD revolute joint names differ from the DAX named-joint contract')
            details['acceptance'] = 'USD parses with mujoco variant and expected joint names; static asset check only'
        except Exception as error:
            blockers.append({'component': 'robot_asset', 'reason': 'DAX USD static inspection failed: ' + _error(error), 'path': str(path)})
    return _section(not blockers, **details), blockers


def _drives(robot):
    blockers = []
    missing = []
    action_contract = None
    try:
        config = robot.load_drives(require_complete=False)
        missing = [row['joint'] for row in config['motors'] if row['armature_kg_m2'] is None]
        if missing:
            blockers.append({'component': 'robot_drives',
                             'reason': 'reflected motor armature is missing; upstream gains/action scale cannot be established',
                             'joints': missing})
        if not missing:
            from .actions import validate_action_interface
            action_contract = validate_action_interface()
            import xml.etree.ElementTree as ET
            model=ET.parse(robot.MJCF_PATH).getroot()
            mismatches=[]
            for name,row in config['action_interface']['joints'].items():
                joint=model.find(f".//joint[@name='{name}']")
                values=[] if joint is None else [float(x) for x in joint.get('range','').split()]
                expected=[row['position_min_rad'],row['position_max_rad']]
                if len(values)!=2 or any(abs(a-b)>1e-6 for a,b in zip(values,expected)):
                    mismatches.append({'joint':name,'mjcf_range':values,'action_range':expected})
            if mismatches:
                blockers.append({'component':'action_geometry_contract',
                    'reason':'Action ranges differ from local MJCF; rebuild matching URDF/MJCF/USD and validate before simulation',
                    'mismatches':mismatches})
    except (ValueError, KeyError, OSError) as error:
        blockers.append({'component': 'robot_drives', 'reason': 'Invalid explicit motor profile: ' + _error(error)})
    return _section(not blockers, missing_armature_joints=missing,
                    robot_config=str(robot.ROBOT_CONFIG), action_contract=action_contract,
                    acceptance='explicit input validation only; motor values are never synthesized'), blockers


def check(*, motion_dir=None, scope='all'):
    from . import robot

    if scope not in ('all', 'environment', 'asset', 'drives'):
        raise ValueError('Unsupported preflight scope: ' + str(scope))
    sections = {name: _section() for name in ('environment', 'assets', 'drives', 'motion')}
    blockers = []
    for selected, name, probe in (('environment', 'environment', _environment),
                                  ('asset', 'assets', lambda: _asset(robot)),
                                  ('drives', 'drives', lambda: _drives(robot))):
        if scope in ('all', selected):
            sections[name], failures = probe()
            blockers.extend(failures)
    folder = Path(motion_dir) if motion_dir else ROOT / 'data/dax_supermimic_v1_20261102/train'
    if not folder.is_absolute():
        folder = ROOT / folder
    if scope == 'all':
        found = folder.is_dir() and bool(list(folder.glob('*.csv')))
        sections['motion'] = _section(found, path=str(folder), acceptance='CSV directory presence only')
        if not found:
            blockers.append({'component': 'motion', 'reason': 'exported reference-motion CSV directory is missing', 'path': str(folder)})
    environment = sections['environment']
    return {'schema': 'dax_supermimic_readiness_v2', 'ready': not blockers, 'requested_scope': scope,
            'upstream_commit': UPSTREAM_COMMIT, 'required_isaaclab_commit': ISAACLAB_COMMIT,
            'runtime_modules': environment.get('modules', {}),
            'cuda': environment.get('probes', {}).get('torch_cuda', {}).get('passed', False),
            'sections': sections, 'motion_directory': str(folder), 'blockers': blockers,
            'physical_acceptance': 'unverified by preflight; requires actual DAX fixed-root/named-joint read/write and rollout validation',
            'scope': 'prerequisites only; small CUDA probes, no DAX physics construction, training, download or hardware'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-dir')
    parser.add_argument('--scope', choices=('all', 'environment', 'asset', 'drives'), default='all',
                        help='Check independent prerequisites without inventing missing motor inputs')
    parser.add_argument('--json', action='store_true', help='Compatibility option; output is always JSON')
    args = parser.parse_args()
    # Native dependencies log initialization details; keep stdout machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        result = check(motion_dir=args.motion_dir, scope=args.scope)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    sys.exit(main())
