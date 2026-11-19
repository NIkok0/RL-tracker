"""Prepare the DAX anatomical MJCF/USD for the pinned upstream Isaac importer.

The previous MJCF is used only as compiled URDF geometry. Its custom actuators,
legacy tracking sites and controller-related solver overrides are not transferred.
The resulting asset has no actuator; Isaac's explicit robot motor profile supplies
upstream-style implicit drives after all reflected motor inertias are provided.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .robot import MJCF_PATH, ROOT, ROBOT_CONFIG


def build_geometry(output=None):
    config = json.loads(ROBOT_CONFIG.read_text())
    output = Path(output or ROOT / config['runtime_mjcf']).resolve()
    if output.exists():
        raise FileExistsError('Use a new DAX anatomy asset output; refusing overwrite: '+str(output))
    source = Path(MJCF_PATH).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    root.set('model', 'dax_supermimic_upper18')
    for node in list(root):
        if node.tag in ('actuator', 'sensor', 'size', 'visual', 'contact', 'default'):
            root.remove(node)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'site':
                parent.remove(child)
    world = root.find('worldbody')
    for node in list(world):
        if node.tag != 'body':
            world.remove(node)  # The upstream scene owns lights, ground and materials.
    for geom in root.findall('.//geom'):
        for key in ('friction', 'solref', 'solimp'):
            geom.attrib.pop(key, None)
    option = root.find('option')
    if option is not None:
        root.remove(option)
    ET.SubElement(root, 'option', timestep='.005', integrator='implicitfast', gravity='0 0 -9.81', cone='pyramidal')
    for joint in root.findall('.//joint'):
        # DAX motor armature is supplied by Isaac's checked motor profile later.
        joint.attrib.pop('armature', None)
    for mesh in root.findall('.//mesh'):
        value = mesh.get('file')
        if value:
            asset = (source.parent / value).resolve()
            if not asset.is_file():
                raise FileNotFoundError('Missing DAX mesh '+str(asset))
            mesh.set('file', os.path.relpath(asset, output.parent))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output, encoding='unicode')
    # Offline geometry compilation is distinct from a working physics drive.
    import mujoco
    compiled = mujoco.MjModel.from_xml_path(str(output))
    if compiled.nq != len(config['joint_names']) or compiled.nu:
        raise ValueError('DAX anatomical asset must contain upper18 and no actuators')
    report = {'schema': 'dax_supermimic_anatomy_asset_v1', 'source': str(source.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'nq': compiled.nq, 'nv': compiled.nv, 'nu': compiled.nu,
              'fixed_base': True, 'motor_profile': config['profile'], 'ready_for_training': False,
              'scope': 'URDF anatomy only; no custom limiter, servo envelope, gravity feedforward or old actuator gains'}
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def convert_usd(mjcf, output):
    """Same standalone importer procedure as the pinned upstream tool."""
    try:
        from pxr import Plug
        import isaacsim.asset.importer.utils.impl as importer_utils
        Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(importer_utils.__file__), 'schemas'))
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    except ModuleNotFoundError as error:
        raise RuntimeError('PINNED_USD_IMPORTER_REQUIRED: run in the upstream kit-less asset-conversion environment') from error
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite DAX USD asset: '+str(output))
    cfg = MJCFImporterConfig(mjcf_path=str(Path(mjcf).resolve()), usd_path=str(output.resolve()),
                             fix_base=bool(json.loads(ROBOT_CONFIG.read_text())['fixed_base']))
    generated = MJCFImporter(cfg).import_mjcf()
    if not generated or not Path(generated).is_file():
        raise RuntimeError('DAX USD importer did not produce a readable asset: '+str(generated))
    return str(Path(generated).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--convert-usd', action='store_true', help='Convert the existing anatomical MJCF with the pinned standalone importer')
    parser.add_argument('--mjcf', type=Path, default=ROOT / 'models/dax_supermimic/robot.xml')
    args = parser.parse_args()
    if args.convert_usd:
        generated = convert_usd(args.mjcf, args.output or ROOT / 'models/dax_supermimic_usd')
        print(json.dumps({'usd_path': generated, 'fixed_base': True}, indent=2))
    else:
        print(json.dumps(build_geometry(args.output), indent=2))


if __name__ == '__main__':
    main()
