"""Process-local geometry state initialized explicitly by each worker."""
import json
from pathlib import Path
from ..robot_sim.collision import CollisionChecker
from ..robot_sim.elbow import ElbowConstraint
from .quality import screening_config,pair_policy

_context=None
_screening=None
_pair_policy=None

def worker_init(asset_path,joint_names,geometry_screening=None):
    global _context,_screening,_pair_policy
    asset=json.loads(Path(asset_path).read_text())
    checker=CollisionChecker(asset['mjcf'],joint_names)
    _context=checker,ElbowConstraint(checker)
    _screening=screening_config(geometry_screening);_pair_policy=pair_policy(checker,_screening)
