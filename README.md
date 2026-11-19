# RL Tracker

RL Tracker contains the DAX SuperMimic motion-tracking training and data-processing code. The repository focuses on reusable model, simulation, trajectory, evaluation, and training components. Robot-specific assets, datasets, trained checkpoints, runtime logs, and deployment configurations are distributed separately and are not included here.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `pyproject.toml`
- CUDA and a compatible NVIDIA driver for GPU training
- Isaac Lab only for workflows that explicitly use its runtime

Install the project and optional dependencies with:

```bash
python -m pip install -e '.[data,monitoring,onnx]'
```

The core package is under `dax_supermimic/`. Reusable data-processing and feasibility utilities are in `dax_supermimic/data_pipeline/`; runtime inference helpers are in `dax_supermimic/runtime/`. Hardware-specific execution workflows are intentionally outside this public baseline.

## Validation

The public baseline is syntax-checked with Python 3.10 or newer. The original integration tests depend on robot profiles, assets, datasets, and runtime scripts that are distributed separately.

## Upstream

The training implementation is based in part on the pinned SuperMimic upstream snapshot identified in `third_party/UPSTREAM.json`. See the accompanying license files under `third_party/`.
