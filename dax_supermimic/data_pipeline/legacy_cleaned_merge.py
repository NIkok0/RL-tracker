"""Create an additive release containing legacy-cleaned references by admission tier."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import numpy as np

from ..common.artifacts import sha
from ..common.paths import ROOT


def build_legacy_cleaned_merge(
    legacy_manifest,
    qualified_manifest,
    output_dir,
    *,
    project_root=ROOT,
):
    project_root = Path(project_root).resolve()
    legacy_manifest = Path(legacy_manifest)
    qualified_manifest = Path(qualified_manifest)
    if not legacy_manifest.is_absolute():
        legacy_manifest = project_root / legacy_manifest
    if not qualified_manifest.is_absolute():
        qualified_manifest = project_root / qualified_manifest
    legacy_manifest = legacy_manifest.resolve()
    qualified_manifest = qualified_manifest.resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError("Preserve existing release directory")
    resuming = staging.exists()

    legacy_manifest_sha = sha(legacy_manifest)
    qualified_manifest_sha = sha(qualified_manifest)
    legacy = json.loads(legacy_manifest.read_text())
    qualified = json.loads(qualified_manifest.read_text())
    if legacy.get("schema") != "dax_physical_v3_clean_reference_v1":
        raise ValueError("Legacy source must be the physical-v3 cleaned reference manifest")
    if qualified.get("schema") != "dax_physical_v3_mixed_spatial_20_30s_release_v1":
        raise ValueError("Qualified source must be the independently audited spatial manifest")
    if legacy.get("admission", {}).get("training_admitted") is not False:
        raise ValueError("Legacy cleaned references must remain a separate unadmitted tier")

    robot_rel = "config/dax_supermimic_robot_v3.json"
    execution_rel = "config/dax_full18_execution.json"
    geometry_rel = "output/dax_physical_v3_geometry_20261111_v4/asset.json"
    robot = json.loads((project_root / robot_rel).read_text())
    execution = json.loads((project_root / execution_rel).read_text())
    joint_names = robot["joint_names"]
    if legacy.get("joint_names") != joint_names or qualified.get("joint_names") != joint_names:
        raise ValueError("Joint ordering differs across sources")
    robot_sha = sha(project_root / robot_rel)
    execution_sha = sha(project_root / execution_rel)
    geometry_sha = sha(project_root / geometry_rel)
    if legacy.get("physical_v3_robot_config_sha256") != robot_sha:
        raise ValueError("Legacy robot profile hash mismatch")
    caps = {
        name: min(
            execution["joint_urdf_velocity_limits_rad_s"][name],
            execution["training_motor_velocity_limits_rad_s"][name],
        )
        for name in joint_names
    }
    if legacy.get("velocity_caps_rad_s") != caps:
        raise ValueError("Legacy per-joint caps differ from the current v3 contract")
    if qualified.get("robot_config_sha256") != robot_sha:
        raise ValueError("Qualified robot profile hash mismatch")
    if qualified.get("geometry_asset_sha256") != geometry_sha:
        raise ValueError("Qualified geometry asset hash mismatch")

    limits = robot["action_interface"]["joints"]
    lower = np.array([limits[name]["position_min_rad"] for name in joint_names])
    upper = np.array([limits[name]["position_max_rad"] for name in joint_names])
    cap_array = np.array([caps[name] for name in joint_names])
    clips = []
    ids = set()
    payloads = Counter()
    legacy_types = Counter()
    qualified_types = Counter()
    lineages = Counter()
    legacy_frames = qualified_frames = 0
    legacy_seconds = qualified_reference_seconds = qualified_motion_seconds = 0.0

    def source_file(manifest_path, row):
        path = Path(row["file"])
        return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()

    try:
        if not resuming:
            staging.mkdir(parents=True)
        for directory in ("train", "receipts", "traces"):
            (staging / directory).mkdir(exist_ok=True)
        checkpoint = staging / "legacy_reference_checkpoint.json"
        legacy_verified = False
        if checkpoint.is_file():
            state = json.loads(checkpoint.read_text())
            legacy_verified = (state.get("manifest_sha256") == legacy_manifest_sha and
                              state.get("clip_count") == len(legacy["clips"]) and
                              state.get("position_velocity_recheck_passed") is True)
        if resuming and not legacy_verified:
            # Previous copy reached the qualified-source binding gate after the complete
            # legacy loop. Re-audit every staged reference before using it as a checkpoint.
            for index, row in enumerate(legacy["clips"], 1):
                source = source_file(legacy_manifest, row)
                clip_id = "legacy_cleaned__" + str(row["clip_id"])
                filename = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id) + ".csv"
                target = staging / "train" / filename
                if not source.is_file() or sha(source) != row["sha256"] or not target.is_file() or sha(target) != row["sha256"]:
                    raise ValueError(f"Existing staged legacy payload mismatch: {row['clip_id']}")
                values = np.loadtxt(target, delimiter=",")
                if values.shape != (int(row["frames"]), 25) or not np.isfinite(values).all():
                    raise ValueError(f"Existing staged legacy schema/finite failure: {row['clip_id']}")
                q = values[:, 7:25]
                if np.any(q < lower - 1e-8) or np.any(q > upper + 1e-8):
                    raise ValueError(f"Existing staged legacy position failure: {row['clip_id']}")
                if np.any(np.abs(np.diff(q, axis=0) * float(legacy['reference_fps'])) > cap_array + 1e-5):
                    raise ValueError(f"Existing staged legacy velocity failure: {row['clip_id']}")
                if abs((len(q)-1)/float(legacy['reference_fps'])-float(row['duration_s']))>1e-6:
                    raise ValueError(f"Existing staged legacy timebase failure: {row['clip_id']}")
                if index % 1000 == 0:
                    print('resumed legacy reference audit', index, '/', len(legacy['clips']), flush=True)
            checkpoint.write_text(json.dumps({
                'manifest_sha256': legacy_manifest_sha, 'clip_count': len(legacy['clips']),
                'position_velocity_recheck_passed': True,
                'copied_payload_hashes_passed': True,
                'geometry_direct_pd_status': 'pending',
                'scope': 'all pre-existing staged legacy references matched the source manifest and passed shape, finite, current-v3 position, sampled per-joint velocity and 50Hz duration checks',
            }, indent=2, ensure_ascii=False)+'\n')
            legacy_verified = True
        for row in legacy["clips"]:
            if row.get("split") != "train":
                raise ValueError("Only legacy train clips may be included")
            source = source_file(legacy_manifest, row)
            if not source.is_file() or sha(source) != row["sha256"]:
                raise ValueError(f"Legacy source hash mismatch: {row['clip_id']}")
            q_len = int(row["frames"])
            if not legacy_verified:
                values = np.loadtxt(source, delimiter=",")
                if values.shape != (q_len, 25) or not np.isfinite(values).all():
                    raise ValueError(f"Legacy reference schema/finite failure: {row['clip_id']}")
                q = values[:, 7:25]
                if np.any(q < lower - 1e-8) or np.any(q > upper + 1e-8):
                    raise ValueError(f"Legacy position recheck failed: {row['clip_id']}")
                velocity = np.diff(q, axis=0) * float(legacy["reference_fps"])
                if np.any(np.abs(velocity) > cap_array + 1e-5):
                    raise ValueError(f"Legacy velocity recheck failed: {row['clip_id']}")
                if abs((len(q) - 1) / float(legacy["reference_fps"]) - row["duration_s"]) > 1e-6:
                    raise ValueError(f"Legacy timebase mismatch: {row['clip_id']}")

            original_id = str(row["clip_id"])
            clip_id = "legacy_cleaned__" + original_id
            if clip_id in ids:
                raise ValueError(f"Duplicate clip id: {clip_id}")
            ids.add(clip_id)
            filename = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id) + ".csv"
            relative = "train/" + filename
            target = staging / relative
            if not target.exists():
                shutil.copyfile(source, target)
            if sha(target) != row["sha256"]:
                raise ValueError(f"Legacy copied hash mismatch: {original_id}")

            lineage = row.get("source_lineage") or row.get("source_recording_id")
            copied = {
                **row,
                "clip_id": clip_id,
                "original_clip_id": original_id,
                "file": relative,
                "tier": "legacy_position_velocity_cleaned_only",
                "v3_reference_position_passed": True,
                "v3_reference_per_joint_velocity_passed": True,
                "v3_reference_acceleration_status": "diagnostic_only_not_admitted",
                "v3_sampled_geometry_status": "pending",
                "v3_direct_pd_status": "pending",
                "v3_actual_state_tracking_status": "pending",
                "hardware_execution_status": "not_evidenced",
                "source_manifest": str(legacy_manifest.relative_to(project_root)),
                "source_manifest_sha256": legacy_manifest_sha,
                "source_lineage": lineage,
                "evidence_scope": (
                    "Legacy cleaner passed position and sampled 50Hz per-joint velocity; "
                    "acceleration is diagnostic and v3 geometry/direct-PD were not run."
                ),
            }
            clips.append(copied)
            legacy_frames += q_len
            legacy_seconds += float(row["duration_s"])
            legacy_types[row["source_type"]] += 1
            payloads[row["sha256"]] += 1
            if lineage:
                lineages[str(lineage)] += 1

        bindings = qualified.get("current_v3_bindings") or qualified.get("bindings") or {}
        if bindings:
            for relative, expected in ((robot_rel, robot_sha), (geometry_rel, geometry_sha)):
                if bindings.get(relative) != expected:
                    raise ValueError(f"Qualified release binding mismatch: {relative}")

        for row in qualified["clips"]:
            if row.get("split") != "train":
                raise ValueError("Qualified release contains non-train data")
            source = source_file(qualified_manifest, row)
            if not source.is_file() or sha(source) != row["sha256"]:
                raise ValueError(f"Qualified reference hash mismatch: {row['clip_id']}")
            original_id = str(row["clip_id"])
            clip_id = "v3_qualified__" + original_id
            if clip_id in ids:
                raise ValueError(f"Duplicate clip id: {clip_id}")
            ids.add(clip_id)
            filename = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id) + ".csv"
            relative = "train/" + filename
            shutil.copyfile(source, staging / relative)
            if sha(staging / relative) != row["sha256"]:
                raise ValueError(f"Qualified copied hash mismatch: {original_id}")

            copied = {
                **row,
                "clip_id": clip_id,
                "original_clip_id": original_id,
                "file": relative,
                "tier": "current_v3_geometry_direct_pd_qualified",
                "source_manifest": str(qualified_manifest.relative_to(project_root)),
                "source_manifest_sha256": qualified_manifest_sha,
            }
            for source_key, output_key, folder in (
                ("physical_trace", "physical_trace", "traces"),
                ("receipt", "receipt", "receipts"),
            ):
                artifact = Path(row[source_key])
                artifact = artifact.resolve() if artifact.is_absolute() else (qualified_manifest.parent / artifact).resolve()
                expected = row["physical_trace_sha256"] if source_key == "physical_trace" else row["receipt_sha256"]
                if not artifact.is_file() or sha(artifact) != expected:
                    raise ValueError(f"Qualified {source_key} hash mismatch: {original_id}")
                name = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id) + artifact.suffix
                artifact_relative = folder + "/" + name
                shutil.copyfile(artifact, staging / artifact_relative)
                if sha(staging / artifact_relative) != expected:
                    raise ValueError(f"Copied {source_key} hash mismatch: {original_id}")
                copied[output_key] = artifact_relative

            clips.append(copied)
            qualified_frames += int(row["frames"])
            qualified_reference_seconds += float(row["duration_s"])
            qualified_motion_seconds += float(row["motion_duration_s"])
            qualified_types[row["parent_source_type"]] += 1
            payloads[row["sha256"]] += 1
            if row.get("canonical_root_lineage"):
                lineages[row["canonical_root_lineage"]] += 1

        duplicate_payload_groups = sum(count > 1 for count in payloads.values())
        summary = {
            "train_clips": len(clips),
            "legacy_reference_only_clips": len(legacy["clips"]),
            "current_v3_geometry_direct_pd_qualified_clips": len(qualified["clips"]),
            "legacy_reference_duration_s": legacy_seconds,
            "legacy_reference_duration_hours": legacy_seconds / 3600,
            "current_v3_qualified_reference_duration_s": qualified_reference_seconds,
            "current_v3_qualified_net_motion_s": qualified_motion_seconds,
            "current_v3_qualified_net_motion_hours": qualified_motion_seconds / 3600,
            "combined_reference_duration_s": legacy_seconds + qualified_reference_seconds,
            "combined_reference_duration_hours": (legacy_seconds + qualified_reference_seconds) / 3600,
            "legacy_frame_count": legacy_frames,
            "current_v3_qualified_frame_count": qualified_frames,
            "legacy_source_type_counts": dict(legacy_types),
            "qualified_parent_source_type_counts": dict(qualified_types),
            "source_lineage_count": len(lineages),
            "duplicate_reference_payload_groups": duplicate_payload_groups,
            "training_admitted": False,
            "training_started": False,
        }
        manifest = {
            "schema": "dax_physical_v3_mixed_legacy_reference_and_qualified_spatial_v1",
            "dataset_id": output.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "joint_names": joint_names,
            "reference_fps": float(legacy["reference_fps"]),
            "robot_config": robot_rel,
            "robot_config_sha256": robot_sha,
            "current_execution_config": execution_rel,
            "current_execution_config_sha256": execution_sha,
            "current_geometry_asset": geometry_rel,
            "current_geometry_asset_sha256": geometry_sha,
            "execution_profile_variants": {
                "legacy_cleaned_reference": legacy["physical_v3_execution_config_sha256"],
                "current_v3_qualified_spatial": qualified["execution_config_sha256"],
            },
            "legacy_cleaned_source": {
                "manifest": str(legacy_manifest.relative_to(project_root)),
                "manifest_sha256": legacy_manifest_sha,
                "legacy_geometry_asset": legacy["source_geometry_asset"],
                "legacy_geometry_asset_sha256": legacy["source_geometry_asset_sha256"],
                "admission_tier": "position_and_velocity_cleaned_only_geometry_direct_pd_pending",
            },
            "qualified_spatial_source": {
                "manifest": str(qualified_manifest.relative_to(project_root)),
                "manifest_sha256": qualified_manifest_sha,
                "admission_tier": "current_v3_sampled_geometry_and_direct_pd_per_clip_pass",
            },
            "tier_policy": {
                "legacy_position_velocity_cleaned_only": (
                    "Automatically included at user direction as train reference data; "
                    "not represented as current-v3 geometry/direct-PD or hardware qualified."
                ),
                "current_v3_geometry_direct_pd_qualified": (
                    "Passed current-v3 per-clip sampled geometry and complete direct-PD tracking gates."
                ),
                "training_admitted": False,
                "training_ready": False,
            },
            "summary": summary,
            "clips": clips,
            "training_admitted": False,
            "training_ready": False,
            "training_started": False,
        }
        shutil.copyfile(project_root / geometry_rel, staging / "asset.json")
        if sha(staging / "asset.json") != geometry_sha:
            raise ValueError("Geometry asset copy hash mismatch")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if sha(legacy_manifest) != legacy_manifest_sha or sha(qualified_manifest) != qualified_manifest_sha:
            raise ValueError("Source manifest changed during merge")
        staging.rename(output)
        return manifest
    except Exception:
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", default="data/dax_physical_v3_cleaned_20261111/manifest.json")
    parser.add_argument("--qualified-manifest", default="data/dax_physical_v3_merged_spatial_50h_20261112/manifest.json")
    parser.add_argument("--output", default="data/dax_physical_v3_mixed_legacy_cleaned_plus_spatial_20261113")
    args = parser.parse_args()
    result = build_legacy_cleaned_merge(args.legacy_manifest, args.qualified_manifest, args.output)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
