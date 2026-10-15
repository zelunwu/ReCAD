from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _socat_versions_from_names(raw_dir: Path) -> list[int]:
    versions: set[int] = set()
    if raw_dir.is_dir():
        for path in raw_dir.iterdir():
            match = re.search(r"SOCATv(\d{4})", path.name, flags=re.IGNORECASE)
            if match:
                versions.add(int(match.group(1)))
    return sorted(versions)


def _read_exposure_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_lock(
    repo_root: Path,
    external_root: Path,
    exposure_summary: Path,
    socat_raw_dir: Path,
) -> dict[str, Any]:
    v2 = repo_root / "v2.0"
    calibration = external_root / "calibration_contract.json"
    frozen_calibration = external_root / "calibration_contract_frozen.json"
    checkpoints = sorted((external_root / "checkpoints").glob("*"))
    if not calibration.is_file() or not frozen_calibration.is_file() or not checkpoints:
        raise FileNotFoundError("Issue 44 calibration objects or checkpoints are incomplete")

    contract = json.loads(calibration.read_text(encoding="utf-8"))
    exposure = _read_exposure_summary(exposure_summary)
    old_socat = [row for row in exposure if row["dataset"] == "socat"]
    current_versions = _socat_versions_from_names(socat_raw_dir)
    prospective_versions = [version for version in current_versions if version > 2026]

    repository_files = [
        "configs/p2_global_reliability_contract_v2.4.json",
        "configs/frozen/global_coastal_mask_manifest_v2.3.json",
        "configs/frozen/global_predictor_cube_manifest_v2.3.json",
        "configs/frozen/product_scope_v2.3.json",
        "docs/experiment_archive/p2_global_reliability_contract_v2.4/REPORT.md",
        "scripts/run_p2_global_reliability_contract_v2_4.py",
        "scripts/build_p2_global_reliability_product_v2_4.py",
        "scripts/build_p2_global_reliability_archive.py",
        "src/recad/evaluate/reliability_contract.py",
    ]
    repository_hashes = {
        path: sha256_normalized_text(v2 / path)
        for path in repository_files
    }
    checkpoint_hashes = {
        str(path.relative_to(external_root)).replace("\\", "/"): sha256_file(path)
        for path in checkpoints
        if path.is_file()
    }

    eligible = bool(prospective_versions)
    blockers = []
    if not eligible:
        blockers.append(
            "No SOCAT release newer than the frozen v2026 snapshot is present for metadata-only selection."
        )
    if any(int(row["groups_in_matchup"]) > 0 for row in old_socat):
        blockers.append(
            "All legacy SOCAT locked/external families overlap P2 development and are permanently ineligible."
        )
    blockers.append("TA/DIC are excluded because Issue 45 closed untriggered.")

    return {
        "schema_version": "2.4",
        "issue": 5,
        "status": "prepared_waiting_for_eligible_independent_set",
        "candidate_git_commit": _git_head(repo_root),
        "targets": {
            "SSS": {
                "candidate": "Issue-62 CatBoost background-residual three-seed ensemble",
                "product_sha256": contract["products"]["SSS"]["sha256"],
            },
            "fCO2": {
                "candidate": "Issue-64 CatBoost/F4 inner-OOF convex ensemble",
                "catboost_weight": contract["fco2_catboost_weight"],
                "product_sha256": contract["products"]["fCO2"]["sha256"],
            },
            "TA": {"included": False, "reason": "Issue 45 untriggered"},
            "DIC": {"included": False, "reason": "Issue 45 untriggered"},
        },
        "calibration": {
            "contract_sha256": sha256_file(calibration),
            "frozen_contract_sha256": sha256_file(frozen_calibration),
        },
        "repository_artifact_sha256": repository_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "input_sha256": {
            "predictor_cube": "0f4e9f37ddd65d5abb0ebd92715a8521703a7778375d05c1a6055254203d0401",
            "coastal_mask": "0e99ca53398a3f4c29d5058b3307857535c78195bcbabdadf06c59b0ee4957cf",
        },
        "metadata_only_eligibility": {
            "frozen_source": "SOCAT v2026",
            "available_versions_from_filenames": current_versions,
            "prospective_versions": prospective_versions,
            "legacy_exposure_summary": old_socat,
            "eligible_exact_set_identified": eligible,
            "target_labels_read": False,
        },
        "label_access_authorized": False,
        "label_access_count": 0,
        "blockers": blockers,
        "unlock_rule": (
            "Replace the null candidate set with a metadata-only cruise list from a SOCAT release "
            "newer than v2026, prove zero normalized cruise/alias overlap with the Issue-61 ledger, "
            "hash that list and its source, obtain the frozen protocol decision, then create a new lock."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--exposure-summary", type=Path, required=True)
    parser.add_argument("--socat-raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = prepare_lock(
        args.repo_root.resolve(),
        args.external_root.resolve(),
        args.exposure_summary.resolve(),
        args.socat_raw_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
