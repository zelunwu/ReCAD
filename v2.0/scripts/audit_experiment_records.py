"""Inventory local experiment artifacts without reading scientific data.

This is a completeness audit, not an evidence grader. Evidence grades and
allowed claims remain curated in ``docs/experiment_registry.md`` because a
file named ``independent`` or ``protocol`` cannot prove statistical validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_PATTERNS = ("*metric*.csv", "*summary*.csv", "*ranking*.csv", "result.json")
CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt")


def matches(directory: Path, patterns: tuple[str, ...]) -> list[str]:
    found: set[str] = set()
    for pattern in patterns:
        found.update(str(path.relative_to(directory)) for path in directory.rglob(pattern))
    return sorted(found)


def audit_directory(directory: Path) -> dict[str, object]:
    reports = sorted(str(path.relative_to(directory)) for path in directory.rglob("*.md"))
    metrics = matches(directory, METRIC_PATTERNS)
    checkpoints = matches(directory, CHECKPOINT_PATTERNS)
    return {
        "experiment_id": directory.name,
        "protocol": (directory / "protocol.json").exists(),
        "data_manifest": (directory / "data_manifest.json").exists(),
        "selection": (directory / "selection.json").exists(),
        "reports": reports,
        "metric_files": metrics,
        "checkpoint_count": len(checkpoints),
        "artifact_completeness": sum(
            [
                (directory / "protocol.json").exists(),
                (directory / "data_manifest.json").exists(),
                (directory / "selection.json").exists(),
                bool(reports),
                bool(metrics),
                bool(checkpoints),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--json", type=Path, help="optional JSON output outside git")
    args = parser.parse_args()

    records = [audit_directory(path) for path in sorted(args.outputs.iterdir()) if path.is_dir()]
    payload = {"outputs": str(args.outputs.resolve()), "experiments": records}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("experiment\tprotocol\tmanifest\tselection\treport\tmetrics\tcheckpoints\tscore/6")
    for record in records:
        print(
            f"{record['experiment_id']}\t{int(record['protocol'])}\t"
            f"{int(record['data_manifest'])}\t{int(record['selection'])}\t"
            f"{int(bool(record['reports']))}\t{int(bool(record['metric_files']))}\t"
            f"{record['checkpoint_count']}\t{record['artifact_completeness']}"
        )


if __name__ == "__main__":
    main()
