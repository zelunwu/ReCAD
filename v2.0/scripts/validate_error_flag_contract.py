"""Validate and summarize a versioned absolute-error flag ``.in`` contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from recad.evaluate.error_tier_contract import load_error_flag_contract


def contract_summary(path: Path) -> dict[str, object]:
    """Return a machine-readable validation summary."""

    contract = load_error_flag_contract(path)
    return {
        "contract_id": contract.contract_id,
        "schema_version": contract.schema_version,
        "status": contract.status,
        "primary_statistic": contract.primary_statistic,
        "absolute_error_quantile": contract.absolute_error_quantile,
        "confidence_level": contract.confidence_level,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "targets": {
            name: {
                "variable_definition": profile.variable_definition,
                "unit": profile.unit,
                "thresholds_frozen": profile.thresholds.frozen,
                "threshold_a": profile.thresholds.a,
                "threshold_b": profile.thresholds.b,
                "threshold_c": profile.thresholds.c,
            }
            for name, profile in contract.targets.items()
        },
        "reason_bits": contract.reason_bits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-thresholds",
        action="store_true",
        help="Fail unless every target has fixed physical thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = contract_summary(args.config)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))
    if args.require_thresholds and not all(
        target["thresholds_frozen"] for target in summary["targets"].values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
