"""Validate and summarize the frozen Issue 88 model-family benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from recad.evaluate.benchmark_contract import load_benchmark_contract, primary_cell_budget


def contract_summary(path: Path) -> dict[str, object]:
    """Return a compact machine-readable contract summary."""

    contract = load_benchmark_contract(path)
    return {
        "contract_id": contract.contract_id,
        "schema_version": contract.schema_version,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "targets": list(contract.targets),
        "families": list(contract.families),
        "protocols": list(contract.protocols),
        "fidelity_rows": list(contract.fidelity_rows),
        "advancers_per_target": contract.advancers_per_target,
        "primary_cell_budget": primary_cell_budget(contract).to_dict("records"),
        "family_resources": {
            name: {
                "minimum_trials": family.minimum_trials,
                "maximum_trials": family.maximum_trials,
                "minimum_seeds": family.minimum_seeds,
                "backend": family.backend,
                "representation": family.representation,
            }
            for name, family in contract.family_contracts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summary = contract_summary(args.config)
    payload = json.dumps(summary, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
