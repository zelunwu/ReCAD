"""Resume every model-family cell in the Issue 89 global benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from recad.evaluate.benchmark_backends import PATCH_FAMILIES, TABULAR_FAMILIES
from run_p2_validation_repair_v2_4 import cached_splits, load_frame

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_model_family_benchmark_v2.5")
WORKER = ROOT / "scripts/run_global_tabular_benchmark_v2_5.py"
ORDER = (
    "ridge",
    "hist_gradient_boosting",
    "extra_trees",
    "random_forest",
    "lightgbm",
    "xgboost",
    "catboost",
    "residual_mlp",
    "realmlp",
    "tabm",
    "ft_transformer",
    "tabpfn_3",
    "unet_convlstm",
    "vision_transformer",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete(target: str, scheme: str, fold: int, family: str) -> bool:
    terminal = OUTPUT / "family_status" / f"{target}_{scheme}_fold{fold}_{family}.json"
    if terminal.is_file():
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        if payload.get("status") == "resource_infeasible":
            return True
    decision = OUTPUT / "search" / f"{target}_{scheme}_fold{fold}_{family}" / "decision.json"
    if not decision.is_file():
        return False
    payload = json.loads(decision.read_text(encoding="utf-8"))
    if not payload["converged"]:
        return True
    seeds = (
        (89001, 89002, 89003, 89004, 89005)
        if family
        in {
            "residual_mlp",
            "realmlp",
            "tabm",
            "ft_transformer",
            "tabpfn_3",
            "unet_convlstm",
            "vision_transformer",
        }
        else (89001, 89002, 89003)
    )
    return all(
        (
            OUTPUT / "partitions" / f"{target}_{scheme}_fold{fold}_{family}_seed{seed}.parquet"
        ).is_file()
        for seed in seeds
    )


def preserve_resource_failure(
    target: str,
    scheme: str,
    fold: int,
    family: str,
) -> bool:
    """Archive a deterministic OOM as a terminal, non-advancing negative result."""

    failure = OUTPUT / "failures" / f"{target}_{scheme}_fold{fold}_{family}.json"
    if not failure.is_file():
        return False
    payload = json.loads(failure.read_text(encoding="utf-8"))
    if payload.get("error_type") != "MemoryError":
        return False
    write_json(
        OUTPUT / "family_status" / f"{target}_{scheme}_fold{fold}_{family}.json",
        {
            "target": target,
            "scheme": scheme,
            "fold": fold,
            "family": family,
            "status": "resource_infeasible",
            "advancement_eligible": False,
            "reason": "full_fidelity_memory_exhausted",
            "failure": str(failure),
            "search_space_changed": False,
            "partial_trials_preserved": True,
            "updated_utc": utc_now(),
        },
    )
    return True


def main() -> None:
    if set(ORDER) != set(TABULAR_FAMILIES) | set(PATCH_FAMILIES):
        raise RuntimeError("pipeline order drifted from the frozen family set")
    log_path = OUTPUT / "tabular_pipeline.log"
    status_path = OUTPUT / "tabular_pipeline.status.json"
    license_failure = OUTPUT / "entry_gate/tabpfn_3.json"
    tabpfn_unavailable = False
    if license_failure.is_file():
        payload = json.loads(license_failure.read_text(encoding="utf-8"))
        tabpfn_unavailable = payload.get("status") == "license_unavailable" or (
            payload.get("status") == "failed"
            and "license" in str(payload.get("traceback", "")).lower()
        )
    with log_path.open("a", encoding="utf-8") as log:
        for target in ("SSS", "fCO2"):
            frame = load_frame(target)
            splits = cached_splits(target, frame)
            for family in ORDER:
                if family == "tabpfn_3" and tabpfn_unavailable:
                    write_json(
                        OUTPUT / "family_status" / f"{target}_{family}.json",
                        {
                            "target": target,
                            "family": family,
                            "status": "license_unavailable",
                            "source": str(license_failure),
                            "substitution": False,
                            "updated_utc": utc_now(),
                        },
                    )
                    continue
                for split in splits:
                    if complete(target, split.scheme, split.fold, family):
                        continue
                    cell = {
                        "target": target,
                        "scheme": split.scheme,
                        "fold": split.fold,
                        "family": family,
                        "status": "running",
                        "started_utc": utc_now(),
                    }
                    write_json(status_path, cell)
                    command = [
                        sys.executable,
                        str(WORKER),
                        "--target",
                        target,
                        "--scheme",
                        split.scheme,
                        "--fold",
                        str(split.fold),
                        "--family",
                        family,
                    ]
                    log.write(f"{utc_now()} START {' '.join(command)}\n")
                    log.flush()
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    log.write(f"{utc_now()} EXIT {result.returncode} {' '.join(command)}\n")
                    log.flush()
                    if result.returncode != 0:
                        if preserve_resource_failure(
                            target,
                            split.scheme,
                            split.fold,
                            family,
                        ):
                            cell["status"] = "resource_infeasible"
                            cell["returncode"] = result.returncode
                            cell["finished_utc"] = utc_now()
                            write_json(status_path, cell)
                            continue
                        cell["status"] = "failed"
                        cell["returncode"] = result.returncode
                        cell["finished_utc"] = utc_now()
                        write_json(status_path, cell)
                        raise SystemExit(result.returncode)
    write_json(
        status_path,
        {"status": "complete", "families": list(ORDER), "finished_utc": utc_now()},
    )
    (OUTPUT / "model_pipeline.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
