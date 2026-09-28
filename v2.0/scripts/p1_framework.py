"""CLI for the shared P1 data/evaluation contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recad.evaluate.p1_framework import (
    FrozenManifest,
    P1DataGateway,
    Purpose,
    build_audit,
    evaluate_predictions,
    load_framework_config,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/p1_framework_v2.2.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--hash-mode", choices=("full", "artifacts", "existence"), default="full")
    audit.add_argument("--output", type=Path)
    describe = sub.add_parser("describe")
    describe.add_argument("target", choices=("sss", "fco2", "ta", "dic"))
    describe.add_argument("purpose", choices=("train", "selection"))
    describe.add_argument("--scheme", choices=("primary", "forward"), default="primary")
    metrics = sub.add_parser("metrics")
    metrics.add_argument("predictions", type=Path)
    metrics.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_framework_config(args.config)
    manifest_path = Path(args.config).resolve().parent / str(config["data_manifest"])
    manifest = FrozenManifest.load(manifest_path)
    gateway = P1DataGateway(manifest)
    if args.command == "audit":
        result = build_audit(gateway, manifest.validate(hash_mode=args.hash_mode))
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
    elif args.command == "describe":
        frame = gateway.load_labels(args.target, Purpose(args.purpose), split_scheme=args.scheme)
        result = {"target": args.target, "purpose": args.purpose, "rows": len(frame),
                  "split_scheme": args.scheme,
                  "cruises": frame.group_key.nunique(), "year_min": int(frame.year.min()),
                  "year_max": int(frame.year.max())}
        print(json.dumps(result, indent=2))
    else:
        frame = pd.read_parquet(args.predictions)
        result = evaluate_predictions(frame, support_bins=config["support_distance_bins_km"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(args.output, index=False)
        print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
