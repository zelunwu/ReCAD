"""Verify reviewer-ready experiment archives before an experiment is closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_MANIFEST_FIELDS = {
    "experiment_id",
    "training_git_commit",
    "data_manifest_sha256",
    "locked_test_opened",
    "external_independent_opened",
    "evidence_scope",
    "decision",
    "figure_source_data",
    "figure_captions",
    "files_sha256",
    "archive_builder_sha256",
    "analysis_script_sha256",
    "source_artifacts_sha256",
}
REQUIRED_REPORT_HEADINGS = {
    "## Scientific question and permitted claim",
    "## Data, splits, and leakage controls",
    "## Candidate models and training",
    "## Main development results",
    "## Decision and limitations",
    "## Figure and table index",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive: Path) -> dict[str, object]:
    """Return a machine-readable verification result for one archive."""
    archive = archive.resolve()
    errors: list[str] = []
    for relative in ("README.md", "REPORT.md", "CAPTIONS.md", "archive_manifest.json"):
        if not (archive / relative).is_file():
            errors.append(f"missing required file: {relative}")

    figures = sorted((archive / "figures").glob("*")) if (archive / "figures").is_dir() else []
    tables = sorted((archive / "tables").glob("*.csv")) if (archive / "tables").is_dir() else []
    figures = [path for path in figures if path.is_file()]
    if not figures:
        errors.append("no reviewer-ready figures found")
    if not tables:
        errors.append("no source CSV tables found")

    report_path = archive / "REPORT.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        for heading in sorted(REQUIRED_REPORT_HEADINGS):
            if heading not in report:
                errors.append(f"REPORT.md missing heading: {heading}")

    manifest_path = archive / "archive_manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid archive_manifest.json: {exc}")

    for field in sorted(REQUIRED_MANIFEST_FIELDS - set(manifest)):
        errors.append(f"manifest missing field: {field}")

    hashes = manifest.get("files_sha256", {})
    if not isinstance(hashes, dict):
        errors.append("manifest files_sha256 must be an object")
        hashes = {}
    for relative, expected in hashes.items():
        path = archive / str(relative)
        if not path.is_file():
            errors.append(f"hashed file missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")

    expected_tracked = {"README.md", "REPORT.md", "CAPTIONS.md"}
    expected_tracked.update(str(path.relative_to(archive)).replace("\\", "/") for path in figures)
    expected_tracked.update(str(path.relative_to(archive)).replace("\\", "/") for path in tables)
    for relative in sorted(expected_tracked - set(hashes)):
        errors.append(f"archive file is not hashed: {relative}")

    mapping = manifest.get("figure_source_data", {})
    if not isinstance(mapping, dict):
        errors.append("manifest figure_source_data must be an object")
        mapping = {}
    figure_names = {path.name for path in figures}
    for figure_name in sorted(figure_names - set(mapping)):
        errors.append(f"figure has no source-data mapping: {figure_name}")
    for figure_name, sources in mapping.items():
        if figure_name not in figure_names:
            errors.append(f"source mapping references missing figure: {figure_name}")
        if not isinstance(sources, list) or not sources:
            errors.append(f"figure has no source data: {figure_name}")
            continue
        for source in sources:
            if str(source).startswith("local:"):
                continue
            if not (archive / "tables" / str(source)).is_file():
                errors.append(f"figure source table missing: {figure_name} -> {source}")

    captions = manifest.get("figure_captions", {})
    if not isinstance(captions, dict):
        errors.append("manifest figure_captions must be an object")
        captions = {}
    for figure_name in sorted(figure_names - set(captions)):
        errors.append(f"figure has no caption: {figure_name}")
    for figure_name, caption in captions.items():
        if figure_name not in figure_names:
            errors.append(f"caption references missing figure: {figure_name}")
        if not isinstance(caption, str) or len(caption.strip()) < 40:
            errors.append(f"figure caption is too short: {figure_name}")
        if report_path.is_file() and f"](figures/{figure_name})" not in report:
            errors.append(f"REPORT.md does not embed figure: {figure_name}")
        if report_path.is_file() and isinstance(caption, str) and caption not in report:
            errors.append(f"REPORT.md does not place caption with figure: {figure_name}")
    captions_path = archive / "CAPTIONS.md"
    captions_text = captions_path.read_text(encoding="utf-8") if captions_path.is_file() else ""
    for figure_name, caption in captions.items():
        if isinstance(caption, str) and (
            figure_name not in captions_text or caption not in captions_text
        ):
            errors.append(f"CAPTIONS.md does not match manifest caption: {figure_name}")

    return {
        "archive": str(archive),
        "experiment_id": manifest.get("experiment_id"),
        "figures": len(figures),
        "captions": len(captions),
        "source_tables": len(tables),
        "verified": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    archives = args.archives or sorted(
        path.parent for path in Path("docs/experiment_archive").glob("*/archive_manifest.json")
    )
    if not archives:
        print(json.dumps({"verified": False, "errors": ["no experiment archives found"]}, indent=2))
        return 1
    results = [verify_archive(path) for path in archives]
    print(json.dumps(results, indent=2))
    return 0 if all(result["verified"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
