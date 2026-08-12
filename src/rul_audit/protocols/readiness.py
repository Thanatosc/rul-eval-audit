"""Fail-closed readiness audit for the registered Kill Test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from rul_audit.data.cmapss import EXPECTED_DATASET_SHAPES, sha256_file, verify_dataset
from rul_audit.experiments.kill_test import expand_cells, load_config
from rul_audit.protocols.assets import validate_run_artifacts, validate_unit_split


def _check(condition: bool, name: str, detail: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": name, "status": "pass" if condition else "fail", "detail": detail})


def source_tree_sha256(project_root: Path) -> str:
    """Hash ordered implementation paths and bytes without requiring a Git commit."""

    source_root = project_root / "src" / "rul_audit"
    paths = sorted(source_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError("no Python implementation files found")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit_readiness(project_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config_path = project_root / "configs" / "kill_test.yaml"
    config = load_config(config_path)
    cells = expand_cells(config)
    _check(
        config.get("status") == "frozen_pending_execution_authorization",
        "kill_test_config_frozen",
        str(config.get("status")),
        checks,
    )
    _check(len(cells) == 120, "registered_grid_shape", f"{len(cells)} cells", checks)
    implementation_hash = source_tree_sha256(project_root)
    _check(
        implementation_hash == config["registration"].get("implementation_sha256"),
        "implementation_source_tree_sha256",
        implementation_hash,
        checks,
    )
    _check(
        config["factors"].get("sensor_set") == ["all_21", "common_14"],
        "literature_supported_sensor_factor",
        str(config["factors"].get("sensor_set")),
        checks,
    )

    protocol_path = project_root / "protocols" / "unified_v1.md"
    protocol_text = protocol_path.read_text(encoding="utf-8")
    protocol_hash = sha256_file(protocol_path)
    _check(
        "Status: `frozen_pending_execution_authorization`" in protocol_text
        and protocol_hash == config["registration"].get("protocol_sha256"),
        "protocol_frozen",
        protocol_hash,
        checks,
    )

    archive_path = project_root / config["dataset_provenance"]["archive_file"]
    archive_hash = sha256_file(archive_path) if archive_path.is_file() else "missing"
    _check(
        archive_hash.lower() == config["dataset_provenance"]["archive_sha256"].lower(),
        "official_archive_sha256",
        archive_hash,
        checks,
    )
    data_dir = project_root / config["dataset_provenance"]["extracted_data_dir"]
    try:
        dataset_report = verify_dataset(data_dir)
        dataset_ok = True
        dataset_detail = ", ".join(
            f"{key}:{value['train_units']}/{value['test_units']}"
            for key, value in dataset_report.items()
        )
    except (FileNotFoundError, ValueError) as exc:
        dataset_ok = False
        dataset_report = {}
        dataset_detail = str(exc)
    _check(dataset_ok, "dataset_files_and_unit_counts", dataset_detail, checks)

    split_hashes: dict[str, str] = {}
    split_ok = dataset_ok
    split_details: list[str] = []
    for subset, expected in EXPECTED_DATASET_SHAPES.items():
        split_path = project_root / "configs" / "splits" / f"{subset}_seed42.json"
        try:
            counts = validate_unit_split(split_path)
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            allocated = set().union(*(set(values) for values in payload["unit_split"].values()))
            expected_units = set(range(1, expected["train_units"] + 1))
            valid = (
                allocated == expected_units
                and payload.get("source_train_sha256", "").lower()
                == dataset_report.get(subset, {}).get("train_sha256", "").lower()
            )
            split_ok = split_ok and valid
            split_hashes[subset] = sha256_file(split_path)
            split_details.append(f"{subset}:{counts},coverage={valid}")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            split_ok = False
            split_details.append(f"{subset}:{exc}")
    _check(split_ok, "unit_splits", "; ".join(split_details), checks)

    status_path = project_root / "results" / "KILL_TEST_STATUS.csv"
    status_rows: list[dict[str, str]] = []
    if status_path.is_file():
        with status_path.open(encoding="utf-8", newline="") as handle:
            status_rows = list(csv.DictReader(handle))
    status_ok = (
        len(status_rows) == 120
        and len({row.get("run_id") for row in status_rows}) == 120
        and all(row.get("status") == "pending" for row in status_rows)
        and {row.get("run_id") for row in status_rows} == {cell.run_id for cell in cells}
    )
    _check(status_ok, "pending_status_register", f"rows={len(status_rows)}", checks)

    smoke_path = project_root / "results" / "runs" / "synthetic_smoke" / "summary.json"
    smoke_ok = False
    smoke_detail = "missing"
    if smoke_path.is_file():
        try:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            run_reports = []
            for run in smoke.get("runs", []):
                run_reports.append(
                    validate_run_artifacts(
                        project_root / run["run_path"], project_root=project_root
                    )
                )
            smoke_ok = (
                smoke.get("status") == "passed"
                and smoke.get("real_kill_test_cells_executed") == 0
                and {report["run_id"].split("__")[1] for report in run_reports}
                == {"lstm", "cnn_1d", "lightgbm"}
            )
            smoke_detail = f"validated_runs={len(run_reports)}"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            smoke_detail = str(exc)
    _check(smoke_ok, "synthetic_smoke_artifacts", smoke_detail, checks)

    phase6_files = [
        project_root / "paper" / "RESEARCH_REPORT_REVISED.md",
        project_root / "project" / "PHASE6_REVISION_LOG.md",
    ]
    phase6_ok = all(path.is_file() and path.stat().st_size > 0 for path in phase6_files)
    _check(phase6_ok, "phase6_revision_artifacts", str(phase6_ok), checks)

    failed = [entry for entry in checks if entry["status"] == "fail"]
    return {
        "schema_version": 1,
        "status": "ready_pending_execution_authorization" if not failed else "not_ready",
        "registered_cells": 120,
        "completed_cells": 0,
        "real_kill_test_started": False,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": implementation_hash,
        "split_sha256": split_hashes,
        "dataset": dataset_report,
        "checks": checks,
        "failed_checks": len(failed),
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Kill Test Readiness Report",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Registered cells:** {report['registered_cells']}  ",
        f"**Completed real cells:** {report['completed_cells']}  ",
        f"**Real Kill Test started:** `{str(report['real_kill_test_started']).lower()}`  ",
        f"**Protocol SHA-256:** `{report['protocol_sha256']}`",
        f"**Implementation SHA-256:** `{report['implementation_sha256']}`",
        "",
        "This report is a pre-execution gate. Synthetic smoke runs exercise code and artifact",
        "paths only; they are not C-MAPSS observations and cannot enter the paper's results.",
        "",
        "## Gate Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for entry in report["checks"]:
        detail = str(entry["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{entry['check']}` | {entry['status'].upper()} | {detail} |")
    lines.extend(
        [
            "",
            "## Dataset Unit Counts",
            "",
            "| Subset | Train units | Official test units | Train rows | Test rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset, values in report.get("dataset", {}).items():
        lines.append(
            f"| {subset} | {values['train_units']} | {values['test_units']} | "
            f"{values['train_rows']} | {values['test_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            "No row in `results/KILL_TEST_STATUS.csv` has left `pending`. The next action",
            "would be a real registered cell on NASA C-MAPSS and requires separate explicit",
            "authorization. Readiness does not authorize training, retries, threshold changes,",
            "or post hoc factor additions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = audit_readiness(args.root)
    if args.write_report:
        path = args.root / "project" / "KILL_TEST_READINESS_REPORT.md"
        path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready_pending_execution_authorization" else 1


if __name__ == "__main__":
    raise SystemExit(main())
