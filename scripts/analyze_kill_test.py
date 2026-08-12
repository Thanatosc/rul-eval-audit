"""Validate and summarize the completed registered Kill Test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rul_audit.experiments.kill_test import decide_kill_test, expand_cells, load_config
from rul_audit.metrics.rul import endpoint_metrics
from rul_audit.protocols.assets import validate_run_artifacts


def _load_registered_runs(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_config(project_root / "configs" / "kill_test.yaml")
    cells = expand_cells(config)
    status_path = project_root / "results" / "KILL_TEST_STATUS.csv"
    with status_path.open(encoding="utf-8", newline="") as handle:
        status_rows = list(csv.DictReader(handle))
    status_by_id = {row["run_id"]: row for row in status_rows}
    if len(status_rows) != len(cells) or len(status_by_id) != len(cells):
        raise ValueError("Kill Test status register does not match the 120 registered cells")

    records: list[dict[str, Any]] = []
    for cell in cells:
        status = status_by_id.get(cell.run_id)
        if status is None or status.get("status") != "completed":
            raise RuntimeError(f"registered cell is incomplete: {cell.run_id}")
        run_dir = project_root / status["run_path"]
        validate_run_artifacts(run_dir, project_root=project_root)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        expected = {
            "run_id": cell.run_id,
            "subset": cell.dataset,
            "model": cell.model,
            "seed": cell.seed,
            "output_mode": "point",
            "protocol_sha256": config["registration"]["protocol_sha256"],
            "code_revision": config["registration"]["implementation_sha256"],
        }
        for field, value in expected.items():
            if meta.get(field) != value:
                raise ValueError(
                    f"{cell.run_id} has meta.{field}={meta.get(field)!r}; expected {value!r}"
                )
        metrics = meta.get("metrics", {})
        recomputed = endpoint_metrics(pd.read_parquet(run_dir / "preds_test.parquet"))
        expected_engines = {"FD001": 100, "FD004": 248}[cell.dataset]
        if metrics.get("engine_count") != expected_engines:
            raise ValueError(f"{cell.run_id} has the wrong endpoint engine count")
        if recomputed["engine_count"] != expected_engines:
            raise ValueError(f"{cell.run_id} predictions have the wrong endpoint count")
        for metric in ("rmse", "nasa_score"):
            if not isinstance(metrics.get(metric), (int, float)) or not math.isfinite(
                metrics[metric]
            ):
                raise ValueError(f"{cell.run_id} has a non-finite {metric}")
            if not math.isclose(float(metrics[metric]), float(recomputed[metric]), rel_tol=1e-12):
                raise ValueError(f"{cell.run_id} meta.{metric} differs from recomputation")
        records.append(
            {
                "run_id": cell.run_id,
                "dataset": cell.dataset,
                "model": cell.model,
                "rul_label": cell.rul_label,
                "sensor_set": cell.sensor_set,
                "seed": cell.seed,
                "rmse": float(metrics["rmse"]),
                "nasa_score": float(metrics["nasa_score"]),
                "training_seconds": float(meta["training_seconds"]),
            }
        )
    return config, records


def analyze_kill_test(project_root: Path) -> dict[str, Any]:
    """Apply the frozen mean-effect and strict-ranking-reversal definitions."""

    config, records = _load_registered_runs(project_root.resolve())
    by_key = {
        (
            row["dataset"],
            row["model"],
            row["rul_label"],
            row["sensor_set"],
            row["seed"],
        ): row
        for row in records
    }
    effects: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            for sensor_set in config["factors"]["sensor_set"]:
                differences = [
                    by_key[(dataset, model, "linear_uncapped", sensor_set, seed)]["rmse"]
                    - by_key[(dataset, model, "piecewise_125", sensor_set, seed)]["rmse"]
                    for seed in config["seeds"]
                ]
                effects.append(
                    {
                        "effect_type": "label",
                        "dataset": dataset,
                        "model": model,
                        "held_level": sensor_set,
                        "contrast": "linear_uncapped-minus-piecewise_125",
                        "mean_rmse_effect": mean(differences),
                        "sd_seed_effect": stdev(differences),
                    }
                )
            for rul_label in config["factors"]["rul_label"]:
                differences = [
                    by_key[(dataset, model, rul_label, "common_14", seed)]["rmse"]
                    - by_key[(dataset, model, rul_label, "all_21", seed)]["rmse"]
                    for seed in config["seeds"]
                ]
                effects.append(
                    {
                        "effect_type": "sensor",
                        "dataset": dataset,
                        "model": model,
                        "held_level": rul_label,
                        "contrast": "common_14-minus-all_21",
                        "mean_rmse_effect": mean(differences),
                        "sd_seed_effect": stdev(differences),
                    }
                )

    grouped_rmse: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped_rmse[
            (row["dataset"], row["rul_label"], row["sensor_set"], row["model"])
        ].append(row["rmse"])
    mean_rmse = {key: mean(values) for key, values in grouped_rmse.items()}

    ranking_reversals: list[dict[str, Any]] = []

    def compare_levels(
        transition: str,
        dataset: str,
        held_level: str,
        level_a: str,
        level_b: str,
        keys_a: tuple[str, str],
        keys_b: tuple[str, str],
    ) -> None:
        for model_a, model_b in combinations(config["models"], 2):
            delta_a = mean_rmse[(dataset, keys_a[0], keys_a[1], model_a)] - mean_rmse[
                (dataset, keys_a[0], keys_a[1], model_b)
            ]
            delta_b = mean_rmse[(dataset, keys_b[0], keys_b[1], model_a)] - mean_rmse[
                (dataset, keys_b[0], keys_b[1], model_b)
            ]
            if delta_a * delta_b < 0:
                ranking_reversals.append(
                    {
                        "transition": transition,
                        "dataset": dataset,
                        "held_level": held_level,
                        "level_a": level_a,
                        "level_b": level_b,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_rmse_delta_a": delta_a,
                        "mean_rmse_delta_b": delta_b,
                    }
                )

    for dataset in config["datasets"]:
        for sensor_set in config["factors"]["sensor_set"]:
            compare_levels(
                "label",
                dataset,
                sensor_set,
                "piecewise_125",
                "linear_uncapped",
                ("piecewise_125", sensor_set),
                ("linear_uncapped", sensor_set),
            )
        for rul_label in config["factors"]["rul_label"]:
            compare_levels(
                "sensor",
                dataset,
                rul_label,
                "all_21",
                "common_14",
                (rul_label, "all_21"),
                (rul_label, "common_14"),
            )

    maximum_effect = max(abs(row["mean_rmse_effect"]) for row in effects)
    decision = decide_kill_test(
        max_absolute_rmse_effect=maximum_effect,
        ranking_reversals=len(ranking_reversals),
        completed_cells=len(records),
        registered_cells=int(config["registration"]["primary_cells"]),
    )
    runtime_groups: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        runtime_groups[(row["dataset"], row["model"])].append(row["training_seconds"])
    mean_rows = [
        {
            "dataset": dataset,
            "rul_label": rul_label,
            "sensor_set": sensor_set,
            "model": model,
            "mean_rmse": value,
        }
        for (dataset, rul_label, sensor_set, model), value in sorted(mean_rmse.items())
    ]
    return {
        "schema_version": 1,
        "analyzed_at": datetime.now(UTC).isoformat(),
        "registration_id": config["registration_id"],
        "protocol_sha256": config["registration"]["protocol_sha256"],
        "implementation_sha256": config["registration"]["implementation_sha256"],
        "registered_cells": int(config["registration"]["primary_cells"]),
        "completed_cells": len(records),
        "artifact_validation_errors": 0,
        "endpoint_metrics_recomputed": len(records),
        "maximum_absolute_registered_mean_rmse_effect": maximum_effect,
        "registered_ranking_reversal_count": len(ranking_reversals),
        "decision": decision,
        "effects": sorted(effects, key=lambda row: abs(row["mean_rmse_effect"]), reverse=True),
        "ranking_reversals": ranking_reversals,
        "mean_rmse_by_protocol": mean_rows,
        "runtime": {
            "sum_training_seconds": sum(row["training_seconds"] for row in records),
            "median_training_seconds": median(row["training_seconds"] for row in records),
            "by_dataset_model": [
                {
                    "dataset": dataset,
                    "model": model,
                    "runs": len(values),
                    "sum_seconds": sum(values),
                    "median_seconds": median(values),
                }
                for (dataset, model), values in sorted(runtime_groups.items())
            ],
        },
        "runs": records,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def write_kill_test_outputs(project_root: Path, report: dict[str, Any]) -> None:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    decision_payload = {key: value for key, value in report.items() if key != "runs"}
    (results_dir / "KILL_TEST_DECISION.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(report["effects"], results_dir / "KILL_TEST_RMSE_EFFECTS.csv")
    _write_csv(report["ranking_reversals"], results_dir / "KILL_TEST_RANKING_REVERSALS.csv")
    _write_csv(report["mean_rmse_by_protocol"], results_dir / "KILL_TEST_MEAN_RMSE.csv")
    run_frame = pd.DataFrame(report["runs"])
    pq.write_table(pa.Table.from_pandas(run_frame, preserve_index=False), results_dir / "KILL_TEST_RUN_METRICS.parquet")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = analyze_kill_test(root)
    if args.write:
        write_kill_test_outputs(root, report)
    summary = {
        key: report[key]
        for key in (
            "registration_id",
            "registered_cells",
            "completed_cells",
            "artifact_validation_errors",
            "endpoint_metrics_recomputed",
            "maximum_absolute_registered_mean_rmse_effect",
            "registered_ranking_reversal_count",
            "decision",
            "runtime",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
