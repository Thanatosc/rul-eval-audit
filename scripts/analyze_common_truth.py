"""Run the frozen common-truth-scale sensitivity analysis for Stage 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import pandas as pd

from rul_audit.data.cmapss import load_subset, sha256_file
from rul_audit.experiments.kill_test import expand_cells, load_config
from rul_audit.metrics.rul import endpoint_rows, rmse

ANALYSIS_ID = "common_truth_v1"
PREDICTION_MANIFEST_SHA256 = (
    "097e0daf846bd5460630451ddc3fa84dabac35a508cb4c322ea627f2b64a6c5b"
)
PREDICTION_TOTAL_BYTES = 18_806_024
INPUT_HASHES = {
    "configs/kill_test.yaml": (
        "370643e71dcb15eb9a9889066fe729069859a2d112a513ce98de30092a3d91ef"
    ),
    "protocols/unified_v1.md": (
        "8e0bcda253ab008079a82d536b011d44a31e4cfa308c6df0032c481daea3de44"
    ),
    "results/KILL_TEST_STATUS.csv": (
        "cadf78128a9552a62d2e890664e6803bcf17dd007124f5207943c93bf1aaa81f"
    ),
    "results/KILL_TEST_DECISION.json": (
        "cf4747dbdc42bd3dabac85039ae1ec03fb2e5f75bb1460e5fd6fbbc451b61263"
    ),
    "data/raw/C-MAPSS_Turbofan.zip": (
        "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
    ),
    "data/interim/cmapss/RUL_FD001.txt": (
        "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
    ),
    "data/interim/cmapss/test_FD001.txt": (
        "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
    ),
    "data/interim/cmapss/RUL_FD004.txt": (
        "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1"
    ),
    "data/interim/cmapss/test_FD004.txt": (
        "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5"
    ),
}
COMMON_TRUTHS = ("raw_rul", "piecewise_125")
EXPECTED_ENGINES = {"FD001": 100, "FD004": 248}


def _prediction_manifest(project_root: Path) -> tuple[list[Path], str, int]:
    paths = sorted(project_root.glob("results/runs/kill_v1__*/preds_test.parquet"))
    lines = [
        f"{path.relative_to(project_root).as_posix()}\t{sha256_file(path)}"
        for path in paths
    ]
    payload = ("\n".join(lines) + "\n").encode()
    return paths, hashlib.sha256(payload).hexdigest(), sum(path.stat().st_size for path in paths)


def _validate_frozen_inputs(project_root: Path) -> dict[str, str]:
    observed = {
        relative: sha256_file(project_root / relative) for relative in INPUT_HASHES
    }
    mismatches = {
        relative: {"expected": INPUT_HASHES[relative], "observed": digest}
        for relative, digest in observed.items()
        if digest != INPUT_HASHES[relative]
    }
    if mismatches:
        raise RuntimeError(f"frozen common-truth input hash mismatch: {mismatches}")
    paths, manifest_hash, total_bytes = _prediction_manifest(project_root)
    if len(paths) != 120:
        raise RuntimeError(f"expected 120 prediction files, found {len(paths)}")
    if manifest_hash != PREDICTION_MANIFEST_SHA256:
        raise RuntimeError(
            "prediction manifest hash mismatch: "
            f"{manifest_hash} != {PREDICTION_MANIFEST_SHA256}"
        )
    if total_bytes != PREDICTION_TOTAL_BYTES:
        raise RuntimeError(
            f"prediction byte count mismatch: {total_bytes} != {PREDICTION_TOTAL_BYTES}"
        )
    return observed


def _load_status(project_root: Path) -> dict[str, dict[str, str]]:
    with (project_root / "results" / "KILL_TEST_STATUS.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["run_id"]: row for row in rows}
    if len(rows) != 120 or len(indexed) != 120:
        raise RuntimeError("Kill Test status register must contain 120 unique rows")
    if any(row["status"] != "completed" for row in rows):
        raise RuntimeError("all Kill Test cells must be completed")
    return indexed


def _truth_lookup(project_root: Path, dataset: str) -> pd.DataFrame:
    test = load_subset(project_root / "data" / "interim" / "cmapss", dataset).test
    truth = test[["unit_id", "cycle", "raw_rul"]].copy()
    truth["piecewise_125"] = truth["raw_rul"].clip(upper=125.0)
    return truth


def _run_metrics(
    project_root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    status = _load_status(project_root)
    truth_by_dataset = {
        dataset: _truth_lookup(project_root, dataset) for dataset in config["datasets"]
    }
    records: list[dict[str, Any]] = []
    native_truth_rows_validated = 0
    for cell in expand_cells(config):
        row = status.get(cell.run_id)
        if row is None:
            raise RuntimeError(f"unregistered run missing from status: {cell.run_id}")
        prediction_path = project_root / row["run_path"] / "preds_test.parquet"
        predictions = pd.read_parquet(prediction_path)
        required = {"unit_id", "cycle", "true_rul", "pred_rul"}
        if required - set(predictions):
            raise ValueError(f"{cell.run_id} is missing prediction columns")
        if predictions.duplicated(["unit_id", "cycle"]).any():
            raise ValueError(f"{cell.run_id} has duplicate unit-cycle rows")
        merged = predictions.merge(
            truth_by_dataset[cell.dataset],
            on=["unit_id", "cycle"],
            how="left",
            validate="one_to_one",
        )
        if merged[["raw_rul", "piecewise_125"]].isna().any().any():
            raise ValueError(f"{cell.run_id} has prediction rows without reconstructed truth")
        native_truth_column = {
            "piecewise_125": "piecewise_125",
            "linear_uncapped": "raw_rul",
        }[cell.rul_label]
        native = merged[native_truth_column].to_numpy(dtype=float)
        if not np.array_equal(merged["true_rul"].to_numpy(dtype=float), native):
            raise ValueError(f"{cell.run_id} saved native truth does not match reconstruction")
        native_truth_rows_validated += len(merged)
        endpoints = endpoint_rows(merged)
        expected = EXPECTED_ENGINES[cell.dataset]
        if len(endpoints) != expected:
            raise ValueError(f"{cell.run_id} has {len(endpoints)} endpoints; expected {expected}")
        predictions_array = endpoints["pred_rul"].to_numpy(dtype=float)
        for common_truth in COMMON_TRUTHS:
            records.append(
                {
                    "run_id": cell.run_id,
                    "dataset": cell.dataset,
                    "model": cell.model,
                    "sensor_set": cell.sensor_set,
                    "seed": cell.seed,
                    "training_label": cell.rul_label,
                    "common_truth": common_truth,
                    "rmse": rmse(
                        endpoints[common_truth].to_numpy(dtype=float),
                        predictions_array,
                    ),
                    "engine_count": len(endpoints),
                    "prediction_clipping": "none",
                }
            )
    return records, {"native_truth_rows_validated": native_truth_rows_validated}


def _label_contrasts(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["dataset"],
            row["model"],
            row["sensor_set"],
            row["seed"],
            row["training_label"],
            row["common_truth"],
        ): row
        for row in records
    }
    contrasts: list[dict[str, Any]] = []
    for common_truth in COMMON_TRUTHS:
        for dataset in config["datasets"]:
            for model in config["models"]:
                for sensor_set in config["factors"]["sensor_set"]:
                    differences = [
                        indexed[
                            (
                                dataset,
                                model,
                                sensor_set,
                                seed,
                                "linear_uncapped",
                                common_truth,
                            )
                        ]["rmse"]
                        - indexed[
                            (
                                dataset,
                                model,
                                sensor_set,
                                seed,
                                "piecewise_125",
                                common_truth,
                            )
                        ]["rmse"]
                        for seed in config["seeds"]
                    ]
                    contrasts.append(
                        {
                            "common_truth": common_truth,
                            "dataset": dataset,
                            "model": model,
                            "sensor_set": sensor_set,
                            "contrast": "linear_trained-minus-capped_trained",
                            "seed_count": len(differences),
                            "mean_rmse_difference": mean(differences),
                            "sd_seed_difference": stdev(differences),
                            "minimum_seed_difference": min(differences),
                            "maximum_seed_difference": max(differences),
                        }
                    )
    return contrasts


def _ranking_reversals(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[
            (
                row["common_truth"],
                row["dataset"],
                row["sensor_set"],
                row["training_label"],
                row["model"],
            )
        ].append(row["rmse"])
    means = {key: mean(values) for key, values in grouped.items()}
    reversals: list[dict[str, Any]] = []
    for common_truth in COMMON_TRUTHS:
        for dataset in config["datasets"]:
            for sensor_set in config["factors"]["sensor_set"]:
                for model_a, model_b in combinations(config["models"], 2):
                    capped_delta = means[
                        (common_truth, dataset, sensor_set, "piecewise_125", model_a)
                    ] - means[
                        (common_truth, dataset, sensor_set, "piecewise_125", model_b)
                    ]
                    linear_delta = means[
                        (common_truth, dataset, sensor_set, "linear_uncapped", model_a)
                    ] - means[
                        (common_truth, dataset, sensor_set, "linear_uncapped", model_b)
                    ]
                    if capped_delta * linear_delta < 0:
                        reversals.append(
                            {
                                "common_truth": common_truth,
                                "dataset": dataset,
                                "sensor_set": sensor_set,
                                "model_a": model_a,
                                "model_b": model_b,
                                "capped_trained_mean_rmse_delta": capped_delta,
                                "linear_trained_mean_rmse_delta": linear_delta,
                            }
                        )
    return reversals


def analyze_common_truth(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    input_hashes = _validate_frozen_inputs(root)
    config = load_config(root / "configs" / "kill_test.yaml")
    records, validation = _run_metrics(root, config)
    contrasts = _label_contrasts(records, config)
    reversals = _ranking_reversals(records, config)
    maximum_by_truth = {
        truth: max(
            abs(row["mean_rmse_difference"])
            for row in contrasts
            if row["common_truth"] == truth
        )
        for truth in COMMON_TRUTHS
    }
    if len(records) != 240 or len(contrasts) != 24:
        raise AssertionError("common-truth output matrix did not close")
    if not all(math.isfinite(row["rmse"]) for row in records):
        raise AssertionError("common-truth metrics contain non-finite values")
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_validated",
        "frozen_plan": "project/stage4_revision/COMMON_TRUTH_SENSITIVITY_PLAN.md",
        "primary_kill_test_member": False,
        "changes_kill_v1_decision": False,
        "registered_input_runs": 120,
        "run_truth_metric_rows": len(records),
        "paired_label_contrasts": len(contrasts),
        "common_truths": list(COMMON_TRUTHS),
        "prediction_clipping": "none",
        "maximum_absolute_mean_difference_by_truth": maximum_by_truth,
        "ranking_reversal_count": len(reversals),
        "validation": {
            **validation,
            "input_hashes": input_hashes,
            "prediction_manifest_sha256": PREDICTION_MANIFEST_SHA256,
            "prediction_total_bytes": PREDICTION_TOTAL_BYTES,
            "endpoint_metrics_recomputed": 240,
            "artifact_errors": 0,
        },
        "contrasts": contrasts,
        "ranking_reversals": reversals,
        "run_metrics": records,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _validation_report(report: dict[str, Any]) -> str:
    raw_max = report["maximum_absolute_mean_difference_by_truth"]["raw_rul"]
    capped_max = report["maximum_absolute_mean_difference_by_truth"]["piecewise_125"]
    return f"""# Common-Truth-Scale Sensitivity Validation

## Status

- Analysis: `{ANALYSIS_ID}`
- Result: **COMPLETED AND VALIDATED**
- Input runs: 120/120
- Run-by-truth endpoint metrics: 240
- Paired five-seed label contrasts: 24
- Prediction clipping: none
- Upstream artifact errors: 0

## Result Boundary

The maximum absolute five-seed mean difference between linear-trained and
capped-trained predictions was `{raw_max:.4f}` RMSE on common raw-RUL truth and
`{capped_max:.4f}` RMSE on common piecewise-125 truth. These values compare
training-target/model bundles on fixed evaluation truths. They do not replace the
native-truth 20.5323 task-definition contrast, isolate a causal training-label
effect, or change the frozen `kill_v1` PASS decision.

The common-truth analysis found {report['ranking_reversal_count']} strict model-pair
ranking reversals across the native training-label levels under the two fixed
evaluation truths. Full rows are retained in
`COMMON_TRUTH_RANKING_REVERSALS.csv`.

## Validation Checks

- All frozen file hashes and the 120-file prediction manifest matched.
- Saved native truth matched reconstructed raw/capped truth for
  {report['validation']['native_truth_rows_validated']:,} prediction rows.
- Every run yielded the registered endpoint count: 100 for FD001 and 248 for FD004.
- Predictions were not clipped, capped, shifted, retrained, or recalibrated.
- The original Kill Test status, decision, protocol, and prediction files were read-only.
- The analysis is secondary and descriptive; no new p-value or preregistration claim is made.
"""


def write_outputs(project_root: Path, report: dict[str, Any]) -> None:
    output = project_root / "results" / "common_truth"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(report["run_metrics"], output / "COMMON_TRUTH_RUN_METRICS.csv")
    _write_csv(report["contrasts"], output / "COMMON_TRUTH_LABEL_CONTRASTS.csv")
    _write_csv(report["ranking_reversals"], output / "COMMON_TRUTH_RANKING_REVERSALS.csv")
    summary = {key: value for key, value in report.items() if key != "run_metrics"}
    (output / "COMMON_TRUTH_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "COMMON_TRUTH_VALIDATION.md").write_text(
        _validation_report(report), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_common_truth(args.root)
    if args.write:
        write_outputs(args.root.resolve(), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"run_metrics", "contrasts"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
