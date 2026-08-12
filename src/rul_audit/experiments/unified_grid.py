"""Fail-closed registration and execution for the 160-cell unified asset grid."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from rul_audit.data.cmapss import (
    apply_rul_label,
    fit_train_minmax,
    load_subset,
    sensor_columns,
    sha256_file,
    split_frame_by_unit,
    transform_features,
)
from rul_audit.experiments.kill_test import (
    _prediction_frame,
    _windows_with_short_endpoint_support,
    _write_parquet,
)
from rul_audit.metrics.rul import endpoint_metrics
from rul_audit.models.baselines import (
    fit_lightgbm,
    fit_neural,
    predict_lightgbm,
    predict_neural,
)
from rul_audit.protocols.assets import validate_run_artifacts, validate_unified_grid
from rul_audit.protocols.readiness import source_tree_sha256

STATUS_COLUMNS = (
    "run_id",
    "dataset",
    "subset",
    "model",
    "seed",
    "output_mode",
    "protocol_version",
    "status",
    "run_path",
    "notes",
)


@dataclass(frozen=True)
class UnifiedGridCell:
    run_id: str
    dataset: str
    subset: str
    model: str
    seed: int
    output_mode: str
    protocol_version: str


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("unified-grid config must be a YAML mapping")
    return payload


def expand_cells(config: dict[str, Any]) -> list[UnifiedGridCell]:
    dataset_subsets = [
        (entry["dataset"], subset)
        for entry in config["datasets"]
        for subset in entry["subsets"]
    ]
    models = [entry["id"] for entry in config["backbones"]]
    modes = list(config["output_modes"])
    protocol_version = str(config["protocol"]["version"])
    template = str(config["run_id_template"])
    cells: list[UnifiedGridCell] = []
    for (dataset, subset), model, seed, output_mode in product(
        dataset_subsets, models, config["seeds"], modes
    ):
        cells.append(
            UnifiedGridCell(
                run_id=template.format(
                    subset=subset,
                    subset_lower=str(subset).lower(),
                    model=model,
                    seed=seed,
                    output_mode=output_mode,
                ),
                dataset=str(dataset),
                subset=str(subset),
                model=str(model),
                seed=int(seed),
                output_mode=str(output_mode),
                protocol_version=protocol_version,
            )
        )
    expected = int(config["expected_cells"])
    if len(cells) != expected or len({cell.run_id for cell in cells}) != expected:
        raise ValueError(f"unified grid expands to {len(cells)} unique cells; expected {expected}")
    return cells


def _status_path(root: Path, config: dict[str, Any]) -> Path:
    return root / str(config["status_register"])


def _registered_row(root: Path, config: dict[str, Any], run_id: str) -> dict[str, str]:
    with _status_path(root, config).open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registered status row for {run_id}, got {len(matches)}")
    return matches[0]


def _update_status(
    root: Path, config: dict[str, Any], run_id: str, status: str, notes: str
) -> None:
    path = _status_path(root, config)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not any(row.get("run_id") == run_id for row in rows):
        raise ValueError(f"run_id is not registered: {run_id}")
    for row in rows:
        if row.get("run_id") == run_id:
            row["status"] = status
            row["notes"] = notes
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_execution_gate(root: Path, config: dict[str, Any]) -> tuple[str, str]:
    gate = config.get("execution_gate", {})
    if config.get("status") != "frozen_authorized_for_execution":
        raise RuntimeError("unified-grid config is not in the frozen authorized state")
    if gate.get("ready") is not True or gate.get("real_cells_authorized") is not True:
        raise PermissionError("real unified-grid execution is not authorized in config")
    if gate.get("automatic_retry_authorized") is not False:
        raise RuntimeError("unified-grid automatic-retry policy is not fail-closed")
    if gate.get("execution_order") != "status_register_row_order":
        raise RuntimeError("unified-grid execution order is not frozen")
    if gate.get("failure_policy") != "stop_on_first_failed_cell_no_retry":
        raise RuntimeError("unified-grid failure policy is not frozen")

    protocol_path = root / str(config["protocol"]["document"])
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != config["registration"].get("protocol_sha256"):
        raise RuntimeError("protocol hash differs from the frozen unified-grid register")
    implementation_hash = source_tree_sha256(root)
    if implementation_hash != config["registration"].get("implementation_sha256"):
        raise RuntimeError("implementation source-tree hash differs from the frozen register")
    validate_unified_grid(root / "configs" / "unified_grid.yaml", _status_path(root, config))
    return protocol_hash, implementation_hash


def _checkpoint_lightgbm(model: Any, checkpoint_dir: Path, output_mode: str) -> None:
    if output_mode == "point":
        if isinstance(model, dict) or not hasattr(model, "booster_"):
            raise TypeError("point LightGBM cell must produce one fitted estimator")
        model.booster_.save_model(str(checkpoint_dir / "model.txt"))
        return
    if not isinstance(model, dict):
        raise TypeError("quantile LightGBM cell must produce three fitted estimators")
    for quantile, label in ((0.10, "q10"), (0.50, "q50"), (0.90, "q90")):
        model[quantile].booster_.save_model(str(checkpoint_dir / f"model_{label}.txt"))


def run_registered_cell(
    project_root: Path,
    cell: UnifiedGridCell,
    *,
    authorize_real_execution: bool = False,
) -> dict[str, Any]:
    """Execute exactly one pending registered cell without retry or overwrite."""

    if not authorize_real_execution:
        raise PermissionError(
            "real unified-grid execution is gated; pass authorize_real_execution=True"
        )
    root = project_root.resolve()
    config = load_config(root / "configs" / "unified_grid.yaml")
    registered = {candidate.run_id: candidate for candidate in expand_cells(config)}
    if cell.run_id not in registered or registered[cell.run_id] != cell:
        raise ValueError("cell does not exactly match the frozen unified-grid register")
    protocol_hash, implementation_hash = _validate_execution_gate(root, config)
    registered_row = _registered_row(root, config, cell.run_id)
    if registered_row.get("status") != "pending":
        raise RuntimeError("registered unified-grid cell is not pending; duplicate run refused")
    run_dir = root / registered_row["run_path"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")

    controls = config["controls"]
    loaded = load_subset(root / "data" / "interim" / "cmapss", cell.subset)
    train = apply_rul_label(loaded.train, str(controls["target"]["label"]))
    test = apply_rul_label(loaded.test, str(controls["target"]["label"]))
    split_path = root / "configs" / "splits" / f"{cell.subset}_seed42.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    partitions = split_frame_by_unit(train, split_payload["unit_split"])
    selected = sensor_columns(str(controls["features"]["sensor_set"]))
    scaler = fit_train_minmax(partitions["train"], selected)
    transformed = {
        name: transform_features(frame, scaler, selected) for name, frame in partitions.items()
    }
    transformed["test"] = transform_features(test, scaler, selected)
    window_size = int(controls["window"]["length"])
    stride = int(controls["window"]["stride"])
    windows = {
        name: _windows_with_short_endpoint_support(
            frame, selected, window_size=window_size, stride=stride
        )
        for name, frame in transformed.items()
    }

    started_at = datetime.now(UTC)
    start = time.perf_counter()
    environment = config["execution_environment"]
    prediction_splits = ("val", "calib", "test")
    if cell.model == "lightgbm":
        budget = config["model_budgets"]["lightgbm"]
        model = fit_lightgbm(
            windows["train"].features,
            windows["train"].labels,
            seed=cell.seed,
            output_mode=cell.output_mode,
            n_estimators=int(budget["n_estimators"]),
        )
        predictions = {
            split: predict_lightgbm(
                model, windows[split].features, output_mode=cell.output_mode
            )
            for split in prediction_splits
        }
        training_detail = {
            "device": "cpu",
            "estimators_per_head": int(budget["n_estimators"]),
            "quantile_models": 3 if cell.output_mode == "quantile" else 1,
        }
        torch_model = None
    else:
        budget = config["model_budgets"][cell.model]
        fitted = fit_neural(
            cell.model,
            windows["train"].features,
            windows["train"].labels,
            windows["val"].features,
            windows["val"].labels,
            seed=cell.seed,
            output_mode=cell.output_mode,
            epochs=int(budget["epochs_max"]),
            batch_size=int(budget["batch_size"]),
            learning_rate=float(budget["learning_rate"]),
            weight_decay=float(budget["weight_decay"]),
            patience=int(budget["early_stopping_patience"]),
            device=str(environment["neural_device"]),
            inference_batch_size=int(environment["neural_inference_batch_size"]),
        )
        torch_model = fitted.model
        predictions = {
            split: predict_neural(
                torch_model,
                windows[split].features,
                output_mode=cell.output_mode,
                batch_size=int(environment["neural_inference_batch_size"]),
            )
            for split in prediction_splits
        }
        training_detail = {
            "device": fitted.device,
            "epochs_completed": fitted.epochs_completed,
            "best_val_rmse": fitted.best_val_rmse,
        }
    frames = {
        split: _prediction_frame(windows[split], predictions[split])
        for split in prediction_splits
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    with (checkpoint_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if cell.model == "lightgbm":
        _checkpoint_lightgbm(model, checkpoint_dir, cell.output_mode)
    else:
        state = {name: tensor.detach().cpu() for name, tensor in torch_model.state_dict().items()}
        torch.save(state, checkpoint_dir / "model.pt")
    for split, frame in frames.items():
        _write_parquet(frame, run_dir / f"preds_{split}.parquet")

    finished_at = datetime.now(UTC)
    meta = {
        "run_id": cell.run_id,
        "dataset": cell.dataset,
        "subset": cell.subset,
        "model": cell.model,
        "output_mode": cell.output_mode,
        "seed": cell.seed,
        "protocol_version": cell.protocol_version,
        "protocol_sha256": protocol_hash,
        "split_file": f"configs/splits/{cell.subset}_seed42.json",
        "split_sha256": sha256_file(split_path),
        "code_revision": implementation_hash,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "training_seconds": time.perf_counter() - start,
        "status": "completed",
        "metrics": endpoint_metrics(frames["test"]),
        "data_class": "NASA_C-MAPSS_real_unified_registered_cell",
        "controls": controls,
        "training_detail": training_detail,
        "execution_environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "neural_device": environment["neural_device"],
            "lightgbm_device": environment["lightgbm_device"],
            "deterministic_algorithms": True,
            "cublas_workspace_config": environment["cublas_workspace_config"],
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_run_artifacts(run_dir, project_root=root)
    _update_status(root, config, cell.run_id, "completed", "artifact_validation_pass")
    return meta


def run_pending_cells(project_root: Path, *, max_cells: int | None = None) -> dict[str, Any]:
    """Run pending cells sequentially in register order and stop on first error."""

    root = project_root.resolve()
    config = load_config(root / "configs" / "unified_grid.yaml")
    _validate_execution_gate(root, config)
    cells = {cell.run_id: cell for cell in expand_cells(config)}
    with _status_path(root, config).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pending_ids = [row["run_id"] for row in rows if row.get("status") == "pending"]
    if max_cells is not None:
        if max_cells <= 0:
            raise ValueError("max_cells must be positive")
        pending_ids = pending_ids[:max_cells]
    completed: list[dict[str, Any]] = []
    for run_id in pending_ids:
        print(f"START {run_id}", flush=True)
        try:
            meta = run_registered_cell(root, cells[run_id], authorize_real_execution=True)
        except Exception as exc:
            _update_status(root, config, run_id, "failed", f"{type(exc).__name__}: {exc}")
            print(f"FAILED {run_id}: {type(exc).__name__}: {exc}", flush=True)
            raise
        completed.append(
            {
                "run_id": run_id,
                "training_seconds": meta["training_seconds"],
                "rmse": meta["metrics"]["rmse"],
            }
        )
        print(
            f"COMPLETED {run_id} seconds={meta['training_seconds']:.3f} "
            f"rmse={meta['metrics']['rmse']:.6f}",
            flush=True,
        )
    return {"completed_this_invocation": len(completed), "runs": completed}


def synthetic_smoke_test(project_root: Path) -> dict[str, Any]:
    """Exercise all model/output paths on generated data without research outcomes."""

    from rul_audit.data.cmapss import (
        SENSOR_COLUMNS,
        create_unit_split,
        make_windows,
    )
    from rul_audit.experiments.kill_test import _synthetic_frame

    root = project_root.resolve()
    smoke_root = root / "results" / "runs" / "synthetic_unified_smoke"
    if smoke_root.exists():
        raise FileExistsError(f"refusing to overwrite synthetic smoke directory: {smoke_root}")
    smoke_root.mkdir(parents=True)
    protocol_path = root / "protocols" / "unified_v1.md"
    protocol_hash = sha256_file(protocol_path)
    split_path = smoke_root / "synthetic_split.json"
    train = apply_rul_label(
        _synthetic_frame(12, 36, seed=20260812, test=False), "piecewise_125"
    )
    test = apply_rul_label(
        _synthetic_frame(4, 36, seed=20260813, test=True), "piecewise_125"
    )
    unit_split = create_unit_split(list(range(1, 13)), seed=42)
    split_path.write_text(
        json.dumps(
            {
                "dataset": "SYNTHETIC",
                "seed": 42,
                "unit_split": unit_split,
                "fractions": {"train": 0.70, "val": 0.15, "calib": 0.15},
                "allocation_unit": "engine_unit",
                "calib_isolation": "never_used_for_training_or_tuning",
                "status": "synthetic_smoke_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    partitions = split_frame_by_unit(train, unit_split)
    scaler = fit_train_minmax(partitions["train"], SENSOR_COLUMNS)
    transformed = {
        name: transform_features(frame, scaler, SENSOR_COLUMNS)
        for name, frame in partitions.items()
    }
    transformed["test"] = transform_features(test, scaler, SENSOR_COLUMNS)
    windows = {
        name: make_windows(frame, SENSOR_COLUMNS, window_size=10, stride=2)
        for name, frame in transformed.items()
    }

    reports: list[dict[str, Any]] = []
    for model_id, output_mode in product(
        ("lstm", "cnn_1d", "transformer", "lightgbm"), ("point", "quantile")
    ):
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        if model_id == "lightgbm":
            model = fit_lightgbm(
                windows["train"].features,
                windows["train"].labels,
                seed=11,
                output_mode=output_mode,
                n_estimators=3,
            )
            predictions = {
                split: predict_lightgbm(
                    model, windows[split].features, output_mode=output_mode
                )
                for split in ("val", "calib", "test")
            }
            training_detail = {"estimators_per_head": 3}
        else:
            fitted = fit_neural(
                model_id,
                windows["train"].features,
                windows["train"].labels,
                windows["val"].features,
                windows["val"].labels,
                seed=11,
                output_mode=output_mode,
                epochs=1,
                batch_size=64,
                patience=1,
                device="cpu",
                inference_batch_size=256,
            )
            model = fitted.model
            predictions = {
                split: predict_neural(
                    model,
                    windows[split].features,
                    output_mode=output_mode,
                    batch_size=256,
                )
                for split in ("val", "calib", "test")
            }
            training_detail = {
                "epochs_completed": fitted.epochs_completed,
                "best_val_rmse": fitted.best_val_rmse,
            }
        frames = {
            split: _prediction_frame(windows[split], predictions[split])
            for split in ("val", "calib", "test")
        }
        run_id = f"synthetic_unified_smoke__{model_id}__seed11__{output_mode}"
        run_dir = smoke_root / run_id
        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        with (checkpoint_dir / "scaler.pkl").open("wb") as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if model_id == "lightgbm":
            _checkpoint_lightgbm(model, checkpoint_dir, output_mode)
        else:
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        for split, frame in frames.items():
            _write_parquet(frame, run_dir / f"preds_{split}.parquet")
        meta = {
            "run_id": run_id,
            "dataset": "SYNTHETIC",
            "subset": "SMOKE",
            "model": model_id,
            "output_mode": output_mode,
            "seed": 11,
            "protocol_version": "unified_v1",
            "protocol_sha256": protocol_hash,
            "split_file": "results/runs/synthetic_unified_smoke/synthetic_split.json",
            "split_sha256": sha256_file(split_path),
            "code_revision": "working-tree-pre-execution",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "training_seconds": time.perf_counter() - start,
            "status": "completed",
            "metrics": endpoint_metrics(frames["test"]),
            "data_class": "generated_synthetic_smoke_only",
            "training_detail": training_detail,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_run_artifacts(run_dir, project_root=root)
        reports.append({"run_id": run_id, "model": model_id, "output_mode": output_mode})
    summary = {
        "schema_version": 1,
        "status": "passed",
        "data_class": "generated_synthetic_smoke_only",
        "real_unified_cells_executed": 0,
        "validated_runs": len(reports),
        "runs": reports,
    }
    (smoke_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def summarize_status(project_root: Path) -> dict[str, int]:
    config = load_config(project_root / "configs" / "unified_grid.yaml")
    with _status_path(project_root, config).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--run-pending", action="store_true")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--authorize-real-execution", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args(argv)
    output: dict[str, Any] = {}
    if args.status:
        output["status"] = summarize_status(args.root)
    if args.synthetic_smoke:
        output["synthetic_smoke"] = synthetic_smoke_test(args.root)
    if args.run_id is not None:
        if not args.authorize_real_execution:
            parser.error("--run-id requires --authorize-real-execution")
        config = load_config(args.root / "configs" / "unified_grid.yaml")
        cells = {cell.run_id: cell for cell in expand_cells(config)}
        if args.run_id not in cells:
            parser.error(f"unknown registered run ID: {args.run_id}")
        output["run"] = run_registered_cell(
            args.root, cells[args.run_id], authorize_real_execution=True
        )
    if args.run_pending:
        if not args.authorize_real_execution:
            parser.error("--run-pending requires --authorize-real-execution")
        output["batch"] = run_pending_cells(args.root, max_cells=args.max_cells)
    if not output:
        parser.error("select --status, --synthetic-smoke, --run-id, and/or --run-pending")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
