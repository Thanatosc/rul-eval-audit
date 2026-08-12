"""Kill Test registration, decision logic, and synthetic-only smoke execution."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from rul_audit.data.cmapss import (
    SENSOR_COLUMNS,
    apply_rul_label,
    create_unit_split,
    fit_train_minmax,
    make_windows,
    sha256_file,
    split_frame_by_unit,
    transform_features,
)
from rul_audit.metrics.rul import endpoint_metrics
from rul_audit.models.baselines import (
    fit_lightgbm,
    fit_neural,
    predict_lightgbm,
    predict_neural,
)
from rul_audit.protocols.assets import validate_run_artifacts

STATUS_COLUMNS = (
    "run_id",
    "dataset",
    "model",
    "rul_label",
    "sensor_set",
    "seed",
    "status",
    "run_path",
    "notes",
)


@dataclass(frozen=True)
class KillTestCell:
    run_id: str
    dataset: str
    model: str
    rul_label: str
    sensor_set: str
    seed: int


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Kill Test config must be a YAML mapping")
    return payload


def expand_cells(config: dict[str, Any]) -> list[KillTestCell]:
    template = config.get(
        "run_id_template",
        "kill_v1__{dataset_lower}__{model}__{rul_label}__{sensor_set}__seed{seed}",
    )
    cells = []
    for dataset, model, rul_label, sensor_set, seed in product(
        config["datasets"],
        config["models"],
        config["factors"]["rul_label"],
        config["factors"]["sensor_set"],
        config["seeds"],
    ):
        run_id = template.format(
            dataset=dataset,
            dataset_lower=dataset.lower(),
            model=model,
            rul_label=rul_label,
            sensor_set=sensor_set,
            seed=seed,
        )
        cells.append(
            KillTestCell(
                run_id=run_id,
                dataset=dataset,
                model=model,
                rul_label=rul_label,
                sensor_set=sensor_set,
                seed=int(seed),
            )
        )
    expected = int(config["registration"]["primary_cells"])
    if len(cells) != expected or len({cell.run_id for cell in cells}) != expected:
        raise ValueError(f"Kill Test expands to {len(cells)} non-unique cells; expected {expected}")
    return cells


def write_status_register(config_path: Path, output_path: Path) -> int:
    cells = expand_cells(load_config(config_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **asdict(cell),
                    "status": "pending",
                    "run_path": f"results/runs/{cell.run_id}",
                    "notes": "execution_not_authorized",
                }
            )
    return len(cells)


def decide_kill_test(
    *,
    max_absolute_rmse_effect: float,
    ranking_reversals: int,
    completed_cells: int,
    registered_cells: int = 120,
) -> str:
    """Apply the frozen three-state decision rule without outcome-dependent repair."""

    if completed_cells != registered_cells:
        return "INCONCLUSIVE"
    if max_absolute_rmse_effect >= 1.0 or ranking_reversals > 0:
        return "PASS"
    if max_absolute_rmse_effect < 0.5 and ranking_reversals == 0:
        return "FAIL"
    return "INCONCLUSIVE"


def _synthetic_frame(unit_count: int, cycles: int, *, seed: int, test: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for unit_id in range(1, unit_count + 1):
        offset = 8.0 + float(unit_id % 5) if test else 0.0
        for cycle in range(1, cycles + 1):
            raw_rul = float(cycles - cycle) + offset
            progress = cycle / cycles
            row: dict[str, float | int] = {
                "unit_id": unit_id,
                "cycle": cycle,
                "raw_rul": raw_rul,
            }
            for index, column in enumerate(SENSOR_COLUMNS, start=1):
                row[column] = (
                    progress * (0.25 + index / 42.0)
                    + unit_id * 0.002
                    + rng.normal(0.0, 0.005)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_frame(windows: Any, predictions: np.ndarray) -> pd.DataFrame:
    if predictions.ndim == 1:
        return pd.DataFrame(
            {
                "unit_id": windows.unit_ids,
                "cycle": windows.cycles,
                "true_rul": windows.labels.astype(float),
                "pred_rul": predictions.astype(float),
            }
        )
    ordered = np.sort(predictions, axis=1)
    return pd.DataFrame(
        {
            "unit_id": windows.unit_ids,
            "cycle": windows.cycles,
            "true_rul": windows.labels.astype(float),
            "pred_rul": ordered[:, 1].astype(float),
            "pred_q10": ordered[:, 0].astype(float),
            "pred_q50": ordered[:, 1].astype(float),
            "pred_q90": ordered[:, 2].astype(float),
        }
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    schema = {
        "unit_id": pa.int64(),
        "cycle": pa.int64(),
        "true_rul": pa.float64(),
        "pred_rul": pa.float64(),
        "pred_q10": pa.float64(),
        "pred_q50": pa.float64(),
        "pred_q90": pa.float64(),
    }
    arrays = {column: pa.array(frame[column], type=schema[column]) for column in frame.columns}
    pq.write_table(pa.table(arrays), path)


def _concat_windows(first: Any, second: Any) -> Any:
    """Concatenate WindowedData objects without losing row provenance."""

    from rul_audit.data.cmapss import WindowedData

    return WindowedData(
        features=np.concatenate([first.features, second.features], axis=0),
        labels=np.concatenate([first.labels, second.labels], axis=0),
        unit_ids=np.concatenate([first.unit_ids, second.unit_ids], axis=0),
        cycles=np.concatenate([first.cycles, second.cycles], axis=0),
        sensor_columns=first.sensor_columns,
    )


def _windows_with_short_endpoint_support(
    frame: pd.DataFrame, sensor_columns: tuple[str, ...], *, window_size: int, stride: int
) -> Any:
    """Keep all complete windows and add a padded endpoint only for short engines."""

    from rul_audit.data.cmapss import WindowedData

    complete = make_windows(
        frame,
        sensor_columns,
        window_size=window_size,
        stride=stride,
        endpoint_only=False,
    )
    covered = {int(value) for value in complete.unit_ids}
    short_units = sorted({int(value) for value in frame["unit_id"]} - covered)
    if not short_units:
        return complete
    padded = make_windows(
        frame.loc[frame["unit_id"].isin(short_units)],
        sensor_columns,
        window_size=window_size,
        stride=stride,
        endpoint_only=True,
        pad_short_endpoint=True,
    )
    if not isinstance(complete, WindowedData) or not isinstance(padded, WindowedData):
        raise TypeError("window builder returned an unexpected object")
    return _concat_windows(complete, padded)


def _update_status(project_root: Path, run_id: str, status: str, notes: str) -> None:
    status_path = project_root / "results" / "KILL_TEST_STATUS.csv"
    with status_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # DictReader field names are stable; use the registered schema explicitly.
    if not any(row.get("run_id") == run_id for row in rows):
        raise ValueError(f"run_id is not registered: {run_id}")
    for row in rows:
        if row.get("run_id") == run_id:
            row["status"] = status
            row["notes"] = notes
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _registered_status(project_root: Path, run_id: str) -> str:
    status_path = project_root / "results" / "KILL_TEST_STATUS.csv"
    with status_path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registered status row for {run_id}, got {len(matches)}")
    return matches[0].get("status", "")


def run_registered_cell(
    project_root: Path,
    cell: KillTestCell,
    *,
    authorize_real_execution: bool = False,
) -> dict[str, Any]:
    """Run one exact registered C-MAPSS cell after an explicit authorization flag.

    This entry point is intentionally unreachable without ``authorize_real_execution``.
    It does not alter factors, thresholds, seeds, or missing-run policy.
    """

    if not authorize_real_execution:
        raise PermissionError(
            "real Kill Test execution is gated; pass authorize_real_execution=True"
        )
    from rul_audit.data.cmapss import load_subset, sensor_columns
    from rul_audit.protocols.readiness import source_tree_sha256

    root = project_root.resolve()
    config = load_config(root / "configs" / "kill_test.yaml")
    if config.get("status") != "frozen_pending_execution_authorization":
        raise RuntimeError("Kill Test configuration is not in the frozen execution-gate state")
    registered = {candidate.run_id: candidate for candidate in expand_cells(config)}
    if cell.run_id not in registered or registered[cell.run_id] != cell:
        raise ValueError("cell does not exactly match the frozen Kill Test register")
    implementation_hash = source_tree_sha256(root)
    if implementation_hash != config["registration"].get("implementation_sha256"):
        raise RuntimeError("implementation source-tree hash differs from the frozen register")
    protocol_hash = sha256_file(root / "protocols" / "unified_v1.md")
    if protocol_hash != config["registration"].get("protocol_sha256"):
        raise RuntimeError("protocol hash differs from the frozen register")
    if _registered_status(root, cell.run_id) != "pending":
        raise RuntimeError("registered cell is not pending; duplicate execution is refused")
    run_dir = root / "results" / "runs" / cell.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")

    loaded = load_subset(root / "data" / "interim" / "cmapss", cell.dataset)
    train = apply_rul_label(loaded.train, cell.rul_label)
    test = apply_rul_label(loaded.test, cell.rul_label)
    split_path = root / "configs" / "splits" / f"{cell.dataset}_seed42.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    partitions = split_frame_by_unit(train, split_payload["unit_split"])
    selected = sensor_columns(cell.sensor_set)
    scaler = fit_train_minmax(partitions["train"], selected)
    transformed = {
        name: transform_features(frame, scaler, selected)
        for name, frame in partitions.items()
    }
    transformed["test"] = transform_features(test, scaler, selected)
    window_size = int(config["controls"]["window"]["length"])
    stride = int(config["controls"]["window"]["stride"])
    windows = {
        name: _windows_with_short_endpoint_support(
            frame, selected, window_size=window_size, stride=stride
        )
        for name, frame in transformed.items()
    }

    started_at = datetime.now(UTC)
    start = time.perf_counter()
    prediction_splits = ("val", "calib", "test")
    if cell.model == "lightgbm":
        model = fit_lightgbm(
            windows["train"].features,
            windows["train"].labels,
            seed=cell.seed,
            n_estimators=int(config["model_budgets"]["lightgbm"]["n_estimators"]),
        )
        predictions = {
            split: predict_lightgbm(model, windowed.features)
            for split, windowed in windows.items()
            if split in prediction_splits
        }
        torch_model = None
        training_detail = {
            "device": "cpu",
            "estimators": int(config["model_budgets"]["lightgbm"]["n_estimators"]),
        }
    else:
        budget = config["model_budgets"][cell.model]
        environment = config["execution_environment"]
        fitted = fit_neural(
            cell.model,
            windows["train"].features,
            windows["train"].labels,
            windows["val"].features,
            windows["val"].labels,
            seed=cell.seed,
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
                windowed.features,
                batch_size=int(environment["neural_inference_batch_size"]),
            )
            for split, windowed in windows.items()
            if split in prediction_splits
        }
        training_detail = {
            "device": fitted.device,
            "epochs_completed": fitted.epochs_completed,
            "best_val_rmse": fitted.best_val_rmse,
        }
    frames = {
        split: _prediction_frame(windows[split], predictions[split])
        for split in ("val", "calib", "test")
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if cell.model == "lightgbm":
        if not hasattr(model, "booster_"):
            raise TypeError("point LightGBM cell must produce one fitted estimator")
        model.booster_.save_model(str(checkpoint_dir / "model.txt"))
    else:
        portable_state = {
            name: tensor.detach().cpu() for name, tensor in torch_model.state_dict().items()
        }
        torch.save(portable_state, checkpoint_dir / "model.pt")
    for split, frame in frames.items():
        _write_parquet(frame, run_dir / f"preds_{split}.parquet")
    finished_at = datetime.now(UTC)
    meta = {
        "run_id": cell.run_id,
        "dataset": "C-MAPSS",
        "subset": cell.dataset,
        "model": cell.model,
        "output_mode": "point",
        "seed": cell.seed,
        "protocol_version": "unified_v1",
        "protocol_sha256": protocol_hash,
        "split_file": f"configs/splits/{cell.dataset}_seed42.json",
        "split_sha256": sha256_file(split_path),
        "code_revision": implementation_hash,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "training_seconds": time.perf_counter() - start,
        "status": "completed",
        "metrics": endpoint_metrics(frames["test"]),
        "data_class": "NASA_C-MAPSS_real_registered_cell",
        "training_detail": training_detail,
        "execution_environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "neural_device": config["execution_environment"]["neural_device"],
            "lightgbm_device": config["execution_environment"]["lightgbm_device"],
            "deterministic_algorithms": True,
            "cublas_workspace_config": config["execution_environment"][
                "cublas_workspace_config"
            ],
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_run_artifacts(run_dir, project_root=root)
    _update_status(root, cell.run_id, "completed", "artifact_validation_pass")
    return meta


def synthetic_smoke_test(project_root: Path) -> dict[str, Any]:
    """Exercise all Kill Test model paths on generated data, never on C-MAPSS."""

    smoke_root = project_root / "results" / "runs" / "synthetic_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    protocol_path = project_root / "protocols" / "unified_v1.md"
    protocol_hash = sha256_file(protocol_path)
    split_path = smoke_root / "synthetic_split.json"

    base_train = apply_rul_label(
        _synthetic_frame(12, 36, seed=20260811, test=False), "piecewise_125"
    )
    base_test = apply_rul_label(
        _synthetic_frame(4, 36, seed=20260812, test=True), "piecewise_125"
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
    split_hash = sha256_file(split_path)
    partitions = split_frame_by_unit(base_train, unit_split)
    scaler = fit_train_minmax(partitions["train"], SENSOR_COLUMNS)
    transformed = {
        name: transform_features(frame, scaler, SENSOR_COLUMNS)
        for name, frame in partitions.items()
    }
    transformed["test"] = transform_features(base_test, scaler, SENSOR_COLUMNS)
    windows = {
        name: make_windows(frame, SENSOR_COLUMNS, window_size=10, stride=2)
        for name, frame in transformed.items()
    }

    reports: list[dict[str, Any]] = []
    for model_id in ("lstm", "cnn_1d", "lightgbm"):
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        if model_id == "lightgbm":
            model = fit_lightgbm(
                windows["train"].features,
                windows["train"].labels,
                seed=11,
                n_estimators=12,
            )
            predict = partial(predict_lightgbm, model)
            training_detail = {"estimators": 12}
        else:
            fitted = fit_neural(
                model_id,
                windows["train"].features,
                windows["train"].labels,
                windows["val"].features,
                windows["val"].labels,
                seed=11,
                epochs=2,
                batch_size=64,
                patience=2,
            )
            model = fitted.model
            predict = partial(predict_neural, model)
            training_detail = {
                "epochs_completed": fitted.epochs_completed,
                "best_val_rmse": fitted.best_val_rmse,
            }

        frames = {
            split: _prediction_frame(windowed, predict(windowed.features))
            for split, windowed in windows.items()
            if split in {"val", "calib", "test"}
        }
        run_id = f"synthetic_smoke__{model_id}__seed11__point"
        run_dir = smoke_root / run_id
        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if model_id == "lightgbm":
            model.booster_.save_model(str(checkpoint_dir / "model.txt"))
        else:
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        for split, frame in frames.items():
            _write_parquet(frame, run_dir / f"preds_{split}.parquet")
        elapsed = time.perf_counter() - start
        finished_at = datetime.now(UTC)
        metrics = endpoint_metrics(frames["test"])
        meta = {
            "run_id": run_id,
            "dataset": "SYNTHETIC",
            "subset": "SMOKE",
            "model": model_id,
            "output_mode": "point",
            "seed": 11,
            "protocol_version": "unified_v1",
            "protocol_sha256": protocol_hash,
            "split_file": "results/runs/synthetic_smoke/synthetic_split.json",
            "split_sha256": split_hash,
            "code_revision": "working-tree-pre-execution",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "training_seconds": elapsed,
            "status": "completed",
            "metrics": metrics,
            "data_class": "generated_synthetic_smoke_only",
            "training_detail": training_detail,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(
            {
                "run_id": run_id,
                "run_path": str(run_dir.relative_to(project_root)).replace("\\", "/"),
                "model": model_id,
                "metrics": metrics,
            }
        )

    summary = {
        "schema_version": 1,
        "status": "passed",
        "data_class": "generated_synthetic_smoke_only",
        "real_kill_test_cells_executed": 0,
        "models_exercised": ["lstm", "cnn_1d", "lightgbm"],
        "runs": reports,
    }
    (smoke_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-status", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--authorize-real-execution", action="store_true")
    args = parser.parse_args()
    output: dict[str, Any] = {}
    if args.write_status:
        output["registered_cells"] = write_status_register(
            args.root / "configs" / "kill_test.yaml",
            args.root / "results" / "KILL_TEST_STATUS.csv",
        )
    if args.synthetic_smoke:
        output["synthetic_smoke"] = synthetic_smoke_test(args.root)
    if args.run_id is not None:
        if not args.authorize_real_execution:
            parser.error("--run-id requires --authorize-real-execution")
        config = load_config(args.root / "configs" / "kill_test.yaml")
        cells = {cell.run_id: cell for cell in expand_cells(config)}
        if args.run_id not in cells:
            parser.error(f"unknown registered run ID: {args.run_id}")
        output["run"] = run_registered_cell(
            args.root, cells[args.run_id], authorize_real_execution=True
        )
    if not output:
        parser.error("select --write-status and/or --synthetic-smoke")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
