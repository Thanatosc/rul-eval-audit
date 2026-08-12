"""Validation for the Topic 6/Topic 7 shared experiment assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

EXPECTED_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_MODELS = ("lstm", "cnn_1d", "transformer", "lightgbm")
EXPECTED_SEEDS = (11, 23, 37, 53, 71)
EXPECTED_OUTPUT_MODES = ("point", "quantile")
EXPECTED_GRID_CELLS = 160

REQUIRED_META_KEYS = frozenset(
    {
        "run_id",
        "dataset",
        "subset",
        "model",
        "output_mode",
        "seed",
        "protocol_version",
        "protocol_sha256",
        "split_file",
        "split_sha256",
        "code_revision",
        "started_at",
        "finished_at",
        "training_seconds",
        "status",
        "metrics",
    }
)
BASE_PREDICTION_COLUMNS = ("unit_id", "cycle", "true_rul", "pred_rul")
QUANTILE_COLUMNS = ("pred_q10", "pred_q50", "pred_q90")
PREDICTION_FILES = ("preds_val.parquet", "preds_calib.parquet", "preds_test.parquet")
STATUS_COLUMNS = frozenset(
    {
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
    }
)
ALLOWED_RUN_STATUSES = frozenset({"pending", "running", "completed", "failed", "blocked"})
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class AssetValidationError(ValueError):
    """Raised when one or more shared-asset contract checks fail."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetValidationError([f"{path}: expected a YAML mapping"])
    return value


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetValidationError([f"{path}: expected a JSON object"])
    return value


def _expected_grid_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_entries = config.get("datasets")
    backbones = config.get("backbones")
    seeds = config.get("seeds")
    output_modes = config.get("output_modes")
    template = config.get("run_id_template")

    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise AssetValidationError(["grid config: datasets must be a non-empty list"])
    if not isinstance(backbones, list) or not backbones:
        raise AssetValidationError(["grid config: backbones must be a non-empty list"])
    if not isinstance(seeds, list) or not seeds:
        raise AssetValidationError(["grid config: seeds must be a non-empty list"])
    if not isinstance(output_modes, dict) or not output_modes:
        raise AssetValidationError(["grid config: output_modes must be a non-empty mapping"])
    if not isinstance(template, str):
        raise AssetValidationError(["grid config: run_id_template must be a string"])

    dataset_subsets: list[tuple[str, str]] = []
    for entry in dataset_entries:
        if not isinstance(entry, dict):
            raise AssetValidationError(["grid config: each dataset entry must be a mapping"])
        dataset = entry.get("dataset")
        subsets = entry.get("subsets")
        if not isinstance(dataset, str) or not isinstance(subsets, list):
            raise AssetValidationError(["grid config: dataset name/subsets are malformed"])
        dataset_subsets.extend((dataset, subset) for subset in subsets)

    model_ids = [entry.get("id") for entry in backbones if isinstance(entry, dict)]
    rows: list[dict[str, Any]] = []
    for (dataset, subset), model, seed, output_mode in product(
        dataset_subsets,
        model_ids,
        seeds,
        output_modes,
    ):
        run_id = template.format(
            subset=subset,
            subset_lower=str(subset).lower(),
            model=model,
            seed=seed,
            output_mode=output_mode,
        )
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "seed": int(seed),
                "output_mode": output_mode,
                "protocol_version": config.get("protocol", {}).get("version"),
                "run_path": f"results/runs/{run_id}",
            }
        )
    return rows


def validate_unified_grid(config_path: Path, status_path: Path) -> dict[str, int]:
    """Validate the frozen dimensions and the expanded 160-cell status register."""

    config = _load_yaml_mapping(config_path)
    errors: list[str] = []

    dataset_entries = config.get("datasets", [])
    observed_subsets = [
        subset
        for entry in dataset_entries
        if isinstance(entry, dict)
        for subset in entry.get("subsets", [])
    ]
    if tuple(observed_subsets) != EXPECTED_SUBSETS:
        errors.append(f"grid subsets must be {EXPECTED_SUBSETS}, got {tuple(observed_subsets)}")

    observed_models = tuple(
        entry.get("id") for entry in config.get("backbones", []) if isinstance(entry, dict)
    )
    if observed_models != EXPECTED_MODELS:
        errors.append(f"grid models must be {EXPECTED_MODELS}, got {observed_models}")

    observed_seeds = tuple(config.get("seeds", []))
    if observed_seeds != EXPECTED_SEEDS:
        errors.append(f"grid seeds must be {EXPECTED_SEEDS}, got {observed_seeds}")

    observed_modes = tuple(config.get("output_modes", {}).keys())
    if observed_modes != EXPECTED_OUTPUT_MODES:
        errors.append(f"grid output modes must be {EXPECTED_OUTPUT_MODES}, got {observed_modes}")

    if config.get("expected_cells") != EXPECTED_GRID_CELLS:
        errors.append(f"grid expected_cells must equal {EXPECTED_GRID_CELLS}")

    expected_rows = _expected_grid_rows(config)
    if len(expected_rows) != EXPECTED_GRID_CELLS:
        errors.append(f"grid expands to {len(expected_rows)} cells, expected {EXPECTED_GRID_CELLS}")

    with status_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        status_rows = list(reader)
        headers = frozenset(reader.fieldnames or [])

    missing_headers = STATUS_COLUMNS - headers
    if missing_headers:
        errors.append(f"status register missing columns: {sorted(missing_headers)}")
    if len(status_rows) != EXPECTED_GRID_CELLS:
        errors.append(
            f"status register contains {len(status_rows)} rows, expected {EXPECTED_GRID_CELLS}"
        )

    run_ids = [row.get("run_id", "") for row in status_rows]
    if len(set(run_ids)) != len(run_ids):
        errors.append("status register has duplicate run_id values")

    expected_by_key = {
        (row["dataset"], row["subset"], row["model"], row["seed"], row["output_mode"]): row
        for row in expected_rows
    }
    observed_by_key: dict[tuple[str, str, str, int, str], dict[str, str]] = {}
    for row_number, row in enumerate(status_rows, start=2):
        try:
            seed = int(row.get("seed", ""))
        except ValueError:
            errors.append(f"status row {row_number}: seed is not an integer")
            continue
        key = (
            row.get("dataset", ""),
            row.get("subset", ""),
            row.get("model", ""),
            seed,
            row.get("output_mode", ""),
        )
        if key in observed_by_key:
            errors.append(f"status register has duplicate grid cell {key}")
        observed_by_key[key] = row

        expected = expected_by_key.get(key)
        if expected is None:
            errors.append(f"status row {row_number}: unexpected grid cell {key}")
            continue
        for field in ("run_id", "protocol_version", "run_path"):
            if row.get(field) != str(expected[field]):
                errors.append(
                    f"status row {row_number}: {field}={row.get(field)!r}, "
                    f"expected {expected[field]!r}"
                )
        if row.get("status") not in ALLOWED_RUN_STATUSES:
            errors.append(f"status row {row_number}: invalid status {row.get('status')!r}")

    missing_cells = set(expected_by_key) - set(observed_by_key)
    if missing_cells:
        errors.append(f"status register is missing {len(missing_cells)} expected grid cells")

    if errors:
        raise AssetValidationError(errors)
    return {"expected_cells": len(expected_rows), "status_rows": len(status_rows)}


def validate_unit_split(split_path: Path, *, require_populated: bool = True) -> dict[str, int]:
    """Validate engine-unit allocation and calibration-set isolation."""

    payload = _load_json_mapping(split_path)
    errors: list[str] = []
    if payload.get("allocation_unit") != "engine_unit":
        errors.append("split allocation_unit must be engine_unit")
    if payload.get("calib_isolation") != "never_used_for_training_or_tuning":
        errors.append("split calib_isolation contract is missing or invalid")

    fractions = payload.get("fractions")
    expected_fractions = {"train": 0.70, "val": 0.15, "calib": 0.15}
    if not isinstance(fractions, dict):
        errors.append("split fractions must be a mapping")
    else:
        for name, expected in expected_fractions.items():
            value = fractions.get(name)
            if not isinstance(value, (int, float)) or not math.isclose(value, expected):
                errors.append(f"split fraction {name} must equal {expected}")

    unit_split = payload.get("unit_split")
    groups: dict[str, list[Any]] = {}
    if not isinstance(unit_split, dict):
        errors.append("unit_split must be a mapping")
    else:
        for name in expected_fractions:
            values = unit_split.get(name)
            if not isinstance(values, list):
                errors.append(f"unit_split.{name} must be a list")
                continue
            groups[name] = values
            canonical = [str(value) for value in values]
            if len(set(canonical)) != len(canonical):
                errors.append(f"unit_split.{name} contains duplicate unit IDs")
            if require_populated and not values:
                errors.append(f"unit_split.{name} must not be empty")

    for left, right in (("train", "val"), ("train", "calib"), ("val", "calib")):
        overlap = {str(value) for value in groups.get(left, [])} & {
            str(value) for value in groups.get(right, [])
        }
        if overlap:
            errors.append(f"unit split overlap between {left} and {right}: {sorted(overlap)}")

    if errors:
        raise AssetValidationError(errors)
    return {name: len(values) for name, values in groups.items()}


def _validate_datetime(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str):
        errors.append(f"meta.{field} must be an ISO-8601 string")
        return
    try:
        datetime.fromisoformat(value)
    except ValueError:
        errors.append(f"meta.{field} is not a valid ISO-8601 timestamp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prediction_table(path: Path, output_mode: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing prediction table: {path.name}")
        return

    schema = pq.read_schema(path)
    required = list(BASE_PREDICTION_COLUMNS)
    if output_mode == "quantile":
        required.extend(QUANTILE_COLUMNS)
    missing = [name for name in required if name not in schema.names]
    if missing:
        errors.append(f"{path.name} missing columns: {missing}")
        return

    unit_type = schema.field("unit_id").type
    if not (pa.types.is_integer(unit_type) or pa.types.is_string(unit_type)):
        errors.append(f"{path.name}.unit_id must have an integer or string Arrow dtype")
    if not pa.types.is_integer(schema.field("cycle").type):
        errors.append(f"{path.name}.cycle must have an integer Arrow dtype")
    for column in required[2:]:
        if not pa.types.is_floating(schema.field(column).type):
            errors.append(f"{path.name}.{column} must have a floating Arrow dtype")

    table = pq.read_table(path, columns=required)
    if table.num_rows == 0:
        errors.append(f"{path.name} must contain at least one prediction row")
    for column in required:
        if table[column].null_count:
            errors.append(f"{path.name}.{column} contains null values")

    if output_mode == "quantile" and table.num_rows:
        frame = table.select(["pred_rul", *QUANTILE_COLUMNS]).to_pandas()
        ordered = (frame["pred_q10"] <= frame["pred_q50"]) & (
            frame["pred_q50"] <= frame["pred_q90"]
        )
        if not bool(ordered.all()):
            errors.append(f"{path.name} violates q10 <= q50 <= q90")
        if not bool((frame["pred_rul"] - frame["pred_q50"]).abs().le(1e-8).all()):
            errors.append(f"{path.name}.pred_rul must equal pred_q50 for quantile runs")


def validate_run_artifacts(run_dir: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    """Validate one completed run directory against the shared artifact contract."""

    errors: list[str] = []
    checkpoint_dir = run_dir / "checkpoint"
    if not checkpoint_dir.is_dir() or not any(path.is_file() for path in checkpoint_dir.rglob("*")):
        errors.append("checkpoint directory must exist and contain at least one file")

    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise AssetValidationError([*errors, "missing meta.json"])
    meta = _load_json_mapping(meta_path)

    missing_meta = REQUIRED_META_KEYS - meta.keys()
    if missing_meta:
        errors.append(f"meta.json missing keys: {sorted(missing_meta)}")

    output_mode = meta.get("output_mode")
    if output_mode not in EXPECTED_OUTPUT_MODES:
        errors.append(f"meta.output_mode must be one of {EXPECTED_OUTPUT_MODES}")
    if meta.get("run_id") != run_dir.name:
        errors.append("meta.run_id must match the run directory name")
    if not isinstance(meta.get("seed"), int) or isinstance(meta.get("seed"), bool):
        errors.append("meta.seed must be an integer")
    if meta.get("status") != "completed":
        errors.append("meta.status must be completed for artifact validation")
    if not isinstance(meta.get("metrics"), dict):
        errors.append("meta.metrics must be an object")
    training_seconds = meta.get("training_seconds")
    if not isinstance(training_seconds, (int, float)) or training_seconds < 0:
        errors.append("meta.training_seconds must be a non-negative number")
    for field in ("protocol_sha256", "split_sha256"):
        if not isinstance(meta.get(field), str) or not SHA256_RE.fullmatch(meta[field]):
            errors.append(f"meta.{field} must be a 64-character SHA-256 hex digest")
    for field in ("started_at", "finished_at"):
        _validate_datetime(meta.get(field), field, errors)

    if project_root is not None:
        split_file = meta.get("split_file")
        if isinstance(split_file, str):
            split_path = project_root / split_file
            if not split_path.is_file():
                errors.append(f"meta.split_file does not exist: {split_file}")
            elif (
                isinstance(meta.get("split_sha256"), str)
                and _sha256(split_path).lower() != meta["split_sha256"].lower()
            ):
                errors.append("meta.split_sha256 does not match split_file")

        protocol_version = meta.get("protocol_version")
        if isinstance(protocol_version, str):
            protocol_path = project_root / "protocols" / f"{protocol_version}.md"
            if not protocol_path.is_file():
                errors.append(f"protocol document does not exist: {protocol_path}")
            elif (
                isinstance(meta.get("protocol_sha256"), str)
                and _sha256(protocol_path).lower() != meta["protocol_sha256"].lower()
            ):
                errors.append("meta.protocol_sha256 does not match the protocol document")

    if output_mode in EXPECTED_OUTPUT_MODES:
        for filename in PREDICTION_FILES:
            _validate_prediction_table(run_dir / filename, output_mode, errors)

    if errors:
        raise AssetValidationError(errors)
    return {"run_id": meta["run_id"], "output_mode": output_mode, "prediction_tables": 3}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", action="append", type=Path, default=[])
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    args = parser.parse_args(argv)

    root = args.root.resolve()
    report: dict[str, Any] = {
        "grid": validate_unified_grid(
            root / "configs" / "unified_grid.yaml",
            root / "results" / "UNIFIED_GRID_STATUS.csv",
        ),
        "splits": {},
        "runs": {},
    }
    for path in args.split:
        resolved = path if path.is_absolute() else root / path
        report["splits"][str(path)] = validate_unit_split(resolved)
    for path in args.run_dir:
        resolved = path if path.is_absolute() else root / path
        report["runs"][str(path)] = validate_run_artifacts(resolved, project_root=root)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
