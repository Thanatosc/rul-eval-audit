"""C-MAPSS loading, unit splitting, scaling, and window construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

OPERATION_COLUMNS = tuple(f"op_setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
CMAPSS_COLUMNS = ("unit_id", "cycle", *OPERATION_COLUMNS, *SENSOR_COLUMNS)
COMMON_14_SENSOR_IDS = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
COMMON_14_SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in COMMON_14_SENSOR_IDS)
EXPECTED_DATASET_SHAPES = {
    "FD001": {"train_rows": 20631, "test_rows": 13096, "train_units": 100, "test_units": 100},
    "FD002": {"train_rows": 53759, "test_rows": 33991, "train_units": 260, "test_units": 259},
    "FD003": {"train_rows": 24720, "test_rows": 16596, "train_units": 100, "test_units": 100},
    "FD004": {"train_rows": 61249, "test_rows": 41214, "train_units": 249, "test_units": 248},
}


@dataclass(frozen=True)
class CMAPSSSubset:
    """Loaded training and official-test trajectories with uncapped RUL."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class WindowedData:
    """Sliding-window features and row provenance."""

    features: np.ndarray
    labels: np.ndarray
    unit_ids: np.ndarray
    cycles: np.ndarray
    sensor_columns: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trajectory(path: Path) -> pd.DataFrame:
    """Read one 26-column C-MAPSS trajectory file."""

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected 26")
    frame.columns = CMAPSS_COLUMNS
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{path} is empty or contains missing values")
    frame["unit_id"] = frame["unit_id"].astype(np.int64)
    frame["cycle"] = frame["cycle"].astype(np.int64)
    if bool((frame[["unit_id", "cycle"]] <= 0).any().any()):
        raise ValueError(f"{path} contains non-positive unit or cycle identifiers")
    if bool(frame.duplicated(["unit_id", "cycle"]).any()):
        raise ValueError(f"{path} contains duplicate unit-cycle rows")
    return frame


def read_test_rul(path: Path, unit_ids: list[int]) -> dict[int, float]:
    """Join the ordered official RUL vector to sorted test engine IDs."""

    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(float).tolist()
    sorted_units = sorted(unit_ids)
    if len(values) != len(sorted_units):
        raise ValueError(
            f"{path} contains {len(values)} offsets for {len(sorted_units)} test engines"
        )
    return dict(zip(sorted_units, values, strict=True))


def load_subset(data_dir: Path, subset: str) -> CMAPSSSubset:
    """Load one subset and derive uncapped cycle-level RUL for train and test."""

    subset = subset.upper()
    if subset not in EXPECTED_DATASET_SHAPES:
        raise ValueError(f"unsupported subset: {subset}")
    train = read_trajectory(data_dir / f"train_{subset}.txt")
    test = read_trajectory(data_dir / f"test_{subset}.txt")

    train_max = train.groupby("unit_id")["cycle"].transform("max")
    train["raw_rul"] = (train_max - train["cycle"]).astype(float)

    unit_ids = test["unit_id"].drop_duplicates().astype(int).tolist()
    offsets = read_test_rul(data_dir / f"RUL_{subset}.txt", unit_ids)
    observed_max = test.groupby("unit_id")["cycle"].transform("max")
    test["raw_rul"] = (
        observed_max - test["cycle"] + test["unit_id"].map(offsets)
    ).astype(float)
    return CMAPSSSubset(subset=subset, train=train, test=test)


def apply_rul_label(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Create the registered evaluation target without changing raw RUL."""

    result = frame.copy()
    if label == "piecewise_125":
        result["true_rul"] = result["raw_rul"].clip(upper=125.0)
    elif label == "linear_uncapped":
        result["true_rul"] = result["raw_rul"].astype(float)
    else:
        raise ValueError(f"unsupported RUL label: {label}")
    return result


def sensor_columns(sensor_set: str) -> tuple[str, ...]:
    if sensor_set == "all_21":
        return SENSOR_COLUMNS
    if sensor_set == "common_14":
        return COMMON_14_SENSOR_COLUMNS
    raise ValueError(f"unsupported sensor set: {sensor_set}")


def create_unit_split(unit_ids: list[int], seed: int = 42) -> dict[str, list[int]]:
    """Deterministically allocate complete engines to 70/15/15 groups."""

    units = np.asarray(sorted({int(unit) for unit in unit_ids}), dtype=np.int64)
    if len(units) < 3:
        raise ValueError("at least three engine units are required")
    shuffled = np.random.default_rng(seed).permutation(units)
    train_count = round(len(units) * 0.70)
    val_count = round((len(units) - train_count) / 2)
    split = {
        "train": sorted(int(value) for value in shuffled[:train_count]),
        "val": sorted(int(value) for value in shuffled[train_count : train_count + val_count]),
        "calib": sorted(int(value) for value in shuffled[train_count + val_count :]),
    }
    if set(split["train"]) & set(split["val"]):
        raise AssertionError("train/val overlap")
    if set(split["train"]) & set(split["calib"]):
        raise AssertionError("train/calib overlap")
    if set(split["val"]) & set(split["calib"]):
        raise AssertionError("val/calib overlap")
    return split


def split_frame_by_unit(
    frame: pd.DataFrame, unit_split: dict[str, list[int]]
) -> dict[str, pd.DataFrame]:
    """Apply a precomputed engine split before any windows are generated."""

    return {
        name: frame.loc[frame["unit_id"].isin(units)].copy()
        for name, units in unit_split.items()
    }


def fit_train_minmax(
    train_frame: pd.DataFrame, feature_columns: tuple[str, ...]
) -> MinMaxScaler:
    """Fit preprocessing statistics exclusively on registered training units."""

    if train_frame.empty:
        raise ValueError("cannot fit scaler on an empty training frame")
    scaler = MinMaxScaler()
    scaler.fit(train_frame.loc[:, feature_columns])
    return scaler


def transform_features(
    frame: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.copy()
    result.loc[:, feature_columns] = scaler.transform(result.loc[:, feature_columns])
    return result


def make_windows(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    window_size: int = 30,
    stride: int = 1,
    endpoint_only: bool = False,
    pad_short_endpoint: bool = True,
) -> WindowedData:
    """Build windows within engines only; no row can cross an engine boundary."""

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    required = {"unit_id", "cycle", "true_rul", *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"window input missing columns: {sorted(missing)}")

    features: list[np.ndarray] = []
    labels: list[float] = []
    unit_ids: list[int] = []
    cycles: list[int] = []
    for unit_id, group in frame.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        values = group.loc[:, feature_columns].to_numpy(dtype=np.float32)
        group_labels = group["true_rul"].to_numpy(dtype=np.float32)
        group_cycles = group["cycle"].to_numpy(dtype=np.int64)
        if endpoint_only:
            endpoints = [len(group) - 1]
        else:
            endpoints = list(range(window_size - 1, len(group), stride))
        if endpoint_only and len(group) < window_size and pad_short_endpoint:
            pad_count = window_size - len(group)
            values = np.vstack([np.repeat(values[:1], pad_count, axis=0), values])
            endpoint = len(values) - 1
            window = values[endpoint - window_size + 1 : endpoint + 1]
            label_index = len(group) - 1
            features.append(window)
            labels.append(float(group_labels[label_index]))
            unit_ids.append(int(unit_id))
            cycles.append(int(group_cycles[label_index]))
            continue
        for endpoint in endpoints:
            if endpoint < window_size - 1:
                continue
            features.append(values[endpoint - window_size + 1 : endpoint + 1])
            labels.append(float(group_labels[endpoint]))
            unit_ids.append(int(unit_id))
            cycles.append(int(group_cycles[endpoint]))

    if not features:
        raise ValueError("window construction produced no samples")
    return WindowedData(
        features=np.stack(features).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        cycles=np.asarray(cycles, dtype=np.int64),
        sensor_columns=tuple(feature_columns),
    )


def verify_dataset(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Validate the official text-file shapes and unit/RUL joins."""

    report: dict[str, dict[str, Any]] = {}
    for subset, expected in EXPECTED_DATASET_SHAPES.items():
        loaded = load_subset(data_dir, subset)
        observed = {
            "train_rows": len(loaded.train),
            "test_rows": len(loaded.test),
            "train_units": int(loaded.train["unit_id"].nunique()),
            "test_units": int(loaded.test["unit_id"].nunique()),
        }
        if observed != expected:
            raise ValueError(f"{subset} shape mismatch: {observed} != {expected}")
        observed["train_sha256"] = sha256_file(data_dir / f"train_{subset}.txt")
        observed["test_sha256"] = sha256_file(data_dir / f"test_{subset}.txt")
        observed["rul_sha256"] = sha256_file(data_dir / f"RUL_{subset}.txt")
        report[subset] = observed
    return report


def split_payload(data_dir: Path, subset: str, seed: int = 42) -> dict[str, Any]:
    loaded = load_subset(data_dir, subset)
    units = loaded.train["unit_id"].drop_duplicates().astype(int).tolist()
    unit_split = create_unit_split(units, seed=seed)
    train_path = data_dir / f"train_{subset}.txt"
    return {
        "schema_version": 1,
        "dataset": "C-MAPSS",
        "subset": subset,
        "seed": seed,
        "source_train_file": f"data/interim/cmapss/train_{subset}.txt",
        "source_train_sha256": sha256_file(train_path),
        "source_unit_count": len(units),
        "split_algorithm": "numpy_default_rng_permutation_round_70_then_half_remainder",
        "unit_split": unit_split,
        "counts": {name: len(values) for name, values in unit_split.items()},
        "fractions": {"train": 0.70, "val": 0.15, "calib": 0.15},
        "allocation_unit": "engine_unit",
        "calib_isolation": "never_used_for_training_or_tuning",
        "status": "ready",
    }


def write_split_files(data_dir: Path, output_dir: Path, seed: int = 42) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for subset in EXPECTED_DATASET_SHAPES:
        path = output_dir / f"{subset}_seed{seed}.json"
        path.write_text(
            json.dumps(split_payload(data_dir, subset, seed), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write-splits", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if args.verify:
        result["dataset"] = verify_dataset(args.data_dir)
    if args.write_splits is not None:
        result["splits"] = [
            str(path) for path in write_split_files(args.data_dir, args.write_splits, args.seed)
        ]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
