from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rul_audit.data.cmapss import (
    CMAPSS_COLUMNS,
    SENSOR_COLUMNS,
    apply_rul_label,
    create_unit_split,
    fit_train_minmax,
    load_subset,
    make_windows,
    split_frame_by_unit,
    transform_features,
)


def _row(unit_id: int, cycle: int, sensor_base: float) -> list[float]:
    return [unit_id, cycle, 0.0, 0.0, 0.0, *[sensor_base + index for index in range(21)]]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_loader_derives_train_and_test_rul(tmp_path: Path) -> None:
    train_rows = [_row(1, cycle, 1.0) for cycle in range(1, 4)] + [
        _row(2, cycle, 2.0) for cycle in range(1, 3)
    ]
    test_rows = [_row(1, cycle, 3.0) for cycle in range(1, 3)] + [
        _row(2, cycle, 4.0) for cycle in range(1, 4)
    ]
    _write_rows(tmp_path / "train_FD001.txt", train_rows)
    _write_rows(tmp_path / "test_FD001.txt", test_rows)
    (tmp_path / "RUL_FD001.txt").write_text("5\n7\n", encoding="utf-8")

    loaded = load_subset(tmp_path, "FD001")

    assert tuple(loaded.train.columns[:26]) == CMAPSS_COLUMNS
    assert loaded.train.loc[loaded.train["unit_id"] == 1, "raw_rul"].tolist() == [2.0, 1.0, 0.0]
    assert loaded.test.loc[loaded.test["unit_id"] == 1, "raw_rul"].tolist() == [6.0, 5.0]
    assert loaded.test.loc[loaded.test["unit_id"] == 2, "raw_rul"].tolist() == [9.0, 8.0, 7.0]


def test_unit_split_is_deterministic_disjoint_and_exhaustive() -> None:
    first = create_unit_split(list(range(1, 101)), seed=42)
    second = create_unit_split(list(range(1, 101)), seed=42)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 70,
        "val": 15,
        "calib": 15,
    }
    assert set().union(*(set(values) for values in first.values())) == set(range(1, 101))


def test_scaler_is_fit_on_training_units_only() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3],
            "cycle": [1, 2, 1, 2, 1, 2],
            "sensor_1": [0.0, 1.0, 100.0, 101.0, 200.0, 201.0],
            "true_rul": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    partitions = split_frame_by_unit(
        frame, {"train": [1], "val": [2], "calib": [3]}
    )
    scaler = fit_train_minmax(partitions["train"], ("sensor_1",))
    transformed_val = transform_features(partitions["val"], scaler, ("sensor_1",))

    assert scaler.data_min_.tolist() == [0.0]
    assert scaler.data_max_.tolist() == [1.0]
    assert transformed_val["sensor_1"].tolist() == [100.0, 101.0]


def test_windows_never_cross_engine_boundaries() -> None:
    rows = []
    for unit_id in (1, 2):
        for cycle in range(1, 5):
            row = {"unit_id": unit_id, "cycle": cycle, "true_rul": float(4 - cycle)}
            row.update({column: float(unit_id) for column in SENSOR_COLUMNS})
            rows.append(row)
    windowed = make_windows(
        pd.DataFrame(rows), SENSOR_COLUMNS, window_size=3, stride=1
    )

    assert windowed.features.shape == (4, 3, 21)
    assert windowed.unit_ids.tolist() == [1, 1, 2, 2]
    for values, unit_id in zip(windowed.features, windowed.unit_ids, strict=True):
        assert bool(np.all(values == float(unit_id)))


def test_label_modes_preserve_raw_rul() -> None:
    frame = pd.DataFrame({"raw_rul": [140.0, 120.0]})

    capped = apply_rul_label(frame, "piecewise_125")
    linear = apply_rul_label(frame, "linear_uncapped")

    assert capped["true_rul"].tolist() == [125.0, 120.0]
    assert linear["true_rul"].tolist() == [140.0, 120.0]
    assert frame.columns.tolist() == ["raw_rul"]
