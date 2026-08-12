from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from rul_audit.data.cmapss import sha256_file
from rul_audit.experiments.unified_grid import (
    UnifiedGridCell,
    expand_cells,
    load_config,
    run_registered_cell,
)
from rul_audit.protocols.readiness import source_tree_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unified_grid_expands_to_160_unique_registered_cells() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "unified_grid.yaml")
    cells = expand_cells(config)

    assert len(cells) == 160
    assert len({cell.run_id for cell in cells}) == 160
    assert {cell.model for cell in cells} == {
        "lstm",
        "cnn_1d",
        "transformer",
        "lightgbm",
    }
    assert {cell.output_mode for cell in cells} == {"point", "quantile"}


def test_unified_runner_refuses_without_authorization(tmp_path: Path) -> None:
    cell = UnifiedGridCell(
        run_id="unified_v1__fd001__lstm__seed11__point",
        dataset="C-MAPSS",
        subset="FD001",
        model="lstm",
        seed=11,
        output_mode="point",
        protocol_version="unified_v1",
    )

    with pytest.raises(PermissionError, match="execution is gated"):
        run_registered_cell(tmp_path, cell)


def test_unified_runner_refuses_source_tree_drift_before_data_read(tmp_path: Path) -> None:
    source_dir = tmp_path / "src" / "rul_audit"
    source_dir.mkdir(parents=True)
    (source_dir / "placeholder.py").write_text("VALUE = 1\n", encoding="utf-8")
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "unified_v1.md"
    protocol_path.write_text("frozen\n", encoding="utf-8")
    config = load_config(PROJECT_ROOT / "configs" / "unified_grid.yaml")
    config["status"] = "frozen_authorized_for_execution"
    config["execution_gate"] = {
        "ready": True,
        "real_cells_authorized": True,
        "automatic_retry_authorized": False,
        "execution_order": "status_register_row_order",
        "failure_policy": "stop_on_first_failed_cell_no_retry",
    }
    config["registration"] = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": "0" * 64,
    }
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "unified_grid.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_status = PROJECT_ROOT / "results" / "UNIFIED_GRID_STATUS.csv"
    (results_dir / "UNIFIED_GRID_STATUS.csv").write_text(
        source_status.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="source-tree hash differs"):
        run_registered_cell(tmp_path, expand_cells(config)[0], authorize_real_execution=True)


def test_unified_runner_refuses_nonpending_cell_before_data_read(tmp_path: Path) -> None:
    source_dir = tmp_path / "src" / "rul_audit"
    source_dir.mkdir(parents=True)
    (source_dir / "placeholder.py").write_text("VALUE = 1\n", encoding="utf-8")
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "unified_v1.md"
    protocol_path.write_text("frozen\n", encoding="utf-8")
    config = load_config(PROJECT_ROOT / "configs" / "unified_grid.yaml")
    config["status"] = "frozen_authorized_for_execution"
    config["execution_gate"] = {
        "ready": True,
        "real_cells_authorized": True,
        "automatic_retry_authorized": False,
        "execution_order": "status_register_row_order",
        "failure_policy": "stop_on_first_failed_cell_no_retry",
    }
    config["registration"] = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": source_tree_sha256(tmp_path),
    }
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "unified_grid.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_status = PROJECT_ROOT / "results" / "UNIFIED_GRID_STATUS.csv"
    status_path = results_dir / "UNIFIED_GRID_STATUS.csv"
    status_path.write_text(source_status.read_text(encoding="utf-8"), encoding="utf-8")
    with status_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "completed"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="not pending"):
        run_registered_cell(tmp_path, expand_cells(config)[0], authorize_real_execution=True)
