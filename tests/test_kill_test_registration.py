from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from rul_audit.data.cmapss import sha256_file
from rul_audit.experiments.kill_test import (
    KillTestCell,
    decide_kill_test,
    expand_cells,
    load_config,
    run_registered_cell,
    write_status_register,
)
from rul_audit.protocols.readiness import source_tree_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_kill_test_expands_to_120_unique_cells(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "kill_test.yaml"
    cells = expand_cells(load_config(config_path))
    status_path = tmp_path / "KILL_TEST_STATUS.csv"

    assert len(cells) == 120
    assert len({cell.run_id for cell in cells}) == 120
    assert write_status_register(config_path, status_path) == 120
    with status_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120
    assert {row["status"] for row in rows} == {"pending"}


def test_three_state_decision_rule_is_fail_closed() -> None:
    assert (
        decide_kill_test(
            max_absolute_rmse_effect=2.0,
            ranking_reversals=0,
            completed_cells=120,
        )
        == "PASS"
    )
    assert (
        decide_kill_test(
            max_absolute_rmse_effect=0.1,
            ranking_reversals=0,
            completed_cells=120,
        )
        == "FAIL"
    )
    assert (
        decide_kill_test(
            max_absolute_rmse_effect=0.7,
            ranking_reversals=0,
            completed_cells=120,
        )
        == "INCONCLUSIVE"
    )


def test_real_runner_refuses_before_reading_data_without_authorization(tmp_path: Path) -> None:
    cell = KillTestCell(
        run_id="kill_v1__fd001__lstm__piecewise_125__all_21__seed11",
        dataset="FD001",
        model="lstm",
        rul_label="piecewise_125",
        sensor_set="all_21",
        seed=11,
    )

    with pytest.raises(PermissionError, match="execution is gated"):
        run_registered_cell(tmp_path, cell)


def test_real_runner_refuses_source_tree_drift_before_reading_data(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "kill_test.yaml")
    config["registration"]["implementation_sha256"] = "0" * 64
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "kill_test.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    source_dir = tmp_path / "src" / "rul_audit"
    source_dir.mkdir(parents=True)
    (source_dir / "placeholder.py").write_text("VALUE = 1\n", encoding="utf-8")
    cell = expand_cells(config)[0]

    with pytest.raises(RuntimeError, match="source-tree hash differs"):
        run_registered_cell(tmp_path, cell, authorize_real_execution=True)


def test_real_runner_refuses_nonpending_cell_before_reading_data(tmp_path: Path) -> None:
    source_dir = tmp_path / "src" / "rul_audit"
    source_dir.mkdir(parents=True)
    (source_dir / "placeholder.py").write_text("VALUE = 1\n", encoding="utf-8")
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "unified_v1.md"
    protocol_path.write_text(
        "Status: `frozen_pending_execution_authorization`\n", encoding="utf-8"
    )
    config = load_config(PROJECT_ROOT / "configs" / "kill_test.yaml")
    config["registration"]["implementation_sha256"] = source_tree_sha256(tmp_path)
    config["registration"]["protocol_sha256"] = sha256_file(protocol_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "kill_test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    status_path = tmp_path / "results" / "KILL_TEST_STATUS.csv"
    write_status_register(config_path, status_path)
    with status_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "completed"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="not pending"):
        run_registered_cell(
            tmp_path, expand_cells(config)[0], authorize_real_execution=True
        )
    assert (
        decide_kill_test(
            max_absolute_rmse_effect=2.0,
            ranking_reversals=2,
            completed_cells=119,
        )
        == "INCONCLUSIVE"
    )
