
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from scripts.uq_reference_arm import expand_cells, load_config, validate_completed_cell

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_unified_aggregate_assets_close() -> None:
    summary = json.loads((ROOT / "results" / "UNIFIED_GRID_SUMMARY.json").read_text())
    metrics = pd.read_parquet(ROOT / "results" / "UNIFIED_GRID_RUN_METRICS.parquet")
    means = pd.read_csv(ROOT / "results" / "UNIFIED_GRID_MEAN_METRICS.csv")
    paired = pd.read_csv(ROOT / "results" / "UNIFIED_GRID_POINT_QUANTILE_PAIRS.csv")

    assert summary["status"] == "completed_validated"
    assert summary["registered_cells"] == summary["completed_cells"] == 160
    assert summary["validated_prediction_tables"] == 480
    assert len(metrics) == 160 and metrics["run_id"].nunique() == 160
    assert len(means) == 32 and set(means["seed_count"]) == {5}
    assert len(paired) == 80


def test_uq_literature_coding_closes_the_declared_denominator() -> None:
    rows = _read_csv(ROOT / "papercorpus" / "PHASE3_UQ_CODEBOOK.csv")
    counts = _read_csv(ROOT / "papercorpus" / "PHASE3_UQ_PRACTICE_COUNTS.csv")
    totals: dict[str, int] = defaultdict(int)
    lookup: dict[tuple[str, str], int] = {}
    for row in counts:
        totals[row["dimension"]] += int(row["n"])
        lookup[(row["dimension"], row["category"])] = int(row["n"])

    assert len(rows) == 19 and len({row["paper_id"] for row in rows}) == 19
    assert set(totals.values()) == {19}
    assert lookup[("UQ reported output", "no")] == 15
    assert lookup[("UQ validation", "yes")] == 1


def test_uq_aggregate_assets_close() -> None:
    result_root = ROOT / "results" / "uq_reference"
    summary = json.loads((result_root / "UQ_REFERENCE_SUMMARY.json").read_text())
    cells = pd.read_csv(result_root / "UQ_REFERENCE_CELL_SUMMARY.csv")
    protocol = pd.read_csv(result_root / "UQ_PROTOCOL_CONTRASTS.csv")
    methods = pd.read_csv(result_root / "UQ_METHOD_COMPARISONS.csv")
    calibration = pd.read_csv(result_root / "UQ_CQR_CALIBRATION_EFFECTS.csv")

    assert summary["status"] == "completed_validated"
    assert summary["registered_cells"] == summary["completed_cells"] == 280
    assert len(cells) == 280
    assert len(protocol) == 24
    assert len(methods) == 16
    assert len(calibration) == 16


def test_common_truth_assets_close() -> None:
    result_root = ROOT / "results" / "common_truth"
    summary = json.loads((result_root / "COMMON_TRUTH_SUMMARY.json").read_text())
    contrasts = pd.read_csv(result_root / "COMMON_TRUTH_LABEL_CONTRASTS.csv")
    reversals = pd.read_csv(result_root / "COMMON_TRUTH_RANKING_REVERSALS.csv")
    metrics = pd.read_csv(result_root / "COMMON_TRUTH_RUN_METRICS.csv")

    assert summary["status"] == "completed_validated"
    assert summary["registered_input_runs"] == 120
    assert summary["run_truth_metric_rows"] == 240
    assert summary["paired_label_contrasts"] == 24
    assert summary["ranking_reversal_count"] == 3
    assert summary["changes_kill_v1_decision"] is False
    assert len(contrasts) == 24
    assert len(reversals) == 3
    assert len(metrics) == 240


def test_registered_figures_are_present() -> None:
    for name in (
        "FIG_KILL_TEST_EFFECTS.png",
        "FIG_UNIFIED_RMSE_STABILITY.png",
        "FIG_RANK_AGREEMENT.png",
        "FIG_UQ_REFERENCE_ARM.png",
    ):
        path = ROOT / "paper" / "figures" / name
        assert path.is_file() and path.stat().st_size > 10_000


def test_companion_dataset_closes_all_per_cell_artifacts_when_restored() -> None:
    run_root = ROOT / "results" / "runs"
    uq_root = ROOT / "results" / "uq_reference"
    if not run_root.is_dir() or not any(uq_root.glob("uq_ref_v1__*/meta.json")):
        pytest.skip("restore all Zenodo result-data parts to run the closure test")

    kill = list(run_root.glob("kill_v1__*/meta.json"))
    unified = list(run_root.glob("unified_v1__*/meta.json"))
    uq_meta = list(uq_root.glob("uq_ref_v1__*/meta.json"))
    assert len(kill) == 120
    assert len(unified) == 160
    assert len(uq_meta) == 280

    for meta_path in kill + unified:
        directory = meta_path.parent
        for name in ("preds_val.parquet", "preds_calib.parquet", "preds_test.parquet"):
            assert (directory / name).is_file()

    config = load_config(ROOT / "configs" / "uq_reference_arm.yaml")
    cells = expand_cells(ROOT, config)
    assert len(cells) == 280
    for cell in cells:
        validated = validate_completed_cell(ROOT, cell)
        assert validated["test_windows"] > 0
