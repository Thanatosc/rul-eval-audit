from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "postgrid"


def test_postgrid_summary_closes_frozen_inputs_and_planned_outputs() -> None:
    summary = json.loads((RESULTS / "POSTGRID_SUMMARY.json").read_text(encoding="utf-8"))

    assert summary["status"] == "completed_verified"
    assert summary["cell_closure"] == {"kill_test": 120, "unified_grid": 160}
    assert summary["kill_test"]["registered_contrasts"] == 24
    assert summary["unified_grid"]["kendall_panels"] == 8
    assert summary["metric_agreement"]["contexts"] == 80
    assert summary["point_quantile"]["paired_cells"] == 80
    assert summary["point_quantile"]["inference_permitted"] is False
    assert summary["fallacy_scan"] == {"checked": 11, "total": 11}
    assert len(summary["output_hashes"]) == 13


def test_postgrid_tables_have_closed_shapes_and_valid_adjustments() -> None:
    contrasts = pd.read_csv(RESULTS / "KILL_TEST_CONTRAST_INFERENCE.csv")
    flip_matrix = pd.read_csv(RESULTS / "KILL_TEST_RANKING_FLIP_MATRIX.csv")
    stability = pd.read_csv(RESULTS / "UNIFIED_MODEL_STABILITY.csv")
    kendall = pd.read_csv(RESULTS / "UNIFIED_KENDALL_W.csv")
    pairwise = pd.read_csv(RESULTS / "UNIFIED_PAIRWISE_STABILITY.csv")
    mode_flips = pd.read_csv(RESULTS / "UNIFIED_POINT_QUANTILE_FLIPS.csv")
    agreement = pd.read_csv(RESULTS / "METRIC_RANK_CONSISTENCY.csv")
    point_quantile = pd.read_csv(RESULTS / "POINT_QUANTILE_DESCRIPTIVE.csv")

    assert len(contrasts) == 24
    assert set(contrasts["seed_count"]) == {5}
    assert (contrasts["holm_adjusted_p"] >= contrasts["exact_sign_flip_p"]).all()
    assert len(flip_matrix) == 6
    assert flip_matrix["ranking_reversal_count"].sum() == 2
    assert len(stability) == 32
    assert len(kendall) == 8
    assert kendall["kendalls_w"].between(0, 1).all()
    assert (kendall["holm_adjusted_p"] >= kendall["exact_permutation_p"]).all()
    assert len(pairwise) == 48
    assert set(pairwise["evaluated_seed_pairs"]) == {10}
    assert len(mode_flips) == 24
    assert len(agreement) == 80
    assert len(point_quantile) == 16
    assert point_quantile["quantile_lower_rmse_seed_count"].sum() == 13


def test_postgrid_report_and_figures_preserve_interpretation_boundaries() -> None:
    report = (RESULTS / "POSTGRID_STATISTICAL_VALIDATION.md").read_text(
        encoding="utf-8"
    )

    assert "11/11" in report
    assert "Overall Confidence:** CAUTION" in report
    assert "No superiority test" in report
    assert "five computational seeds" in report
    for filename in (
        "FIG_KILL_TEST_EFFECTS.png",
        "FIG_UNIFIED_RMSE_STABILITY.png",
        "FIG_RANK_AGREEMENT.png",
    ):
        path = PROJECT_ROOT / "paper" / "figures" / filename
        assert path.is_file()
        assert path.stat().st_size > 10_000
