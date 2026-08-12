from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from rul_audit.literature_analysis import (
    MODEL_PAPER_IDS,
    aggregate_practices,
    build_evidence_matrix,
    load_and_validate_codebook,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT_ROOT / "papercorpus"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_lookup(rows: list[dict[str, str | int]]) -> dict[tuple[str, str], int]:
    return {(str(row["dimension"]), str(row["category"])): int(row["n"]) for row in rows}


def test_phase3_codebook_is_complete_and_uses_controlled_categories() -> None:
    rows = load_and_validate_codebook(CORPUS / "PHASE3_ANALYSIS_CODEBOOK.csv")

    assert tuple(row["paper_id"] for row in rows) == MODEL_PAPER_IDS


def test_phase3_practice_counts_match_the_frozen_coding() -> None:
    rows = load_and_validate_codebook(CORPUS / "PHASE3_ANALYSIS_CODEBOOK.csv")
    counts = aggregate_practices(rows)
    lookup = _count_lookup(counts)

    assert lookup[("RUL cap", "cap_125")] == 12
    assert lookup[("RUL cap", "other_explicit")] == 4
    assert lookup[("RUL cap", "unreported")] == 3
    assert lookup[("Normalization fit scope", "train_or_source_only")] == 6
    assert lookup[("Test prediction denominator", "endpoint_per_engine")] == 9
    assert lookup[("Test prediction denominator", "all_window_or_chunk")] == 2
    assert lookup[("Validation and test boundary", "test_monitored_or_tuned")] == 3
    assert lookup[("Validation and test boundary", "separate_non_test")] == 5
    assert lookup[("Validation and test boundary", "transductive_no_independent_test")] == 2
    assert lookup[("Seed reporting", "exact_values")] == 3
    assert lookup[("Seed reporting", "count_only")] == 1
    assert lookup[("Independent repetition count", "yes")] == 9
    assert lookup[("Run-level dispersion", "usable")] == 5
    assert lookup[("Run-level dispersion", "partial_unclear")] == 1
    assert lookup[("RMSE and NASA Score pair", "yes")] == 16
    assert lookup[("Code path", "some_code_path")] == 3
    assert lookup[("C-MAPSS subset coverage", "yes")] == 9
    assert lookup[("N-CMAPSS inclusion", "yes")] == 4

    totals: dict[str, int] = defaultdict(int)
    for row in counts:
        totals[str(row["dimension"])] += int(row["n"])
    assert set(totals.values()) == {len(MODEL_PAPER_IDS)}


def test_phase3_evidence_matrix_joins_only_model_papers() -> None:
    codebook = load_and_validate_codebook(CORPUS / "PHASE3_ANALYSIS_CODEBOOK.csv")
    protocol = _read_csv(CORPUS / "protocol_coding.csv")

    matrix = build_evidence_matrix(codebook, protocol)

    assert len(matrix) == 19
    assert {row["paper_id"] for row in matrix} == set(MODEL_PAPER_IDS)
    assert all(row["title"] and row["protocol_notes"] for row in matrix)


def test_generated_phase3_csvs_match_the_validated_analysis() -> None:
    codebook = load_and_validate_codebook(CORPUS / "PHASE3_ANALYSIS_CODEBOOK.csv")
    expected_counts = aggregate_practices(codebook)
    expected_matrix = build_evidence_matrix(codebook, _read_csv(CORPUS / "protocol_coding.csv"))

    actual_counts = _read_csv(CORPUS / "PHASE3_PROTOCOL_PRACTICE_COUNTS.csv")
    actual_matrix = _read_csv(CORPUS / "PHASE3_EVIDENCE_MATRIX.csv")

    normalized_counts = [{key: str(value) for key, value in row.items()} for row in expected_counts]
    assert actual_counts == normalized_counts
    assert actual_matrix == expected_matrix
