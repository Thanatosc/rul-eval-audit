from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MODEL_PAPER_IDS = tuple(f"C{index:02d}" for index in range(1, 19)) + ("C21",)

CATEGORY_VOCABULARY = {
    "cap_category": {"cap_125", "other_explicit", "unreported"},
    "normalization_fit_scope": {"train_or_source_only", "not_confirmed"},
    "test_window_policy": {
        "endpoint_per_engine",
        "all_window_or_chunk",
        "unclear_or_nonindependent",
    },
    "validation_boundary": {
        "separate_non_test",
        "test_monitored_or_tuned",
        "transductive_no_independent_test",
        "unclear",
    },
    "seed_reporting": {"exact_values", "count_only", "not_reported"},
    "repetition_count_reported": {"yes", "no"},
    "run_level_dispersion": {"usable", "partial_unclear", "absent"},
    "rmse_nasa_pair": {"yes", "no"},
    "code_path": {"some_code_path", "none_or_nonreproducible"},
    "cmapss_all_four": {"yes", "no"},
    "n_cmapss_included": {"yes", "no"},
    "source_quality": {"B", "C"},
    "verification_basis": {
        "local_pdf_preflight_pass",
        "remote_exact_fulltext_no_local_preflight",
    },
}


@dataclass(frozen=True)
class CountSpec:
    dimension: str
    category: str
    field: str
    value: str
    interpretive_limit: str


def _specs_for_field(
    dimension: str,
    field: str,
    values: Iterable[str],
    limit: str,
) -> list[CountSpec]:
    return [CountSpec(dimension, value, field, value, limit) for value in values]


COUNT_SPECS = (
    _specs_for_field(
        "RUL cap",
        "cap_category",
        ("cap_125", "other_explicit", "unreported"),
        "A documented cap describes label construction, not its causal effect on scores.",
    )
    + _specs_for_field(
        "Normalization fit scope",
        "normalization_fit_scope",
        ("train_or_source_only", "not_confirmed"),
        "Not confirmed means not reproducible from the report; it does not prove leakage occurred.",
    )
    + _specs_for_field(
        "Test prediction denominator",
        "test_window_policy",
        ("endpoint_per_engine", "all_window_or_chunk", "unclear_or_nonindependent"),
        "Endpoint and all-window metrics use different observational denominators.",
    )
    + _specs_for_field(
        "Validation and test boundary",
        "validation_boundary",
        (
            "separate_non_test",
            "test_monitored_or_tuned",
            "transductive_no_independent_test",
            "unclear",
        ),
        "The category records disclosed practice and cannot recover undocumented author behavior.",
    )
    + _specs_for_field(
        "Seed reporting",
        "seed_reporting",
        ("exact_values", "count_only", "not_reported"),
        "A seed declaration improves traceability but does not by itself establish robustness.",
    )
    + _specs_for_field(
        "Independent repetition count",
        "repetition_count_reported",
        ("yes", "no"),
        "Stochastic forward passes and hyperparameter candidates are not counted as repetitions.",
    )
    + _specs_for_field(
        "Run-level dispersion",
        "run_level_dispersion",
        ("usable", "partial_unclear", "absent"),
        "Predictive intervals are distinct from variation across independent training runs.",
    )
    + _specs_for_field(
        "RMSE and NASA Score pair",
        "rmse_nasa_pair",
        ("yes", "no"),
        "Dual metric reporting permits comparison but does not establish metric disagreement.",
    )
    + _specs_for_field(
        "Code path",
        "code_path",
        ("some_code_path", "none_or_nonreproducible"),
        "A stated or located code path is not equivalent to exact end-to-end reproduction.",
    )
    + _specs_for_field(
        "C-MAPSS subset coverage",
        "cmapss_all_four",
        ("yes", "no"),
        "Four-subset coverage broadens benchmark scope but not real-fleet external validity.",
    )
    + _specs_for_field(
        "N-CMAPSS inclusion",
        "n_cmapss_included",
        ("yes", "no"),
        "N-CMAPSS results use distinct data and are not pooled with C-MAPSS scores.",
    )
    + _specs_for_field(
        "Source quality",
        "source_quality",
        ("B", "C"),
        "Quality grades measure fitness for protocol-audit claims, not model merit.",
    )
)


class LiteratureAnalysisError(ValueError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_and_validate_codebook(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    ids = [row.get("paper_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise LiteratureAnalysisError("Phase 3 codebook contains duplicate paper IDs")
    if set(ids) != set(MODEL_PAPER_IDS):
        missing = sorted(set(MODEL_PAPER_IDS) - set(ids))
        extra = sorted(set(ids) - set(MODEL_PAPER_IDS))
        raise LiteratureAnalysisError(f"Phase 3 model-paper IDs mismatch: missing={missing}, extra={extra}")

    for field, allowed in CATEGORY_VOCABULARY.items():
        for row in rows:
            value = row.get(field, "")
            if value not in allowed:
                raise LiteratureAnalysisError(
                    f"{row['paper_id']} has invalid {field}={value!r}; expected one of {sorted(allowed)}"
                )
    return sorted(rows, key=lambda row: row["paper_id"])


def aggregate_practices(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    denominator = len(rows)
    if denominator != len(MODEL_PAPER_IDS):
        raise LiteratureAnalysisError(
            f"Expected {len(MODEL_PAPER_IDS)} model papers, received {denominator}"
        )

    output: list[dict[str, str | int]] = []
    for spec in COUNT_SPECS:
        paper_ids = sorted(row["paper_id"] for row in rows if row[spec.field] == spec.value)
        output.append(
            {
                "dimension": spec.dimension,
                "category": spec.category,
                "n": len(paper_ids),
                "denominator": denominator,
                "percent": f"{100 * len(paper_ids) / denominator:.1f}",
                "paper_ids": ";".join(paper_ids),
                "interpretive_limit": spec.interpretive_limit,
            }
        )

    counts_by_dimension = Counter(row["dimension"] for row in output)
    for dimension in counts_by_dimension:
        dimension_total = sum(int(row["n"]) for row in output if row["dimension"] == dimension)
        if dimension_total != denominator:
            raise LiteratureAnalysisError(
                f"Categories for {dimension!r} do not partition the {denominator}-paper corpus"
            )
    return output


def build_evidence_matrix(
    codebook_rows: list[dict[str, str]],
    protocol_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    protocol_by_id = {row["paper_id"]: row for row in protocol_rows}
    missing = sorted(set(MODEL_PAPER_IDS) - set(protocol_by_id))
    if missing:
        raise LiteratureAnalysisError(f"Protocol coding is missing model papers: {missing}")

    matrix: list[dict[str, str]] = []
    for coded in codebook_rows:
        source = protocol_by_id[coded["paper_id"]]
        matrix.append(
            {
                "paper_id": coded["paper_id"],
                "title": source["title"],
                "year": source["year"],
                "venue": source["venue"],
                "dataset_subsets": source["dataset_subsets"],
                "reported_rul_cap": source["rul_cap"],
                **{field: coded[field] for field in CATEGORY_VOCABULARY},
                "protocol_notes": source["notes"],
            }
        )
    return matrix


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise LiteratureAnalysisError(f"Refusing to write empty analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_analysis_outputs(project_root: Path) -> dict[str, int]:
    corpus = project_root / "papercorpus"
    codebook_rows = load_and_validate_codebook(corpus / "PHASE3_ANALYSIS_CODEBOOK.csv")
    protocol_rows = _read_csv(corpus / "protocol_coding.csv")
    counts = aggregate_practices(codebook_rows)
    matrix = build_evidence_matrix(codebook_rows, protocol_rows)
    _write_csv(corpus / "PHASE3_PROTOCOL_PRACTICE_COUNTS.csv", counts)
    _write_csv(corpus / "PHASE3_EVIDENCE_MATRIX.csv", matrix)
    return {
        "model_papers": len(codebook_rows),
        "count_rows": len(counts),
        "matrix_rows": len(matrix),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = generate_analysis_outputs(root)
    print(
        "Phase 3 literature analysis generated "
        f"{result['count_rows']} counts and {result['matrix_rows']} evidence rows."
    )
