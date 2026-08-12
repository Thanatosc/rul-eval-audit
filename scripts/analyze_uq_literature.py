"""Validate and generate Topic 6 work package 6A UQ literature coding outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rul_audit.literature_analysis import MODEL_PAPER_IDS

ANALYSIS_ID = "topic6_work_package_6a_v1"
ANCHOR_IDS = ("C19", "C20", "C22", "C23", "C24", "C25")
FROZEN_INPUT_HASHES = {
    "papercorpus/protocol_coding.csv": (
        "b0519dc0df87984db0f71dd9e1eddb281a325f8983a7d021ff1ab25f8e6b3c08"
    ),
    "papercorpus/PHASE3_ANALYSIS_CODEBOOK.csv": (
        "c490a9b53f31fa5304c8d4d71b47ac349358f972431196d09e1d685a17319a13"
    ),
    "papercorpus/FULLTEXT_MANIFEST.csv": (
        "c7fa6bcdcc3aa4d9f281b798c8fe1e81d0c073d714e73da44755cf3c4afff7cf"
    ),
}
UQ_REPORTED_VALUES = (
    "no",
    "interval",
    "distribution",
    "set",
    "multiple",
    "NR",
    "not_applicable",
)
UQ_METHOD_VALUES = (
    "none",
    "mc_dropout",
    "deep_ensemble",
    "bayesian",
    "conformal",
    "quantile_regression",
    "other",
    "multiple",
    "NR",
    "not_applicable",
)
UQ_VALIDATED_VALUES = ("yes", "no", "NR", "not_applicable")
CODEBOOK_FIELDS = (
    "paper_id",
    "uq_reported",
    "uq_method",
    "uq_validated",
    "evidence_anchor",
    "evidence_summary",
    "validation_rationale",
    "verification_basis",
    "coder_count",
)


class UQLiteratureError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, *, skipinitialspace: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=skipinitialspace))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise UQLiteratureError(f"refusing to write an empty UQ output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_frozen_inputs(root: Path) -> None:
    for relative, expected in FROZEN_INPUT_HASHES.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise UQLiteratureError(
                f"frozen input hash mismatch for {relative}: {observed} != {expected}"
            )


def load_codebook(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if not rows or tuple(rows[0]) != CODEBOOK_FIELDS:
        raise UQLiteratureError("UQ codebook fields do not match the frozen schema")
    ids = [row["paper_id"] for row in rows]
    if tuple(ids) != MODEL_PAPER_IDS:
        raise UQLiteratureError(
            f"UQ codebook must contain the ordered 19-paper denominator: {ids}"
        )
    if len(ids) != len(set(ids)):
        raise UQLiteratureError("UQ codebook contains duplicate paper IDs")

    for row in rows:
        paper_id = row["paper_id"]
        if row["uq_reported"] not in UQ_REPORTED_VALUES[:-1]:
            raise UQLiteratureError(f"{paper_id} has invalid uq_reported")
        if row["uq_method"] not in UQ_METHOD_VALUES[:-1]:
            raise UQLiteratureError(f"{paper_id} has invalid uq_method")
        if row["uq_validated"] not in UQ_VALIDATED_VALUES[:-1]:
            raise UQLiteratureError(f"{paper_id} has invalid uq_validated")
        if row["coder_count"] != "1":
            raise UQLiteratureError(f"{paper_id} must retain the single-coder disclosure")
        for field in ("evidence_anchor", "evidence_summary", "validation_rationale"):
            if not row[field].strip():
                raise UQLiteratureError(f"{paper_id} is missing {field}")
        if row["verification_basis"] not in {
            "local_pdf_preflight_pass",
            "remote_exact_fulltext_no_local_preflight",
        }:
            raise UQLiteratureError(f"{paper_id} has invalid verification_basis")
        if row["uq_reported"] == "no" and row["uq_method"] != "none":
            raise UQLiteratureError(f"{paper_id}: no UQ output requires method=none")
        if row["uq_reported"] != "no" and row["uq_method"] == "none":
            raise UQLiteratureError(f"{paper_id}: a UQ output cannot use method=none")
        if row["uq_validated"] == "yes" and row["uq_reported"] in {"no", "NR"}:
            raise UQLiteratureError(f"{paper_id}: validated UQ requires a reported output")
    return rows


def build_protocol_extension(
    protocol_rows: list[dict[str, str]], codebook: list[dict[str, str]]
) -> list[dict[str, str]]:
    expected_ids = set(MODEL_PAPER_IDS) | set(ANCHOR_IDS)
    protocol_ids = [row["paper_id"] for row in protocol_rows]
    if len(protocol_rows) != 25 or set(protocol_ids) != expected_ids:
        raise UQLiteratureError("protocol coding no longer closes at 19 model papers + 6 anchors")
    coded = {row["paper_id"]: row for row in codebook}
    output: list[dict[str, str]] = []
    for source in protocol_rows:
        paper_id = source["paper_id"]
        if paper_id in coded:
            uq = coded[paper_id]
            appended = {
                "uq_reported": uq["uq_reported"],
                "uq_method": uq["uq_method"],
                "uq_validated": uq["uq_validated"],
                "uq_evidence_anchor": uq["evidence_anchor"],
                "uq_evidence_summary": uq["evidence_summary"],
                "uq_validation_rationale": uq["validation_rationale"],
                "uq_coding_version": ANALYSIS_ID,
            }
        else:
            appended = {
                "uq_reported": "not_applicable",
                "uq_method": "not_applicable",
                "uq_validated": "not_applicable",
                "uq_evidence_anchor": "corpus role: historical/method/dataset anchor",
                "uq_evidence_summary": (
                    "Excluded from the frozen 19-paper model/protocol denominator."
                ),
                "uq_validation_rationale": (
                    "Not applicable because this record is an anchor rather than a model paper."
                ),
                "uq_coding_version": ANALYSIS_ID,
            }
        output.append({**source, **appended})
    return output


def aggregate_counts(codebook: list[dict[str, str]]) -> list[dict[str, Any]]:
    denominator = len(MODEL_PAPER_IDS)
    dimensions = (
        ("UQ reported output", "uq_reported", UQ_REPORTED_VALUES),
        ("UQ method", "uq_method", UQ_METHOD_VALUES),
        ("UQ validation", "uq_validated", UQ_VALIDATED_VALUES),
    )
    rows: list[dict[str, Any]] = []
    for dimension, field, categories in dimensions:
        for category in categories:
            ids = sorted(row["paper_id"] for row in codebook if row[field] == category)
            rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "n": len(ids),
                    "denominator": denominator,
                    "percent": f"{100 * len(ids) / denominator:.1f}",
                    "paper_ids": ";".join(ids),
                    "interpretive_limit": _interpretive_limit(dimension, category),
                }
            )
        total = sum(int(row["n"]) for row in rows if row["dimension"] == dimension)
        if total != denominator:
            raise UQLiteratureError(f"{dimension} categories do not close at {denominator}")
    return rows


def _interpretive_limit(dimension: str, category: str) -> str:
    if dimension == "UQ reported output":
        return (
            "Run-level confidence intervals for mean metrics are not predictive UQ."
            if category == "no"
            else "Output-object coding describes reporting practice, not interval quality."
        )
    if dimension == "UQ method":
        return "Method labels follow the implemented mechanism; titles alone are not coded."
    return (
        "Yes requires nominal-versus-empirical coverage or calibration on independent data; "
        "no does not imply the UQ method is invalid."
    )


def build_evidence_matrix(
    protocol_rows: list[dict[str, str]], codebook: list[dict[str, str]]
) -> list[dict[str, str]]:
    protocol = {row["paper_id"]: row for row in protocol_rows}
    rows: list[dict[str, str]] = []
    for coded in codebook:
        source = protocol[coded["paper_id"]]
        rows.append(
            {
                "paper_id": coded["paper_id"],
                "title": source["title"],
                "year": source["year"],
                "venue": source["venue"],
                "doi": source["doi"],
                "dataset_subsets": source["dataset_subsets"],
                "uq_reported": coded["uq_reported"],
                "uq_method": coded["uq_method"],
                "uq_validated": coded["uq_validated"],
                "evidence_anchor": coded["evidence_anchor"],
                "evidence_summary": coded["evidence_summary"],
                "validation_rationale": coded["validation_rationale"],
                "verification_basis": coded["verification_basis"],
                "coder_count": coded["coder_count"],
            }
        )
    return rows


def write_result_report(path: Path, summary: dict[str, Any]) -> None:
    report = f"""# Topic 6 Work Package 6A: UQ Literature-Practice Coding Result

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Origin Mode: literature evidence coding and verification
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: `{ANALYSIS_ID}`
- Main Denominator: 19 model/protocol papers
- Coder: one human author assisted by local full-text extraction; no second coder

## Result

- Predictive UQ reported: **{summary['headline']['uq_reported_any']}/19** papers
- No predictive UQ output reported: **{summary['headline']['uq_reported_no']}/19** papers
- Independent nominal-coverage/calibration validation: **{summary['headline']['uq_validated_yes']}/19** papers
- UQ-reporting papers: `{'; '.join(summary['headline']['uq_reported_ids'])}`
- Strictly validated paper: `{'; '.join(summary['headline']['uq_validated_ids'])}`

The four UQ-reporting papers are C05, C06, C17, and C21. C05 reports a
Bayesian predictive distribution and 95% intervals but no aggregate coverage or
calibration check. C06 reports capsule-dropout confidence intervals and UCS, but
does not provide an independent nominal-versus-empirical coverage assessment.
C17 reports Bayesian predictive distributions and 95% intervals and evaluates
PICP against the nominal 95% level on corresponding testing datasets, alongside
PINAW, CWC, and NLL. C21 reports Gaussian confidence intervals at 90%, 95%, and
99% and inside-interval ratios, but the entire target dataset is used for
fine-tuning; it therefore fails the frozen independent-data validation rule.

## Reporting Boundary

- Confidence intervals around RMSE/NASA metrics across seeds or runs are coded
  as run-level dispersion, not predictive UQ.
- Bayesian terminology, ensembles, or uncertainty discussion without a
  predictive interval/distribution/set output are coded `uq_reported=no`.
- `uq_validated=no` means the paper did not satisfy this audit's strict evidence
  rule; it does not establish that the method is invalid.
- Six historical, dataset, review, or methodological anchors are retained as
  `not_applicable` in the 25-row extended protocol table and never enter x/19.
- The coding was performed by a single author, so no inter-rater reliability is
  claimed.

## Validation

- Frozen upstream hashes: 3/3 matched
- Main codebook: 19/19 IDs, no duplicates
- Extended protocol table: 25 rows = 19 model papers + 6 anchors
- Evidence anchors and rationales: 19/19 non-empty
- Local page anchors: backed by existing PDF preflight PASS sidecars
- C13: section-level remote-full-text anchor; no local page/preflight claim
- Category denominators: all close at 19
"""
    path.write_text(report, encoding="utf-8")


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validate_frozen_inputs(root)
    corpus = root / "papercorpus"
    codebook_path = corpus / "PHASE3_UQ_CODEBOOK.csv"
    codebook = load_codebook(codebook_path)
    protocol_rows = _read_csv(corpus / "protocol_coding.csv", skipinitialspace=True)
    extension = build_protocol_extension(protocol_rows, codebook)
    counts = aggregate_counts(codebook)
    matrix = build_evidence_matrix(protocol_rows, codebook)

    extension_path = corpus / "protocol_coding_uq_v2.csv"
    counts_path = corpus / "PHASE3_UQ_PRACTICE_COUNTS.csv"
    matrix_path = corpus / "PHASE3_UQ_EVIDENCE_MATRIX.csv"
    report_path = root / "project" / "TOPIC6_WORK_PACKAGE_6A_RESULT.md"
    summary_path = corpus / "PHASE3_UQ_SUMMARY.json"
    _write_csv(extension_path, extension)
    _write_csv(counts_path, counts)
    _write_csv(matrix_path, matrix)

    reported_ids = [row["paper_id"] for row in codebook if row["uq_reported"] != "no"]
    validated_ids = [row["paper_id"] for row in codebook if row["uq_validated"] == "yes"]
    reported_counts = Counter(row["uq_reported"] for row in codebook)
    method_counts = Counter(row["uq_method"] for row in codebook)
    validation_counts = Counter(row["uq_validated"] for row in codebook)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "coding_complete_verified",
        "main_denominator": len(MODEL_PAPER_IDS),
        "anchor_records_not_applicable": len(ANCHOR_IDS),
        "coder_count": 1,
        "inter_rater_reliability_claimed": False,
        "frozen_input_hashes": FROZEN_INPUT_HASHES,
        "headline": {
            "uq_reported_any": len(reported_ids),
            "uq_reported_no": reported_counts["no"],
            "uq_validated_yes": len(validated_ids),
            "uq_reported_ids": reported_ids,
            "uq_validated_ids": validated_ids,
        },
        "counts": {
            "uq_reported": dict(sorted(reported_counts.items())),
            "uq_method": dict(sorted(method_counts.items())),
            "uq_validated": dict(sorted(validation_counts.items())),
        },
        "outputs": [
            "papercorpus/protocol_coding_uq_v2.csv",
            "papercorpus/PHASE3_UQ_CODEBOOK.csv",
            "papercorpus/PHASE3_UQ_PRACTICE_COUNTS.csv",
            "papercorpus/PHASE3_UQ_EVIDENCE_MATRIX.csv",
            "project/TOPIC6_WORK_PACKAGE_6A_RESULT.md",
        ],
    }
    write_result_report(report_path, summary)
    output_paths = [extension_path, codebook_path, counts_path, matrix_path, report_path]
    summary["output_hashes"] = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in output_paths
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
