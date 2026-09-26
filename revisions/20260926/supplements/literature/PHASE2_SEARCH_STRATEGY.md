# Phase 2 Search Strategy

Status: `phase_2_complete`; last searched: 2026-08-11.

This document records acquisition and screening only. It is not a literature
synthesis; Phase 2 closure depends on the complete validated artifact bundle
recorded in `project/pipeline_state.yaml`.

## Scope

The primary sampling frame is peer-reviewed C-MAPSS or N-CMAPSS RUL work from
2021-2026 that exposes at least one evaluation-protocol field. Historical
dataset/benchmark sources and one adjacent-domain benchmark-audit precedent are
retained as labelled anchors, outside the contemporary model-paper denominator.

### Sources searched

- Crossref and OpenAlex for metadata discovery and DOI identity.
- Unpaywall for DOI-level OA status.
- arXiv for exact-title author manuscripts.
- NASA NTRS, publisher proceedings, and publisher static assets for anchors.
- Exact-title web search for author and institutional repository copies.
- Publisher HTML and linked supplements when the article page itself supplies
  the complete citable text.
- Managed extraction from an exact DOI-bearing institutional bitstream when
  direct download is blocked; this route is labelled remote-only and never
  receives a fabricated local hash or preflight verdict.
- User-provided lawful copies when official/OA retrieval was unavailable;
  identity and read-integrity checks remain mandatory, and redistribution
  rights are not inferred from local availability.
- User-entitled publisher access for C10; the IEEE institutional restriction is
  recorded and the PDF is excluded from redistribution.
- CORE and OpenAIRE only when an exact identity could be established.

Generic repository fallback is disabled after three documented false-positive
PDFs. Sci-Hub is excluded from every acquisition route.

### Search concepts

Primary concepts:

```text
(C-MAPSS OR CMAPSS OR N-CMAPSS)
AND (remaining useful life OR RUL OR prognostics)
AND (evaluation OR benchmark OR protocol OR leakage OR window
     OR normalization OR sensor selection OR score)
```

Acquisition follow-up used exact title, DOI, first author, `PDF`, `repository`,
and `arXiv`. Exact strings and provider outcomes are in `SEARCH_LOG.md`.

## Eligibility

Include a contemporary model paper when it:

1. evaluates RUL on C-MAPSS or N-CMAPSS;
2. provides enough full text to code at least one protocol field;
3. has title/author/DOI identity verified against authoritative metadata; and
4. is available through an official, OA, author, or institutional full-text
   copy, including complete official HTML or an exact remote institutional PDF.

Retain a source as a labelled anchor when it defines the dataset/metric, reviews
historical benchmarking, documents data quality, or supplies a directly relevant
benchmark-audit precedent.

Exclude or defer records when full-text identity mismatches, only an abstract is
available, the source is unrelated to RUL protocol practice, or no acceptable
full-text route has been found. Access failure is not treated as evidence that a
paper lacks protocol details. A remote-only PDF is kept distinct from a local
PDF with a SHA-256 manifest row and preflight PASS.

## Screening State

| Stage | Count | State |
|---|---:|---|
| Records identified | 25 | fixed candidate register |
| Duplicates removed | 0 | DOI/title check |
| Records title/abstract screened | 25 | complete |
| Full texts sought | 25 | complete |
| Full texts not retrieved | 0 | acquisition gap closed |
| Full texts assessed and coded | 25 | 24 local PDFs with preflight PASS; 1 exact remote institutional PDF without local preflight |
| Contemporary model-protocol papers | 19 | separate denominator |
| Dataset/historical/adjacent anchors | 6 | excluded from model denominator |

The count is an acquisition-stage PRISMA analogue, not a claim of PRISMA 2020
systematic-review compliance. No candidate has been silently discarded.

## Distributional Coverage

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: time distribution
- Concentration: 2021-2026 = 22/25 (88.0%) of verified full texts
- Advisory: This is a coverage-distribution signal, not a defect.
- Search response: no expansion; the registered sampling frame deliberately
  emphasizes 2021-2026 and retains three older foundational anchors.

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational benchmark/model/dataset studies = 23/25 (92.0%)
- Advisory: This is a coverage-distribution signal, not a quality downgrade.
- Search response: no expansion; the RQ concerns executable evaluation
  protocols. The two historical review sources are retained to cover prior
  cross-paper guidance.

No venue-family concentration reaches 70%. Geography is omitted because the
sources evaluate shared simulated/public datasets rather than sampled regional
populations, and study-site metadata would not be meaningful.

## Acquisition Closure

- All 25 screened candidates have an accepted exact full text and protocol or
  anchor coding. No human full-text request list remains open.
- C02-C04, C06-C07, C09, C10, and C17 were supplied by the user and passed
  title/author/venue/DOI identity checks plus local PDF preflight.
- C08 uses the exact author-hosted manuscript and retains its official author
  code repository link.
- C10 is a user-entitled IEEE publisher PDF with a SOUTHWEST JIAOTONG
  UNIVERSITY license notice; it is coded locally and must not be redistributed.
- C13 remains an exact remote Dicle institutional bitstream with complete text,
  but direct Dicle/Wiley download is HTTP 403. It is the only no-local-PDF and
  no-preflight record.
- Semantic Scholar body snippets and journal metrics are unavailable in the
  current CLI configuration; absence is recorded rather than inferred as a
  negative result.

## Reproducibility Files

- Candidate decisions: `candidate_corpus.csv`
- Exact searches and negative outcomes: `SEARCH_LOG.md`
- PDF checksums and acquisition routes: `FULLTEXT_MANIFEST.csv`
- Protocol extraction: `protocol_coding.csv`
- Machine-readable counts: `PRISMA_COUNTS.csv`
