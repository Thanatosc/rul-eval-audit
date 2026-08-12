
# RUL Evaluation Audit

Reproducible code and aggregate artifacts for **Protocol-Conditioned Measurement
in C-MAPSS Remaining-Useful-Life Evaluation: A Literature-Practice Audit and
Controlled Re-Benchmark**.

The study asks how declared label construction, sensor selection, windowing,
preprocessing boundaries, prediction populations, and metric choices affect
reported C-MAPSS remaining-useful-life (RUL) results. It is an evaluation audit
and controlled re-benchmark, not a new forecasting architecture or a state-of-
the-art claim.

## Released study components

- 25-record frozen literature corpus: 19 model/protocol papers and 6 labelled anchors;
- 120-cell registered Kill Test over two subsets, three backbones, two labels,
  two sensor sets, and five seeds;
- 160-cell unified point/quantile grid over four subsets, four backbones, five
  seeds, and two output modes;
- 280-cell empirical uncertainty reference arm using residual split conformal
  prediction and conformalized quantile regression;
- frozen configurations, engine-unit split manifests, aggregate results,
  figures, schemas, and validation tests.

## Repository layout

```text
configs/        frozen experiment configurations and split manifests
data/           provenance and dataset acquisition instructions only
paper/figures/  generated study figures
papercorpus/    machine-readable literature coding (no source PDFs)
project/        public result and registration notes
protocols/      frozen evaluation protocol
results/        aggregate result tables, registers, and schemas
scripts/        analysis and UQ post-processing scripts
src/            reusable Python implementation
tests/          public core and data-backed validation tests
```

## Data boundary

The NASA C-MAPSS archive and extracted source tables are not redistributed. Use
the official URLs and SHA-256 in `data/DATASETS.md`, then extract the official
text files to `data/interim/cmapss/` if rerunning model training.

Restricted literature PDFs and publisher supplements are not included. The
machine-readable coding tables contain bibliographic metadata and bounded
protocol observations only.

Generated per-window prediction and interval artifacts are distributed through
a separate Zenodo dataset because they are too large and too granular for the
GitHub repository. Copy every dataset part into this repository root before
running the data-backed validation suite.

## Environment

Python 3.11 is required. The original run used Python 3.11.9; exact observed
package versions are recorded in `environment/requirements-lock.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

PyTorch CUDA wheels are platform-specific. The recorded experimental build was
`torch 2.13.0+cu130` on an NVIDIA RTX 4060 Laptop GPU. Install an appropriate
official PyTorch build for your platform before attempting neural retraining.

## Verification

Core tests do not require NASA source data or the companion Zenodo dataset.
The data-closure test is skipped until the dataset parts are restored:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

After restoring all companion dataset parts into the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\analyze_kill_test.py --root .
.\.venv\Scripts\python.exe scripts\analyze_unified_grid.py --root .
.\.venv\Scripts\python.exe scripts\analyze_postgrid.py --root .
.\.venv\Scripts\python.exe scripts\uq_reference_arm.py --root . --analyze
```

The UQ arm is an empirical reference under the declared engine-disjoint
calibration design. It is not a strict window-level, conditional, trajectory-
wise, or deployment coverage certificate.

## Citation and archival identifiers

- Software v0.1.0: <https://doi.org/10.5281/zenodo.21905029>
- Software concept DOI: <https://doi.org/10.5281/zenodo.21905028>
- Generated result dataset v1.0.0: <https://doi.org/10.5281/zenodo.21905033>
- Dataset concept DOI: <https://doi.org/10.5281/zenodo.21905032>
- GitHub release source: <https://github.com/Thanatosc/rul-eval-audit/tree/v0.1.0>

Use the version-specific software DOI when citing this exact code release and
the version-specific dataset DOI when using the generated prediction or
interval artifacts. Concept DOIs resolve to the latest published version.

## Authorship and declarations

Siyu Cai, School of Computing and Artificial Intelligence, Southwest Jiaotong
University. ORCID: 0009-0003-3716-0008.

This research received no external funding. The author declares no competing
interests.

## License

Software is released under the MIT License. Generated result data, figures,
coding tables, and documentation are released under CC BY 4.0; see `LICENSES/`.
Raw third-party datasets and publications are not included. NASA-derived factual
fields retained in prediction artifacts remain under applicable upstream terms;
the project does not assert ownership over those facts.
