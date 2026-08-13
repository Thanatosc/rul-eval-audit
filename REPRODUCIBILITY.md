
# Reproducibility guide

## Two levels of reproduction

1. **Analysis reproduction** uses the companion Zenodo prediction/interval
   dataset and does not refit models.
2. **Full computational rerun** additionally downloads the official NASA
   C-MAPSS source archive and retrains the registered cells.

Analysis reproduction is the recommended first step because the released
prediction artifacts are the direct inputs to all reported post-hoc statistics.

The secondary common-truth analysis is frozen separately from the primary Kill
Test. Its plan is retained at
`project/stage4_revision/COMMON_TRUTH_SENSITIVITY_PLAN.md`; its aggregate outputs
are under `results/common_truth/`. The all-model-pair scan beyond the pre-listed
LSTM-versus-CNN checks is exploratory and does not update `kill_v1`.

## Restore the companion result dataset

Every dataset ZIP has the same top-level directory,
`rul-eval-audit-results-v1.0.2/`. Extract all parts into one temporary directory,
then copy that directory's contents into the repository root. Parts contain
disjoint files and may be extracted in any order.

Expected generated assets:

- 120 Kill Test run directories;
- 160 unified-grid run directories;
- 280 UQ result directories;
- complete registered checkpoints and no NASA source rows.

The run directories retain `checkpoint/`, `meta.json`, and validation,
calibration, and test prediction Parquet files. UQ directories retain
calibration scores, interval tables, metadata, and summaries. Treat checkpoint
files as potentially unsafe if their checksums do not match this release.

## Dataset acquisition for retraining

Download the official NASA archive listed in `data/DATASETS.md`, verify
SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`,
and extract the official `train_FDxxx.txt`, `test_FDxxx.txt`, and
`RUL_FDxxx.txt` files under `data/interim/cmapss/`.

Verify the source data:

```powershell
.\.venv\Scripts\python.exe -m rul_audit.data.cmapss `
  --data-dir data/interim/cmapss --verify
```

## Frozen execution identifiers

- Kill Test source revision: `ecfb7d1991fdda11572d36ad61eb59e824bae09de3e8c0bd7418227b56f9d9b9`
- Unified grid source revision: `2cb56862cb299ac24bd5edc32fa65925096de28009f3c1b9088083428da6de69`
- Protocol SHA-256: `8e0bcda253ab008079a82d536b011d44a31e4cfa308c6df0032c481daea3de44`
- UQ implementation SHA-256: `b4ee424936b3333793e46d517fc79916f57403aef5f9d5d40b7d4377727e24c7`
- UQ frozen cell register SHA-256: `fb8bbb687feae1c85816c3f821872d38d70839b3041e7ed1ba406686c4be5c99`

These are content hashes recorded before or during the registered execution;
they are not Git commit identifiers.
