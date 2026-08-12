# Unified Evaluation Protocol v1

Status: `frozen_pending_execution_authorization`

Frozen: August 11, 2026. This document fixes the shared evaluation and artifact
contract for Topic 7 and the downstream Topic 6 conformal-RUL interface. A new
factor, threshold, aggregation rule, or preprocessing boundary requires a new
protocol version. Freeze and readiness do not authorize a real model run.

## Data Identity and Partition

The source is NASA's official C-MAPSS double-ZIP distribution. The preserved
outer archive is `data/raw/C-MAPSS_Turbofan.zip`, SHA-256
`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`,
accessed August 11, 2026. The archive has no explicit license file; the project
retains attribution and does not redistribute the archive by default.

Each training fleet is split before window generation by complete engine
`unit_id`, using seed 42 and the deterministic algorithm recorded in
`configs/splits/FDxxx_seed42.json`:

- approximately 70% train;
- approximately 15% validation, used for model selection and early stopping;
- approximately 15% calibration, never used for training, tuning, scaler fit,
  or model selection and reserved for Topic 6;
- the official test fleet remains outside this split.

The file-verified train/test unit counts are FD001 100/100, FD002 260/259,
FD003 100/100, and FD004 249/248. These counts override the FD004 train/test
labels accidentally reversed in the distributed `readme.txt` because they are
verified directly from unit identifiers and the corresponding RUL vectors.

## Target and Features

The two Kill Test label levels are applied consistently to train, validation,
calibration, and evaluation truth:

- `piecewise_125`: `true_rul = min(raw_rul, 125)`;
- `linear_uncapped`: `true_rul = raw_rul`.

The feature factor is either all 21 sensor columns or the literature-supported
common 14-sensor set `[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]`.
The earlier `common_12` draft is withdrawn because the frozen corpus does not
support a canonical 12-sensor identity. Operational settings and cycle number
are excluded from the Kill Test input so the sensor factor has an exact meaning.

A MinMax scaler to `[0, 1]` is fitted only on rows from registered training
units and only after the unit split. Validation, calibration, and official test
rows are transformed without refitting. Train-plus-test scaling is a separate,
explicitly contaminated diagnostic; it is not one of the 120 primary cells and
cannot determine PASS or FAIL.

## Windows and Prediction Populations

Windows have length 30 and stride 1 and never cross an engine boundary.
Training, validation, and calibration use all complete windows. Prediction
artifacts retain all complete windows so Topic 6 can reconstruct stage- or
cycle-conditional analyses. Primary RMSE and NASA Score use exactly the last
valid window for each official test engine. A test trajectory shorter than 30
cycles is left-padded by repeating its first observation for endpoint prediction
only; no training trajectory is padded.

The metric endpoint populations are therefore 100 engines for FD001, 259 for
FD002, 100 for FD003, and 248 for FD004. No FD001 denominator is generalized to
another subset.

## Models and Fixed Budgets

The Kill Test uses LSTM, 1D-CNN, and LightGBM. LSTM has one 64-unit layer.
The CNN uses 64/64 channels, kernels 5/3, ReLU, and adaptive average pooling.
Both neural models use Adam, learning rate 0.001, weight decay 0.00001, batch
size 256, at most 50 epochs, and validation-RMSE early stopping with patience 8.
LightGBM flattens the same chronological 30-by-sensor window without engineered
summary statistics and uses 500 trees, learning rate 0.05, 31 leaves, full row
and column sampling, one CPU thread, and deterministic mode. No per-cell tuning
is permitted. Seeds are 11, 23, 37, 53, and 71.

The shared 160-cell asset grid adds a Transformer encoder and point/quantile
output modes across FD001-FD004. Point output uses MSE. Quantile output uses
q10/q50/q90 pinball objectives; LightGBM fits one model per quantile and neural
models use three heads. Reported `pred_rul` equals q50 in quantile artifacts.
That grid remains pending and is not part of the 120-cell Kill Test decision.

## Metrics

For each endpoint, `d = pred_rul - true_rul`.

`RMSE = sqrt(mean(d^2))`.

The PHM08/NASA Score is summed over engine endpoints:

- if `d < 0`, penalty = `exp(-d / 13) - 1`;
- if `d >= 0`, penalty = `exp(d / 10) - 1`.

Both metrics are always reported. The primary Kill Test threshold is expressed
only in RMSE; NASA Score cannot be selected post hoc as a rescue criterion.

## Registered Kill Test Decision

The primary matrix is
`2 datasets x 3 models x 2 labels x 2 sensor sets x 5 seeds = 120 cells`.
Registered label contrasts are paired within dataset/model/sensor/seed;
registered sensor contrasts are paired within dataset/model/label/seed. A
ranking reversal occurs when any model pair changes strict ordering by mean
five-seed RMSE between registered protocol levels within a dataset.

- PASS: at least one absolute registered mean RMSE effect is at least 1.0, or
  at least one registered model-pair ranking reversal occurs.
- FAIL: the maximum absolute registered mean RMSE effect is below 0.5 and no
  registered ranking reversal occurs.
- INCONCLUSIVE: every other complete outcome, or any missing/failed primary
  cell until a rerun under the unchanged configuration completes it.

No primary cell is imputed. Thresholds and factors cannot be changed after
outcomes are observed to rescue the central claim.

## Run Artifacts

Each real run must write:

```text
results/runs/<run_id>/checkpoint/
results/runs/<run_id>/preds_val.parquet
results/runs/<run_id>/preds_calib.parquet
results/runs/<run_id>/preds_test.parquet
results/runs/<run_id>/meta.json
```

Prediction rows contain `unit_id`, `cycle`, `true_rul`, and `pred_rul`;
quantile runs add `pred_q10`, `pred_q50`, and `pred_q90`. Metadata binds the
protocol hash, split hash, seed, model, dataset, output mode, code revision,
timestamps, duration, and metrics. Checkpoints and all three prediction tables
are mandatory even when Topic 7 reports endpoint aggregates only.

Because the current project tree has not yet been committed, the execution gate
uses the canonical ordered SHA-256 of all `src/rul_audit/**/*.py` path-and-byte
pairs as the frozen code revision. A later Git commit may replace this identifier
only through a recorded configuration version; a hash mismatch blocks execution.

## Execution Boundary

Synthetic smoke artifacts may validate the loader, scaler, windows, model
interfaces, metrics, checkpoints, metadata, and Parquet schemas. They carry
`data_class=generated_synthetic_smoke_only` and are not research results. Real
C-MAPSS execution begins only after a readiness PASS and separate explicit user
authorization. At this freeze, completed real cells equal zero.
