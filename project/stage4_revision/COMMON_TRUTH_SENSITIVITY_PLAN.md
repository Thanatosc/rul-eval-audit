# Frozen Common-Truth-Scale Sensitivity Plan

**Analysis ID:** `common_truth_v1`  
**Frozen:** 2026-08-13, before Stage 4 read or analysis of prediction values  
**Status at freeze:** not executed  
**Purpose:** reviewer-requested secondary diagnostic for R1/R2  
**Primary Kill Test membership:** false  
**May change `kill_v1` PASS:** false

## Inputs

- Exactly 120 completed `kill_v1` `preds_test.parquet` files.
- Path-and-file-hash manifest SHA-256:
  `097e0daf846bd5460630451ddc3fa84dabac35a508cb4c322ea627f2b64a6c5b`.
- Total input bytes at freeze: `18,806,024`.
- `configs/kill_test.yaml`:
  `370643e71dcb15eb9a9889066fe729069859a2d112a513ce98de30092a3d91ef`.
- `protocols/unified_v1.md`:
  `8e0bcda253ab008079a82d536b011d44a31e4cfa308c6df0032c481daea3de44`.
- `results/KILL_TEST_STATUS.csv`:
  `cadf78128a9552a62d2e890664e6803bcf17dd007124f5207943c93bf1aaa81f`.
- `results/KILL_TEST_DECISION.json`:
  `cf4747dbdc42bd3dabac85039ae1ec03fb2e5f75bb1460e5fd6fbbc451b61263`.
- NASA archive:
  `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.

Official-test raw RUL is reconstructed from the verified test tables and terminal
RUL vectors using the existing loader. Required source hashes are:

| File | SHA-256 |
|---|---|
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD004.txt` | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |
| `test_FD004.txt` | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |

## Frozen Estimands

For every retained model prediction, preserve `pred_rul` exactly; do not clip,
cap, shift, or recalibrate it. Select one terminal prediction per official test
engine by maximum cycle, matching the primary endpoint rule. Reconstruct:

- `raw_truth = raw_rul`;
- `capped_truth = min(raw_rul, 125)`.

For every dataset-model-sensor-seed-native-training-label cell, calculate RMSE
against both common truths. Then pair the predictions from linear-uncapped-trained
and piecewise-125-trained models within dataset, model, sensor set, seed, and
common truth:

`delta = RMSE(linear-trained predictions, common truth) - RMSE(capped-trained predictions, common truth)`.

Report all 24 five-seed mean deltas: 12 contexts x two common truths. Also report
the maximum absolute mean delta for each truth, strict LSTM-versus-CNN ranking
reversals across native training labels on each common truth, and an explicit
comparison with the native-truth 20.5323 contrast. The implementation may scan
the other configured model pairs as an exploratory descriptive extension, but
any reversal found outside the pre-listed LSTM-versus-CNN check must be labelled
as exploratory and must not be represented as prospectively specified. No
p-value is required; this is a descriptive estimand clarification with five
fixed seeds.

## Validation

1. All 120 registered input paths exist, have unique run IDs, and match hashes.
2. Saved `true_rul` exactly matches the native label reconstructed from raw RUL.
3. Endpoint selection yields 100 FD001 and 248 FD004 engines per run.
4. Unit/cycle keys and raw/capped truths are identical across paired native-label
   runs within dataset, model, sensor, and seed.
5. Original `pred_rul` and all upstream result files remain byte-unchanged.
6. Outputs are deterministic CSV/JSON plus a validation report and tests.

## Interpretation Boundary

The raw-truth result asks how both trained predictors perform on the uncapped
physical horizon; the capped-truth result asks how both perform on the capped
benchmark target. Neither isolates training-target choice from model fitting,
optimization, or output-scale adaptation. The diagnostic clarifies how much of
the native-label contrast persists under common evaluation truth; it is not a
new registered test, a causal decomposition, or grounds to alter the frozen PASS.
