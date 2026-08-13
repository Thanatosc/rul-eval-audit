# Common-Truth-Scale Sensitivity Validation

## Status

- Analysis: `common_truth_v1`
- Result: **COMPLETED AND VALIDATED**
- Input runs: 120/120
- Run-by-truth endpoint metrics: 240
- Paired five-seed label contrasts: 24
- Prediction clipping: none
- Upstream artifact errors: 0

## Result Boundary

The maximum absolute five-seed mean difference between linear-trained and
capped-trained predictions was `11.9195` RMSE on common raw-RUL truth and
`25.7227` RMSE on common piecewise-125 truth. These values compare
training-target/model bundles on fixed evaluation truths. They do not replace the
native-truth 20.5323 task-definition contrast, isolate a causal training-label
effect, or change the frozen `kill_v1` PASS decision.

The common-truth analysis found 3 strict model-pair
ranking reversals across the native training-label levels under the two fixed
evaluation truths. Full rows are retained in
`COMMON_TRUTH_RANKING_REVERSALS.csv`.

## Validation Checks

- All frozen file hashes and the 120-file prediction manifest matched.
- Saved native truth matched reconstructed raw/capped truth for
  2,657,280 prediction rows.
- Every run yielded the registered endpoint count: 100 for FD001 and 248 for FD004.
- Predictions were not clipped, capped, shifted, retrained, or recalibrated.
- The original Kill Test status, decision, protocol, and prediction files were read-only.
- The analysis is secondary and descriptive; no new p-value or preregistration claim is made.
