# Supplement S4. Bounded training-budget sensitivity

## S4.1 Purpose and scope

This additional arm checks the nine fits flagged for a recent validation best at
the final budget cap in the original corrective grid. It retains the original
96 grid fits and two numerical-scale diagnostics. Nine additional fits bring the
total to 107, with 321 validation/calibration/test prediction tables across those
fits. Historical prediction inputs are supplied separately and are not counted
as new fits.

All nine flagged configurations were included, without selecting them by official
test performance. The configuration list and the threefold caps were frozen in
`budget_sensitivity/BUDGET_SPEC.json` before generating the additional results.
Historical and primary-grid test results were already known; this remains an
exploratory sensitivity analysis, not a preregistered confirmatory experiment.

| ID | Subset | Partition seed | Model | Training target | Sensors |
| --- | --- | --- | --- | --- | --- |
| B1 | FD002 | 137 | LightGBM | capped125 | 14 |
| B2 | FD003 | 137 | LightGBM | capped125 | 14 |
| B3 | FD004 | 137 | LightGBM | capped125 | 14 |
| B4 | FD004 | 271 | LightGBM | capped125 | 14 |
| B5 | FD004 | 42 | LSTM | capped125 | 14 |
| B6 | FD004 | 137 | LightGBM | raw | 14 |
| B7 | FD004 | 271 | LightGBM | capped125 | 21 |
| B8 | FD004 | 42 | LightGBM | raw | 14 |
| B9 | FD004 | 42 | LightGBM | capped125 | 21 |

All nine use initialization seed 11. B1–B5 are core fits; B6–B9 complete the
target/input factorial design. Each LightGBM fit is deterministic and contributes
one fitting record. Full run identifiers are in the CSV files and specification.

## S4.2 Execution and selection rules

The LSTM cap was increased from 150 to 450 epochs and the LightGBM cap from
1,500 to 4,500 trees. Architecture, data, engine partitions, inputs, target and
numerical scale, preprocessing, initialization, optimizer, scheduler, validation
objective and early-stopping patience were unchanged. The original neural
optimizer state was not archived, so fitting restarted from the same
initialization. It did not resume a best checkpoint with a reset optimizer.

The original training-curve prefix was checked against the control; the neural
checkpoint selected at the original cap was exactly equal tensor by tensor.
All three original prediction tables were reproduced at the original selected
checkpoint or tree iteration. Test errors did not choose the extended checkpoint
or stopping iteration. The extended model uses the best validation RMSE.

All nine fits stopped by validation patience before the new maximum. The tree
fits evaluated 1,549–3,455 iterations; the LSTM selected epoch 157 and stopped
at epoch 172. The FD003 tree retained iteration 1,499, so its predictions were
unchanged. The 0 remaining cap flags refer to this
extended arm; the nine original flags remain in the preserved primary records.
Validation stopping is a practical selection rule and does not establish global
convergence or optimal hyperparameters.

| ID | Original selected/evaluated | Extended selected/evaluated | Original validation RMSE | Extended validation RMSE | Change | Cap flag |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 1500/1500 | 3405/3455 | 18.5980 | 18.3854 | -0.2126 | no |
| B2 | 1499/1500 | 1499/1549 | 14.4516 | 14.4516 | +0.0000 | no |
| B3 | 1494/1500 | 2085/2135 | 20.3408 | 20.2730 | -0.0679 | no |
| B4 | 1487/1500 | 1615/1665 | 17.8029 | 17.7796 | -0.0233 | no |
| B5 | 148/150 | 157/172 | 18.5100 | 18.4785 | -0.0315 | no |
| B6 | 1489/1500 | 1590/1640 | 56.0775 | 56.0639 | -0.0136 | no |
| B7 | 1499/1500 | 1551/1601 | 17.6648 | 17.6554 | -0.0094 | no |
| B8 | 1500/1500 | 1591/1641 | 47.6099 | 47.5685 | -0.0415 | no |
| B9 | 1499/1500 | 1620/1670 | 18.5525 | 18.5389 | -0.0136 | no |

## S4.3 Paired changes in official-endpoint errors

Differences below are extended minus original; a negative value favors the
extended fit. Each interval uses 5,000 paired bootstrap samples of complete
official test engines, with generator seed 20260926. The intervals condition on
the two validation-selected fits and are marginal, descriptive intervals. They
do not constitute equivalence tests or simultaneous family-wise guarantees.

The endpoint RMSE changes range from -0.0237 to +0.1009 cycles
on capped truth and -0.0600 to +0.0813 cycles on raw truth.
Validation gains do not require test-error gains: several endpoint errors rose
slightly. B2's zero-width difference interval reflects identical predictions.

| ID | Scoring truth | Original RMSE | Extended RMSE | RMSE change [95% interval] | Mean NASA change [95% interval] |
| --- | --- | --- | --- | --- | --- |
| B1 | capped125 | 16.2415 | 16.2833 | +0.0417 [-0.1230, +0.2030] | +0.0545 [-0.1211, +0.2646] |
| B1 | raw | 29.4573 | 29.4167 | -0.0406 [-0.2234, +0.1420] | -0.9171 [-5.6351, +3.9922] |
| B2 | capped125 | 14.8732 | 14.8732 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| B2 | raw | 16.2304 | 16.2304 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| B3 | capped125 | 17.3058 | 17.2961 | -0.0097 [-0.1248, +0.1035] | +0.0313 [-0.1782, +0.3060] |
| B3 | raw | 29.1900 | 29.1728 | -0.0172 [-0.1262, +0.0879] | +0.0114 [-0.6087, +0.6351] |
| B4 | capped125 | 17.7070 | 17.7044 | -0.0027 [-0.0667, +0.0649] | -0.0462 [-0.3043, +0.2168] |
| B4 | raw | 29.1183 | 29.1043 | -0.0140 [-0.0714, +0.0418] | +0.0097 [-0.3680, +0.3768] |
| B5 | capped125 | 19.1132 | 19.0895 | -0.0237 [-0.1876, +0.1184] | -0.2687 [-0.6567, +0.0441] |
| B5 | raw | 30.6613 | 30.6013 | -0.0600 [-0.1773, +0.0515] | -0.1064 [-1.2384, +1.1510] |
| B6 | capped125 | 38.9756 | 39.0765 | +0.1009 [-0.0449, +0.2469] | -297.5465 [-1412.8876, +494.6079] |
| B6 | raw | 34.1835 | 34.2374 | +0.0539 [-0.0847, +0.1949] | -9.2065 [-52.2336, +20.3575] |
| B7 | capped125 | 17.7178 | 17.7171 | -0.0007 [-0.0316, +0.0299] | -0.0299 [-0.0953, +0.0211] |
| B7 | raw | 28.9990 | 28.9930 | -0.0060 [-0.0383, +0.0268] | -0.0328 [-0.2207, +0.1726] |
| B8 | capped125 | 43.9009 | 43.9751 | +0.0742 [-0.0649, +0.2151] | -239.0091 [-816.7681, +211.8835] |
| B8 | raw | 38.3594 | 38.4407 | +0.0813 [-0.0503, +0.2152] | -19.9002 [-107.7779, +52.5224] |
| B9 | capped125 | 18.0309 | 18.0433 | +0.0124 [-0.0405, +0.0627] | +0.0519 [-0.0729, +0.2172] |
| B9 | raw | 29.1850 | 29.2013 | +0.0162 [-0.0310, +0.0639] | +0.1168 [-0.1848, +0.4545] |

## S4.4 Training-target and prediction-population contrasts

A sensitivity view substitutes the nine extended fits for their controls in the
96-cell grid. It does not add another 96 independently fitted models. All other
fits, predictions and scoring rules remain unchanged. The target contrasts still
use initialization 11 for LSTM and one deterministic LightGBM fit per cell.

All 24 fitted pairs favor capped training at official endpoints under common raw
truth, while all 24 favor raw training under both window-weighting rules. None
of the 144 target preferences changes across the two scoring truths and three
populations. The full ranges, including capped-truth window results, are below.
Differences are raw-trained minus capped-trained RMSE.

| Scoring truth | Population | Difference range (cycles) | Capped training lower | Raw training lower |
| --- | --- | --- | --- | --- |
| capped125 | endpoint | +8.276 to +28.357 | 24 | 0 |
| capped125 | equal_engine_windows | +11.812 to +45.871 | 24 | 0 |
| capped125 | pooled_windows | +11.026 to +49.435 | 24 | 0 |
| raw | endpoint | +4.570 to +14.848 | 24 | 0 |
| raw | equal_engine_windows | -22.245 to -3.097 | 0 | 24 |
| raw | pooled_windows | -33.943 to -5.258 | 0 | 24 |

The revised input contrasts, including their conditional engine intervals, are
retained in `sensitivity_paired_factor_contrasts.csv`; no subset or contrast is
omitted because its estimate is small or its interval includes zero.

## S4.5 Model winners and empirical intervals

RMSE and mean NASA-loss winning sets are unchanged in all 144 core contexts
(four subsets, three partitions, two neural-initialization contexts, two scoring
truths and three prediction populations). The same tree prediction appears as a
shared reference in the two neural-initialization contexts; these are not
independent tree replications.

The primary capped-truth endpoint analysis retains 6/24 RMSE-versus-NASA winner
conflicts. Replacing endpoints by equal-engine or pooled windows changes the
capped-truth RMSE winner in 5/24 contexts under either weighting, as before.
These counts are descriptive, dependent comparisons of the same test engines.

Residual intervals were recalibrated using the extended fits' calibration
predictions. The full per-engine and stage tables are supplied. No CNN fit was
retrained, so the reported FD004 CNN result remains 90.9% overall equal-engine
coverage and 73.2% [63.2%, 82.5%] near failure. Its nominal level is 90%; it still
does not establish stage-conditional or deployment coverage.

## S4.6 Numerical and artifact verification

The original 96-fit training source, specification, metadata and analysis tables
were preserved. The maximum absolute difference between original and replayed
prediction prefixes was 0.0; the
largest checked training/validation-curve difference was
5.68e-14. Source, split, scaler and input-data
hashes were checked by the runner.

Independent verification reloaded all 9 extended
models and reproduced 408,732 rows in 27 prediction
tables. It separately recomputed 54 truth/population scores and 36 paired
endpoint intervals. Maximum reload error was 0;
the maximum RMSE absolute/NASA relative score discrepancy was
1.42e-14. The machine-readable check is
`budget_sensitivity/BUDGET_VERIFICATION.json`.

## S4.7 Reproduction and interpretation

From the shared revision directory, the following commands analyze supplied
predictions and regenerate this supplement without fitting models:

    python -X utf8 budget_sensitivity/analyze_budget_checks.py
    python -X utf8 budget_sensitivity/build_budget_supplement.py

The full saved-model reload check additionally needs the official NASA inputs
and recorded training environment:

    python -X utf8 budget_sensitivity/verify_budget_checks.py

For fresh fitting, use a separate workspace with the original controls retained
and the corresponding extended outputs absent. Matching completed fits are
reused. Follow the frozen specification and run:

    python -X utf8 budget_sensitivity/run_budget_checks.py --model lstm
    python -X utf8 budget_sensitivity/run_budget_checks.py --model lightgbm

Keep the original source and specification unchanged. Hardware/software changes
may fail the exact-prefix check and require investigation. The checks support
robustness to this budget extension; external validity, noise/drift, calibration
uncertainty, architecture optimality and real maintenance benefits remain untested.

Files below are under `budget_sensitivity/analysis/` in Supporting Information S3.

| File | Contents |
| --- | --- |
| `training_budget_comparison.csv` | Nine controls, original/extended selected iterations, validation changes and remaining flags. |
| `new_run_score_changes.csv` | 54 scores and original-versus-extended differences: nine fits, two truths, three populations. |
| `paired_endpoint_budget_changes.csv` | 18 endpoint comparisons with paired-engine RMSE and mean NASA-loss intervals. |
| `sensitivity_all_run_scores.csv` | 576 scores from the primary grid with only the nine flagged fits substituted. |
| `sensitivity_metric_winner_contexts.csv` | 144 contexts with RMSE and NASA winning sets. |
| `metric_winner_changes.csv` | Original and substituted winner sets, including unchanged contexts. |
| `sensitivity_target_population_contrasts.csv` | 144 target contrasts from 24 fitted pairs on two truths and three populations. |
| `sensitivity_target_population_summary.csv` | Six summaries of target-comparison directions and ranges. |
| `target_preference_changes.csv` | All 144 original and substituted training-target contrasts. |
| `sensitivity_core_scores.csv` | 216 core summaries; endpoint intervals included. |
| `sensitivity_paired_factor_contrasts.csv` | 96 common-truth endpoint target/input contrasts and paired-engine intervals. |
| `sensitivity_uq_per_engine_stage.parquet` | Recalibrated coverage, width and interval-score contributions by engine and stage. |
| `sensitivity_uq_stage_summary.csv` | 108 stage summaries under the substituted core fits. |
