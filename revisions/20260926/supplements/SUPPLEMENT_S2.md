# Supplement S2. Corrective experiments and complete numerical results

## S2.1 Study status and experiment units

The analysis plan (`../ANALYSIS_SPEC.md`) was recorded before inspection of the new
corrective fits, after the historical benchmark results and rejection were known.
The work is exploratory. All 96 specified grid fits and both diagnostic fits
completed. Core fits comprise four subsets × three engine partitions × two LSTM
initializations, two CNN initializations and one deterministic LightGBM record.
The 36 factor fits complete the raw/capped target × 14/21-sensor design for LSTM
initialization 11 and LightGBM on FD001 and FD004. Tree results are stored once.

Independent test resampling uses engines, not windows or arbitrary seed labels.
Engine partitions reuse the available development population, and every partition
is evaluated on the same official test engines. Partition ranges, initialization
variation and conditional test-engine intervals describe different sources of
variation and are not combined into a pooled effective sample size.

## S2.2 Data, partition and retained-window counts

The following counts are calculated from the actual split and prediction records.
Within each cell, numbers are fitting / validation / calibration / official test.
Development trajectories are complete; official test trajectories are truncated.
The original NASA archive SHA-256 is
`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
The actual per-file identities used by the new runs appear below. Engine counts
are taken from unit identifiers, including FD004's 249 development/248 test
engines. Operating settings are not supplied as features, and no operating-regime
normalization, smoothing or target-aware sensor selection is fitted.

| Subset | Partition seed | Engine counts | Window counts |
|---|---|---|---|
| FD001 | 42 | 70/15/15/100 | 12377/2725/2629/10196 |
| FD001 | 137 | 70/15/15/100 | 12255/2858/2618/10196 |
| FD001 | 271 | 70/15/15/100 | 12626/2561/2544/10196 |
| FD002 | 42 | 182/39/39/259 | 32245/7035/6939/26511 |
| FD002 | 137 | 182/39/39/259 | 32762/6866/6591/26511 |
| FD002 | 271 | 182/39/39/259 | 32737/6690/6792/26511 |
| FD003 | 42 | 70/15/15/100 | 15376/3253/3191/13696 |
| FD003 | 137 | 70/15/15/100 | 15305/3437/3078/13696 |
| FD003 | 271 | 70/15/15/100 | 14779/3636/3405/13696 |
| FD004 | 42 | 174/38/37/248 | 38665/7701/7662/34092 |
| FD004 | 137 | 174/38/37/248 | 37724/8187/8117/34092 |
| FD004 | 271 | 174/38/37/248 | 38422/7406/8200/34092 |

| Source file | SHA-256 |
|---|---|
| train_FD001.txt | 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8 |
| test_FD001.txt | 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851 |
| RUL_FD001.txt | a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca |
| train_FD002.txt | dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f |
| test_FD002.txt | de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02 |
| RUL_FD002.txt | c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b |
| train_FD003.txt | 2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2 |
| test_FD003.txt | 299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790 |
| RUL_FD003.txt | df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb |
| train_FD004.txt | 27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218 |
| test_FD004.txt | 1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5 |
| RUL_FD004.txt | 196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1 |

Windows have 30 cycles and stride one. The last observation is the prediction
time. Engines shorter than 30 observations contribute one endpoint, padded on the
left by their first observation. Other engines contribute all windows from cycle
30 onwards; the first 29 cycles do not receive separate predictions. Min–max
scalers use only fitting-engine observations and are applied without clipping.
Raw RUL and capped RUL are kept as separate columns, and all scores are in cycles.

## S2.3 Fitting and selection

The LSTM is one 64-unit layer followed by a scalar linear head. The 1D CNN uses
two convolutions with 64 channels (kernel sizes 5 and 3), ReLU activations,
global average pooling and a scalar linear head. Neural targets are divided by
125 for both raw and capped fits; predictions are converted back to cycles.
No prediction is clipped. Adam uses learning rate 0.001, weight decay 0.00001,
batch size 256 and deterministic shuffling. Validation uses every retained
validation window, pooled equally, and the fitting target of the run. Thus
validation selection in a training-target contrast is part of the trained bundle.
Calibration data do not select checkpoints or hyperparameters.

The corrective neural recipe halves the learning rate after five unimproved
validation epochs (PyTorch ReduceLROnPlateau with its default relative threshold),
down to 0.00001. The saved checkpoint has the lowest validation RMSE, using an
absolute improvement tolerance of 1e-9 cycles. Early stopping requires 15
unimproved epochs and at least 20 elapsed epochs. The initial cap is 150 epochs.
If the best epoch is in the last 15 epochs and the best RMSE improved by at least
0.1 cycles during those epochs, the same optimizer/run continues to a cap of 300.

LightGBM receives the flattened 30-cycle window, uses squared loss, 31 leaves,
learning rate 0.05, full row/feature sampling, zero L2 penalty, deterministic
single-thread CPU fitting, at most 1,500 trees and validation stopping patience 50.
The first metric is RMSE. Predictions use the selected best iteration. Further
library defaults are reproducible from the recorded environment and source.

| Model | Fits | Trainable neural parameters | Selected epoch/iteration range | Seconds: median (range) |
|---|---|---|---|---|
| 1D CNN | 24 | 16961 | 6–109 | 36.6 (12.4–83.5) |
| LightGBM | 30 | Tree count below | 203–1500 | 66.4 (20.3–194.7) |
| LSTM | 42 | 20545, 22337 | 4–148 | 56.0 (6.7–230.9) |

Runtime includes fitting, validation, checkpoint handling and final split
prediction in each model routine. It excludes dataset loading and window
construction and is not an isolated throughput benchmark. Neural runs used an
NVIDIA GeForce RTX 4060 Laptop GPU and tree runs one CPU thread; the two jobs ran
concurrently. These measurements do not establish a hardware-independent speed
ranking. Software versions are supplied in `../ENVIRONMENT_SNAPSHOT.json`.

There were 0 neural budget extensions and
9 grid fits flagged for a recent best at the final cap. For neural fits
the flag means the best epoch is in the final 15; for trees it means 1,500
iterations were evaluated and the best is above 1,450. A flagged fit is not
declared converged. Absence of the flag is also not proof of a global optimum.

| Flagged run | Selected epoch/iteration | Evaluated epoch/iteration |
|---|---|---|
| `core__fd002__s137__lightgbm__i11__piecewise_125__common_14__scale125` | 1500 | 1500 |
| `core__fd003__s137__lightgbm__i11__piecewise_125__common_14__scale125` | 1499 | 1500 |
| `core__fd004__s137__lightgbm__i11__piecewise_125__common_14__scale125` | 1494 | 1500 |
| `core__fd004__s271__lightgbm__i11__piecewise_125__common_14__scale125` | 1487 | 1500 |
| `core__fd004__s42__lstm__i11__piecewise_125__common_14__scale125` | 148 | 150 |
| `factor__fd004__s137__lightgbm__i11__linear_uncapped__common_14__scale125` | 1489 | 1500 |
| `factor__fd004__s271__lightgbm__i11__piecewise_125__all_21__scale125` | 1499 | 1500 |
| `factor__fd004__s42__lightgbm__i11__linear_uncapped__common_14__scale125` | 1500 | 1500 |
| `factor__fd004__s42__lightgbm__i11__piecewise_125__all_21__scale125` | 1499 | 1500 |

These counts and curves describe the preserved primary grid. Supplement S4
reports a separate threefold-budget extension for all nine flagged fits. All
nine extended fits stopped by validation patience before their new caps. The
primary records and this table are retained so the original stopping decisions
remain inspectable; the extended models and sensitivity results are supplied
under `budget_sensitivity/` in Supporting Information S3.

## S2.4 Original-recipe diagnostic

The two FD002 fits hold architecture, 14-sensor inputs, split 42, initialization
11 and the original constant-learning-rate, 50-epoch, patience-eight recipe
fixed. Numerical target scale is the changed setting. The original-scale test
predictions exactly reproduce the corresponding archived prediction array.
Scaling also changes relative regularization and optimizer numerics, so it is a
recipe diagnostic rather than causal identification of one internal mechanism.

| Target divisor | Selected epoch | Validation RMSE | Endpoint test RMSE | All-test prediction range |
|---|---|---|---|---|
| 1.0 | 49 | 41.805493 | 43.510689 | 0.002914 |
| 125.0 | 42 | 17.354044 | 17.197940 | 139.981860 |

The initial diagnostic metadata mislabeled window counts as `units`. These counts
were moved unchanged to `prediction_rows`, and distinct engine counts now occupy
`units`. Original metadata bytes and hashes are retained alongside the corrected
records; `../DIAGNOSTIC_METADATA_CORRECTION.json` documents this correction.
No checkpoint, curve, prediction or error was changed by that metadata repair.

For the archived LSTM runs, the final recurrent state lies componentwise in
[-1,1], so a scalar linear head with weights w and bias b has upper enclosure
b+sum(abs(w)) in real arithmetic. A small distance below this enclosure implies
that the head-weighted recurrent representation lies near its maximizing bounded
direction. The complete 20-run prediction-dispersion table is supplied in
`../historical_evidence/lstm_prediction_degeneracy.csv`; 16 runs span less than
0.004 cycles over calibration and test records. This locates the observed
degeneracy in the fitted representation/output geometry, but does not identify
which optimization dynamics caused it.

## S2.5 Common-truth scores and resampling

For each fitted model, both capped-125 and raw truth are scored over (i) one
official endpoint per engine, (ii) equal engine weights after averaging losses
within each engine's windows and (iii) equal weights for all retained windows.
Squared losses are averaged before taking the square root. Mean NASA loss is
exp(-d/13)-1 for underprediction and exp(d/10)-1 otherwise, with d = prediction
minus scoring truth. Its endpoint sum equals its mean times the engine count.

The 5,000 bootstrap replicates use generator seed 20260926 and resample complete
test engines with replacement. Paired contrasts reuse the same sampled engine
indices. Core tables average loss contributions over the two neural
initializations within engine, before resampling or taking a square root. They
are not prediction ensembles. LightGBM contributes one fitting record. The 2.5th
and 97.5th percentiles are conditional, marginal intervals; no familywise
significance test is applied to the many comparisons.

All three primary partitions and initialization ranges are shown here. The
machine-readable file also includes raw-truth and window-population scores.

| Subset | Split | Model | RMSE | 95% engine interval | Initialization RMSE range | Mean NASA loss |
|---|---|---|---|---|---|---|
| FD001 | 137 | 1D CNN | 15.954 | 13.329–18.465 | 15.673–16.230 | 4.976 |
| FD001 | 137 | LightGBM | 13.303 | 11.099–15.358 | 13.303–13.303 | 2.898 |
| FD001 | 137 | LSTM | 13.809 | 11.505–16.058 | 13.562–14.051 | 3.732 |
| FD001 | 271 | 1D CNN | 15.126 | 12.587–17.641 | 15.071–15.180 | 4.341 |
| FD001 | 271 | LightGBM | 13.824 | 11.639–15.893 | 13.824–13.824 | 3.329 |
| FD001 | 271 | LSTM | 13.532 | 11.297–15.802 | 13.265–13.794 | 3.246 |
| FD001 | 42 | 1D CNN | 15.993 | 13.151–18.770 | 15.771–16.211 | 5.890 |
| FD001 | 42 | LightGBM | 13.767 | 11.569–15.937 | 13.767–13.767 | 3.339 |
| FD001 | 42 | LSTM | 13.604 | 11.257–15.930 | 13.426–13.780 | 3.623 |
| FD002 | 137 | 1D CNN | 22.215 | 20.303–24.161 | 22.041–22.386 | 21.601 |
| FD002 | 137 | LightGBM | 16.242 | 14.728–17.683 | 16.242–16.242 | 4.514 |
| FD002 | 137 | LSTM | 15.962 | 14.338–17.548 | 15.898–16.025 | 5.093 |
| FD002 | 271 | 1D CNN | 22.393 | 20.525–24.277 | 22.171–22.612 | 16.295 |
| FD002 | 271 | LightGBM | 16.005 | 14.493–17.503 | 16.005–16.005 | 4.415 |
| FD002 | 271 | LSTM | 16.930 | 15.270–18.569 | 16.912–16.949 | 6.160 |
| FD002 | 42 | 1D CNN | 22.482 | 20.568–24.455 | 22.409–22.554 | 18.252 |
| FD002 | 42 | LightGBM | 16.418 | 14.963–17.815 | 16.418–16.418 | 4.535 |
| FD002 | 42 | LSTM | 16.610 | 14.908–18.260 | 16.577–16.643 | 5.768 |
| FD003 | 137 | 1D CNN | 22.890 | 18.935–26.890 | 22.703–23.077 | 33.207 |
| FD003 | 137 | LightGBM | 14.873 | 12.085–17.814 | 14.873–14.873 | 6.853 |
| FD003 | 137 | LSTM | 13.693 | 10.801–16.515 | 13.576–13.810 | 4.604 |
| FD003 | 271 | 1D CNN | 20.320 | 17.100–23.472 | 17.264–22.972 | 19.196 |
| FD003 | 271 | LightGBM | 14.110 | 11.127–17.388 | 14.110–14.110 | 8.083 |
| FD003 | 271 | LSTM | 14.945 | 12.418–17.403 | 13.621–16.160 | 5.926 |
| FD003 | 42 | 1D CNN | 21.260 | 17.601–24.953 | 21.001–21.516 | 19.986 |
| FD003 | 42 | LightGBM | 14.713 | 12.147–17.282 | 14.713–14.713 | 5.554 |
| FD003 | 42 | LSTM | 13.694 | 11.433–15.928 | 12.562–14.740 | 4.574 |
| FD004 | 137 | 1D CNN | 27.775 | 25.458–30.063 | 26.997–28.532 | 51.210 |
| FD004 | 137 | LightGBM | 17.306 | 15.389–19.245 | 17.306–17.306 | 8.458 |
| FD004 | 137 | LSTM | 19.092 | 16.960–21.192 | 19.056–19.127 | 13.355 |
| FD004 | 271 | 1D CNN | 24.394 | 22.440–26.295 | 24.097–24.687 | 17.491 |
| FD004 | 271 | LightGBM | 17.707 | 15.699–19.657 | 17.707–17.707 | 9.632 |
| FD004 | 271 | LSTM | 19.547 | 17.623–21.486 | 18.887–20.186 | 8.861 |
| FD004 | 42 | 1D CNN | 24.994 | 23.016–27.009 | 24.848–25.138 | 20.615 |
| FD004 | 42 | LightGBM | 17.706 | 15.847–19.514 | 17.706–17.706 | 7.876 |
| FD004 | 42 | LSTM | 18.927 | 17.053–20.807 | 18.740–19.113 | 9.039 |

## S2.6 Stage-specific residual intervals

For each core fit, absolute residuals on all calibration windows and capped truth
are pooled. The half-width is order ceil((n+1)×0.9), capped at n, from the n
ordered scores. The interval is the point prediction plus/minus this half-width;
endpoints are not clipped. Calibration windows overlap and do not supply n
independent calibration units. This procedure is used as a standard empirical
reference, without an exchangeability or stage-conditional guarantee.

Coverage, width and interval score are averaged over the windows of each engine
within a raw-RUL stage, then over contributing engines. For neural summaries,
initialization-specific engine summaries are averaged before resampling engines.
The interval score is width plus 20 times the shortfall below the lower endpoint
or excess above the upper endpoint. The reference-split results follow; all
splits, conditional intervals and calibration counts are in the CSV files.
Because the symmetric half-width is constant within each fitted model, mean
width has no test-engine resampling variation when calibration is fixed. This
does not imply that width is known without calibration-sample uncertainty.
If every contributing engine has empirical coverage one, the nonparametric
bootstrap also returns a degenerate [1,1] interval. This boundary behavior cannot
rule out unobserved failures or establish perfect population coverage.

| Subset | Model | Raw-RUL stage | Engines | Windows | Coverage %, 95% interval | Width | Interval score |
|---|---|---|---|---|---|---|---|
| FD001 | 1D CNN | 0-30 | 25 | 332 | 99.1 [97.2, 100.0] | 57.04 | 57.69 |
| FD001 | 1D CNN | 31-80 | 45 | 1748 | 80.9 [72.1, 88.9] | 57.04 | 92.70 |
| FD001 | 1D CNN | >80 | 100 | 8116 | 95.8 [94.0, 97.4] | 57.04 | 61.86 |
| FD001 | LightGBM | 0-30 | 25 | 332 | 98.8 [97.2, 100.0] | 56.07 | 56.47 |
| FD001 | LightGBM | 31-80 | 45 | 1748 | 80.6 [72.0, 88.3] | 56.07 | 88.29 |
| FD001 | LightGBM | >80 | 100 | 8116 | 95.2 [93.0, 97.2] | 56.07 | 61.87 |
| FD001 | LSTM | 0-30 | 25 | 332 | 100.0 [100.0, 100.0] | 46.77 | 46.77 |
| FD001 | LSTM | 31-80 | 45 | 1748 | 80.2 [71.9, 87.8] | 46.77 | 87.39 |
| FD001 | LSTM | >80 | 100 | 8116 | 92.5 [90.0, 94.6] | 46.77 | 56.58 |
| FD002 | 1D CNN | 0-30 | 61 | 1087 | 89.8 [82.7, 95.7] | 83.72 | 116.04 |
| FD002 | 1D CNN | 31-80 | 130 | 4753 | 89.3 [84.7, 93.5] | 83.72 | 104.56 |
| FD002 | 1D CNN | >80 | 259 | 20671 | 93.0 [90.7, 95.1] | 83.72 | 94.10 |
| FD002 | LightGBM | 0-30 | 61 | 1087 | 96.6 [92.9, 99.4] | 67.25 | 71.27 |
| FD002 | LightGBM | 31-80 | 130 | 4753 | 86.4 [82.1, 90.2] | 67.25 | 87.61 |
| FD002 | LightGBM | >80 | 259 | 20671 | 94.4 [92.8, 95.8] | 67.25 | 74.41 |
| FD002 | LSTM | 0-30 | 61 | 1087 | 97.7 [95.4, 99.4] | 69.61 | 74.39 |
| FD002 | LSTM | 31-80 | 130 | 4753 | 86.4 [82.4, 90.0] | 69.61 | 92.69 |
| FD002 | LSTM | >80 | 259 | 20671 | 94.9 [93.4, 96.2] | 69.61 | 77.11 |
| FD003 | 1D CNN | 0-30 | 20 | 291 | 91.1 [79.4, 99.8] | 83.18 | 103.30 |
| FD003 | 1D CNN | 31-80 | 53 | 1789 | 84.2 [75.6, 92.1] | 83.18 | 114.35 |
| FD003 | 1D CNN | >80 | 100 | 11616 | 99.4 [98.8, 99.9] | 83.18 | 84.05 |
| FD003 | LightGBM | 0-30 | 20 | 291 | 99.5 [98.5, 100.0] | 60.43 | 60.50 |
| FD003 | LightGBM | 31-80 | 53 | 1789 | 81.2 [72.8, 88.9] | 60.43 | 96.27 |
| FD003 | LightGBM | >80 | 100 | 11616 | 97.4 [96.2, 98.5] | 60.43 | 62.51 |
| FD003 | LSTM | 0-30 | 20 | 291 | 100.0 [100.0, 100.0] | 47.48 | 47.48 |
| FD003 | LSTM | 31-80 | 53 | 1789 | 80.0 [72.0, 87.5] | 47.48 | 97.14 |
| FD003 | LSTM | >80 | 100 | 11616 | 93.2 [91.2, 95.1] | 47.48 | 56.03 |
| FD004 | 1D CNN | 0-30 | 53 | 864 | 73.2 [63.2, 82.5] | 76.54 | 135.84 |
| FD004 | 1D CNN | 31-80 | 113 | 4247 | 73.2 [66.7, 79.6] | 76.54 | 128.30 |
| FD004 | 1D CNN | >80 | 248 | 28981 | 94.7 [92.8, 96.4] | 76.54 | 87.11 |
| FD004 | LightGBM | 0-30 | 53 | 864 | 89.3 [82.7, 95.2] | 61.41 | 87.16 |
| FD004 | LightGBM | 31-80 | 113 | 4247 | 68.1 [61.3, 74.7] | 61.41 | 128.91 |
| FD004 | LightGBM | >80 | 248 | 28981 | 95.0 [93.6, 96.3] | 61.41 | 69.69 |
| FD004 | LSTM | 0-30 | 53 | 864 | 93.5 [88.5, 97.4] | 64.58 | 83.84 |
| FD004 | LSTM | 31-80 | 113 | 4247 | 75.4 [70.0, 80.6] | 64.58 | 112.93 |
| FD004 | LSTM | >80 | 248 | 28981 | 95.2 [94.0, 96.4] | 64.58 | 74.22 |

## S2.7 Explanatory decompositions

These descriptive decompositions were added during writing after some corrective
results were available. They use all relevant prespecified cells and introduce no
additional fitting or hypothesis test. Let a and b be predictions from raw- and
capped-trained models, and let r and c=min(r,125) be the two scoring truths. Then

$$[(a-r)^2-(b-r)^2]-[(a-c)^2-(b-c)^2]=-2(r-c)(a-b).$$

Consequently, only endpoints above 125 can change the squared-loss contrast when
switching scoring truth. The decomposition table gives the contributions on the
full-engine denominator, separately for endpoints at/below and above the cap.
The identity is independently checked against direct squared-error differences
at every paired endpoint. It is elementary loss accounting, not a new theorem or
an identification of how either predictor was learned.

The full rescoring matrix also separates training-target preference by scoring
population. Positive differences below favor capped training. Each row contains
the same 24 fitted pairs (two subsets × three partitions × two models × two
sensor sets). These are descriptive point-estimate ranges across fitted settings,
not confidence intervals or additional independent test populations.

| Scoring truth | Population | Raw minus capped RMSE range | Raw training lower | Capped training lower |
|---|---|---|---|---|
| capped125 | endpoint | +8.28 to +28.33 | 0 | 24 |
| capped125 | equal_engine_windows | +11.81 to +45.82 | 0 | 24 |
| capped125 | pooled_windows | +11.03 to +49.38 | 0 | 24 |
| raw | endpoint | +4.57 to +14.85 | 0 | 24 |
| raw | equal_engine_windows | -22.24 to -3.10 | 24 | 0 |
| raw | pooled_windows | -33.92 to -5.26 | 24 | 0 |

On raw truth, the endpoint preference for capped training reverses for all 24
pairs under both window populations. On FD004, raw training improves the
above-125 component of endpoint squared error for every fitted pair, while capped
training improves the at/below-125 component. Only 67 of the 248 official
endpoints are above the cap. The component table shows how the latter gains
outweigh the former losses in the endpoint population. Window populations place
more weight on early life and answer a different prediction question.

Loss-concentration tables report the share of each core model's NASA loss from
its largest ceil(0.05×N) endpoint losses, the largest single loss and the share
from positive errors. Every core fit is included. The per-engine loss table
permits exact reproduction of a metric-winner comparison. Stage-composition
tables show the weighted proportions of raw RUL 0–30, 31–80 and >80 for each
prediction population, using one reference prediction record per subset; input
identities and stage membership do not depend on the fitted model.

## S2.8 Historical results retained for traceability

The archived models use the original recipes and include numerically degenerate
LSTM configurations. Their complete common-truth results remain accessible.
Deterministic tree copies and seed-labelled historical contrasts are not used as
independent inferential repetitions in this revision. Historical quantile and
point-model intervals compare different trained pipelines; they do not isolate
a calibration-method effect. The original arbitrary experiment PASS/FAIL
labels and seed-level significance claims do not support the revised conclusions.

The original seven RMSE/NASA winner conflicts are listed below. This table belongs
to the earlier 80-context analysis, including its original target, inputs and
objectives, and must not be pooled with the 24 corrective primary contexts.
Original machine labels are retained here solely to identify archived records.

| Historical arm | Subset | Target/objective | Sensor setting | Seed | RMSE winner | NASA winner |
|---|---|---|---|---|---|---|
| kill_test | FD001 | linear_uncapped | all_21 | 23 | LightGBM | LSTM |
| kill_test | FD001 | linear_uncapped | all_21 | 53 | LightGBM | 1D CNN |
| kill_test | FD001 | linear_uncapped | common_14 | 11 | LightGBM | 1D CNN |
| kill_test | FD001 | linear_uncapped | common_14 | 37 | LightGBM | 1D CNN |
| kill_test | FD001 | linear_uncapped | common_14 | 71 | LightGBM | 1D CNN |
| kill_test | FD004 | linear_uncapped | common_14 | 11 | LightGBM | 1D CNN |
| unified_grid | FD001 | quantile | not_applicable | 37 | LightGBM | transformer |

## S2.9 File guide and reproduction

All paths in the following table are relative to `../analysis/`.

| File | Contents |
|---|---|
| `all_run_scores.csv` | 576 rows: 96 fits × two scoring truths × three prediction populations |
| `core_engine_bootstrap.csv` | 216 rows: core summaries, paired engine intervals and initialization ranges |
| `paired_factor_contrasts.csv` | 96 rows: all target/input contrasts, three splits, two common truths |
| `metric_winner_contexts.csv` | 144 rows: 24 contexts × two truths × three populations; tied winning sets retained |
| `training_quality.csv` | All 96 fits, selected epochs/iterations, runtime, parameter count and cap flags |
| `repaired_vs_historical_baselines.csv` | Matched split-42 neural and tree baseline comparisons; conditional paired engine intervals |
| `uq_stage_engine_bootstrap.csv` | 108 stage summaries with coverage/width/interval score and contributing engine counts |
| `uq_overall_engine_bootstrap.csv` | 36 overall equal-engine interval summaries reconstructed from all stages |
| `uq_per_engine_stage.parquet` | Initialization-specific per-engine stage summaries and calibration half-widths |
| `historical_common_truth_all_runs.csv` | All 120 archived prediction bundles rescored on both common truths; original fitting limitations retained |
| `historical_common_truth_descriptive_contrasts.csv` | Complete historical target contrasts; descriptive archive only |
| `historical_metric_winner_conflicts.csv` | The original seven metric-winner conflicts, shown separately from corrective fits |
| `target_contrast_decomposition.csv` | Exact squared-loss attribution to endpoints above and below the cap |
| `target_population_contrasts.csv` | 144 descriptive target contrasts: 24 fitted pairs × two truths × three populations |
| `target_population_summary.csv` | Direction counts and RMSE-difference ranges for all six truth/population choices |
| `endpoint_loss_concentration.csv` | All 60 core fits: largest 5% loss share and signed-error contributions |
| `core_endpoint_loss_contributions.parquet` | Per-endpoint squared and NASA losses for all core fits |
| `population_stage_composition.csv` | Life-stage weights for each prediction population; four subsets |

Each `../runs/<run_id>/` and `../diagnostics/<run_id>/` directory contains the
metadata, partition, scaler, validation curve, selected checkpoint and validation,
calibration and test predictions. Histories and SHA-256 values make numerical
claims traceable. The original frozen specification and source snapshot are
retained. Historical archives are identified by DOIs 10.5281/zenodo.21915989 and
10.5281/zenodo.21915990; this revision does not claim the new artifacts are already
present in those public versions.

Acquire the official NASA C-MAPSS files separately and place them in
`rul-eval-audit/data/interim/cmapss/`, verifying the hashes above. Preserve the
package layout, with the revision scripts in
`post_rejection_20260926/transfer_revision/`. From that directory run:

```text
python -X utf8 train_repair.py --arm diagnostic
python -X utf8 train_repair.py --arm grid --model neural
python -X utf8 train_repair.py --arm grid --model lightgbm
python -X utf8 analyze_repair.py
python -X utf8 explain_results.py
python -X utf8 make_figures.py
python -X utf8 build_supplements.py
```

The two grid commands may run as separate GPU/CPU jobs. Completed records are
reused only if their identities and specification hash match; an incomplete
record requires inspection before retry. To reproduce existing tables and figures,
the provided predictions suffice and training need not be repeated. The full
matched historical-baseline comparison additionally uses the original archived
split-42 prediction files included in the reproducibility package. Environment
differences can change floating-point training results even with fixed seeds.
