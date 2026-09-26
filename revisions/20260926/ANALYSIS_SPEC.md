Transfer revision: analysis and corrective-training specification
Recorded 2026-09-26 before inspection of the new corrective results.

Purpose: repair the existing evaluation audit. These are post-rejection exploratory
analyses. Historical official test data have already been examined; no new split
can retrospectively make this a pristine confirmatory study.

Primary estimand: error of retained predictions against min(raw RUL,125), with one
last available prediction per official test engine. Raw RUL is a separately named
secondary estimand. Training-target contrasts are compared on each common truth,
never interpreted from a difference between their separate native-truth RMSEs.
Feature changes are input ablations. Metric/denominator comparisons retain exactly
the same fitted model and predictions.

Corrective models: original 64-hidden-unit LSTM and two-layer 1D CNN architectures,
plus flattened-window LightGBM. No new predictor is proposed. Transformer and
quantile models in the historical grid remain archival sensitivity material; their
historical inadequacy will not be hidden by presenting all runs as equally tuned.

Data: original NASA C-MAPSS FD001..FD004. Window length 30, stride 1; min-max scaler
fit only on training engines, then applied unchanged. Short official test engines
receive a left-padded endpoint. Engine partitions use existing 70/15/15 algorithm,
split seeds 42,137,271. Validation selects models; calibration is never used for
training or hyperparameters. Partitions share the finite source dataset and are
not independent replications or new devices.

Core grid (60 runs): four subsets x three engine splits x capped125/common14 x
{LSTM initialization 11 and 23, CNN initialization 11 and 23, one deterministic
LightGBM}. Split 42 is the reference context; other splits assess sensitivity.

Factor follow-up (36 runs): FD001/FD004 x three engine splits x
{raw/common14, raw/all21, capped125/all21} x {LSTM init11, deterministic LightGBM}.
Together with the corresponding core cells this completes the historical two-label,
two-sensor design for two representative models under the repaired training recipe.
No claim of initialization robustness for every factor contrast is intended.

Neural recipe: train and validate numerical targets y/125, invert predictions to
cycles; same constant for raw and capped targets. Adam lr=.001, weight_decay=1e-5,
batch=256. Reduce LR by .5 after 5 unimproved validation epochs, min_lr=1e-5.
Maximum 150 epochs, minimum 20 before stopping, patience 15; retain best validation
RMSE checkpoint. If best epoch lies within the final 15 epochs at the cap and
validation improved by at least 0.1 cycles across the final 15 epochs, continue the
same run to at most 300 epochs and record the extension. Save every epoch's train
loss, validation RMSE, LR, elapsed time, and selected epoch. Report any run that
still hits the cap with unresolved improvement; do not call it converged.

LightGBM recipe: existing flattened 30-cycle inputs, lr=.05, num_leaves=31,
subsample=colsample=1, deterministic CPU, one thread. Up to 1500 trees, select via
validation RMSE with early-stopping patience 50. Targets remain in cycles. Save
validation curve and best iteration. Deterministic repetitions are stored once.

Small optimization diagnostic, separate from the grid: FD002, split42, init11,
LSTM, capped125/common14. Compare original y-in-cycles training with y/125 under
the same original 50-epoch, patience8, fixed-LR recipe; save both curves. This is
an optimization-recipe comparison, not proof of a unique saturation mechanism.
The changed numerical scale also changes the relative effect of regularization.

Pure rescoring: common raw/capped truths x endpoint/equal-engine-windows/pooled-
windows x RMSE/asymmetric NASA score. Report per-run results and model orders;
do not use arbitrary PASS/FAIL thresholds or rescore competing methods on different
targets. Expose all factor contrasts, not only maxima and reversal counts.

Uncertainty: paired engine bootstrap conditional on the fitted models; use 5000
replicates and fixed seed 20260926. Average squared-error/loss contributions over
neural initialization within engine before resampling for core summaries. Present
split-to-split ranges and initialization spreads separately. Marginal 95% intervals
are descriptive, not simultaneous hypothesis tests. No seed sign-flip p-values,
no multiplication of deterministic tree records, no window-level effective n.

Empirical UQ supplement: symmetric residual calibration at nominal 90% for the
new core point models, calibration scores from calibration engines only. This
reuses a standard method and supplies no new conformal guarantee. Target is capped
RUL; stage bins use raw RUL [0,30], (30,80], >80. Report coverage, width, interval
score, contributing engines/windows, and engine-cluster uncertainty. Historical
CQR/CP comparisons, if retained, are explicitly different model/loss bundles.

Output: all cells, diagnostics and failures retained; no manual removal of hard
subsets or seeds. Report runtime and parameter count as computational context.
No t-SNE or noise study is added solely for decoration; those do not resolve the
identified estimand or optimization problems.
