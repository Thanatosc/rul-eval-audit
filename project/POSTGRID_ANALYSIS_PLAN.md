# Post-grid Mainline Statistical Analysis Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan -> validate
- Origin Date: 2026-08-12
- Verification Status: FROZEN BEFORE NEW INFERENTIAL OUTPUTS
- Version Label: `postgrid_analysis_v1`
- Upstream Dependencies: `kill_v1_result_v1`, `unified_v1_result_v1`

## 1. Timing and Scope

This plan was frozen after the registered Kill Test decision and the descriptive
Unified Grid summaries were available, but before any inferential post-grid
tests, confidence intervals, rank-agreement coefficients, or post-grid figures
were computed. It is therefore a prospective analysis specification for the
new analyses below, not an outcome-blind preregistration of the completed model
training.

The analysis is read-only with respect to the registered run artifacts. It must
not modify `protocols/unified_v1.md`, `configs/kill_test.yaml`,
`configs/unified_grid.yaml`, or `src/rul_audit/**/*.py`. The two authoritative
seed-level inputs are:

| Input | Rows | SHA-256 |
|---|---:|---|
| `results/KILL_TEST_RUN_METRICS.parquet` | 120 | `60e1b19cfa2f0be10640d4431ce400dbd491bb019059d6662a7785822f000a8a` |
| `results/UNIFIED_GRID_RUN_METRICS.parquet` | 160 | `48e99c2fc935f17b1b92295cceec549d149e21150567ff534c2087ce19130976` |

The completed Kill Test used two subsets, three models, two RUL labels, two
sensor sets, and five seeds. The Unified Grid used four subsets, four models,
two output objectives, and the same five seeds. No failed or missing registered
cell is present. If these hashes, row counts, uniqueness constraints, or closed
factor levels differ, the analysis must fail rather than impute or silently
drop a cell.

## 2. Analysis Unit and Estimands

The random seed is the paired computational replicate. Windows from the same
engine and engines within a completed run are not treated as independent
replicates. C-MAPSS subsets are reported separately because their operating
conditions and fault modes differ; they are not pooled into a single
super-population estimate.

Primary outcome: official-test endpoint RMSE. NASA Score is a secondary metric
used to audit ranking agreement and disagreement. Lower values are better for
both metrics.

The post-grid analyses estimate:

1. Paired RMSE changes under each registered Kill Test label or sensor
   transition.
2. Within-subset, within-objective agreement of model rankings across five
   seeds in the Unified Grid.
3. Agreement between RMSE-based and NASA-Score-based model rankings.
4. Descriptive point-versus-quantile differences, without treating the two
   objectives as a fair performance contest.

## 3. Kill Test Analysis

For each of the 24 registered contrasts, compute the five paired seed-level
differences in the registered direction:

- label: `linear_uncapped - piecewise_125`, holding dataset, model, and sensor
  set fixed;
- sensor: `common_14 - all_21`, holding dataset, model, and RUL label fixed.

Report the mean and median RMSE difference, seed SD, number of positive/zero/
negative differences, and the matched-pairs rank-biserial correlation. The
rank-biserial effect is the signed-rank sum divided by the total non-zero rank
sum; it is undefined only when every paired difference is zero.

A two-sided exact sign-flip permutation test on the mean difference enumerates
all `2^5 = 32` sign assignments. The 24 p-values form one family and receive
Holm family-wise error correction at `alpha = 0.05`. Because five seeds imply a
minimum attainable non-zero two-sided p-value of 0.0625 in a no-tie extreme
case, statistical significance is not expected or required for interpretation.
Raw effects and their uncertainty remain the main outputs.

For descriptive uncertainty, use a paired seed bootstrap with 20,000 resamples,
a fixed master seed of 20260812, and percentile 95% intervals for the mean RMSE
difference. With only five seeds, these intervals describe computational-seed
dispersion and are not population confidence claims.

The frozen registered ranking-reversal definition remains authoritative. In
addition, aggregate its events into a model-pair matrix by transition type;
do not redefine the Kill Test PASS/FAIL decision.

## 4. Unified Grid Ranking Analysis

For each of the eight `subset x output_mode` panels, rank the four models within
each seed by RMSE. Report mean rank, rank SD, first-place count, RMSE mean, SD,
median, range, coefficient of variation, and a 20,000-resample seed-bootstrap
percentile 95% interval for mean RMSE.

Ranking agreement is Kendall's coefficient of concordance:

`W = Friedman Q / (n_seeds * (n_models - 1))`.

The Friedman null distribution is computed exactly by enumerating the
`(4!)^5` within-seed label permutations through dynamic programming. The eight
omnibus p-values form one family and receive Holm correction. No model-pair
post-hoc significance tests are planned. Pairwise stability is instead
described by wins, losses, ties, and the fraction of the ten seed pairs in
which a model pair has opposite order.

For point-to-quantile protocol flips, compare each model pair within the same
subset and seed. Count an event only when its strict RMSE order changes sign.
Ties are reported separately and are not counted as flips.

## 5. RMSE versus NASA Score Agreement

Within each complete context, rank models once by RMSE and once by NASA Score:

- Kill Test context: dataset, RUL label, sensor set, seed (three models);
- Unified context: subset, output mode, seed (four models).

For every context report Spearman's rho, Kendall's tau-b, the best model under
each metric, whether the winner differs, and the number of discordant model
pairs. These are descriptive diagnostics; no p-values are used because the
question is metric agreement within the complete benchmark rather than a
sampled association test. Every winner conflict is retained in a contradiction
table for case-level inspection.

## 6. Point versus Quantile Boundary

For each of 16 `subset x model` cells, summarize the five paired differences
`quantile - point` for RMSE and NASA Score and count the seeds in which the
quantile q50 value is lower. Also report the global count over all 80 pairs.
No hypothesis test, corrected p-value, or superiority language is permitted:
the point models optimize MSE while the quantile models optimize pinball loss.

## 7. Missingness, Ties, Multiplicity, and Interpretation

- Any absent, duplicate, non-finite, or out-of-register seed-level cell stops
  the analysis. There is no imputation and no automatic rerun.
- Average ranks are used for metric ties. A strict protocol flip requires a
  non-zero product of pairwise differences; ties do not become flips.
- Holm correction is applied separately to the 24 Kill Test contrasts and the
  eight Unified Grid omnibus tests. Descriptive agreement analyses do not enter
  either family.
- Corrected `p < 0.05` is reported as evidence against the relevant null, not as
  practical importance. Raw RMSE changes, rank-biserial effects, W, instability,
  and contradiction cases carry the substantive interpretation.
- Results support claims about these frozen datasets, models, budgets, seeds,
  and protocols. They do not establish field-wide causal effects, SOTA status,
  or general superiority of an architecture.

## 8. Frozen Outputs

The implementation is `scripts/analyze_postgrid.py`. It writes only to
`results/postgrid/` and `paper/figures/`:

- `POSTGRID_SUMMARY.json`
- `KILL_TEST_CONTRAST_INFERENCE.csv`
- `KILL_TEST_RANKING_FLIP_MATRIX.csv`
- `UNIFIED_MODEL_STABILITY.csv`
- `UNIFIED_KENDALL_W.csv`
- `UNIFIED_PAIRWISE_STABILITY.csv`
- `UNIFIED_POINT_QUANTILE_FLIPS.csv`
- `METRIC_RANK_CONSISTENCY.csv`
- `METRIC_WINNER_CONFLICTS.csv`
- `POINT_QUANTILE_DESCRIPTIVE.csv`
- `POSTGRID_STATISTICAL_VALIDATION.md`
- `FIG_KILL_TEST_EFFECTS.png`
- `FIG_UNIFIED_RMSE_STABILITY.png`
- `FIG_RANK_AGREEMENT.png`

Completion requires deterministic regeneration, closed output shapes, Holm
monotonicity, exact-test validity, all 11 experiment-agent fallacy checks, the
full project test suite, and Ruff.
