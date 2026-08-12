# UQ Reference Arm Statistical Validation

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: `uq_ref_v1_validation_v1`
- Source: `uq_ref_v1`

## Validation Report

- **Overall Confidence:** CAUTION
- **Registered post-processing cells:** 280
- **Completed/validated:** 280/280
- **Failed/missing:** 0
- **Nominal coverage:** 90%
- **Inference unit:** official test engine
- **Bootstrap:** paired engine-cluster percentile, 20,000 resamples; five registered
  seeds averaged within engine before resampling
- **Validity label:** empirical reference, not a strict window-level certificate

`CAUTION` is assigned because the descriptive protocol effects are reproducible and
the artifacts close, but overlapping calibration windows are not exchangeable
independent examples, the bootstrap conditions on the realized calibration scores,
and residual CP versus CQR also compares separately trained point and quantile
models. No p-value or finite-sample coverage guarantee is inferred from the bootstrap
intervals.

## Statistical Findings

| Finding | Registered comparison | Result | Interpretation | Confidence |
|---|---|---:|---|---|
| RUL-label coverage shift | 12 paired label contrasts | mean -7.14 pp; range -14.08 to -3.71 pp; 12/12 >=3 pp in magnitude; 12/12 engine-bootstrap intervals exclude 0 | Changing from piecewise-125 to linear-uncapped consistently reduced empirical coverage and increased mean width by 75.04 RUL units on average | CAUTION |
| Sensor-set coverage shift | 12 paired sensor contrasts | mean +0.11 pp; range -0.66 to +1.68 pp; 0/12 >=3 pp; 7/12 bootstrap intervals exclude 0 | Several small shifts are precisely estimated but none reaches the preregistered practical reference; statistical precision is not practical importance | CAUTION |
| Unified residual split CP | 80 cells | mean engine-balanced coverage 94.49%; range 89.92%-99.50%; mean width 83.26 | Usually conservative on average; pooling hides subset/model variation | CAUTION |
| Unified CQR | 80 cells | mean engine-balanced coverage 90.11%; range 75.41%-96.06%; mean width 78.63 | Aggregate mean is close to nominal, but substantial model-specific undercoverage remains | CAUTION |
| CQR calibration versus base q10-q90 | 16 subset-model panels | closer to 90% in 12/16 | CQR materially repaired raw quantile undercoverage for CNN, LSTM, and Transformer panels; it moved all four LightGBM panels farther from nominal by adding conservative coverage | CAUTION |
| Residual CP versus CQR bundles | 16 subset-model panels | CQR closer to nominal in 9/16 | Descriptive comparison only: point and quantile models have different fitted objectives, so this is not a pure causal method effect | CAUTION |

## Heterogeneity that must remain visible

The overall CQR mean of 90.11% is not representative of every backbone. Mean CQR
coverage by model was 93.85% for 1D-CNN, 93.10% for LightGBM, 91.31% for
Transformer, and 82.20% for LSTM. LSTM CQR cell coverage ranged from 75.41% to
88.65%, despite the aggregate CQR mean being close to nominal. Conversely, residual
split CP averaged 97.89% for LSTM, with wider intervals. Any manuscript statement
must report this coverage-width and model-heterogeneity structure rather than only
the pooled mean.

## Warnings

| Type | Detail | Affected results |
|---|---|---|
| Exchangeability boundary | Calibration pools overlapping windows within engine; unit-disjoint allocation prevents leakage but does not create independent window-level observations | All conformal coverage results |
| Calibration/test mismatch | Calibration engines are complete run-to-failure trajectories; official test engines are truncated | All empirical coverage estimates, especially endpoint interpretation |
| Conditional bootstrap | Engine bootstrap resamples test engines while holding fitted models and calibration quantiles fixed | All reported 95% intervals |
| Method-bundle confounding | Residual CP consumes point models while CQR consumes separately trained quantile models | Residual-CP/CQR comparison |
| Multiple comparisons | 24 protocol, 16 method-bundle, and 16 calibration-effect comparisons are reported; no familywise binary significance claim is made | All contrast tables |
| Width degeneracy across engines | For symmetric residual CP and fixed-qhat CQR correction, paired width shifts can be algebraically fixed within a run; zero-width bootstrap intervals are not evidence of population certainty | Some mean-width contrast intervals |
| Practical versus statistical difference | Seven sensor contrasts have bootstrap intervals excluding zero, but all 12 are below the registered 3 pp practical reference | Sensor-set contrasts |
| No deployment guarantee | Empirical C-MAPSS coverage is not a safety, transfer, conditional-coverage, or maintenance-decision certificate | Entire work package |

## Fallacy Scan

- **Coverage:** 11/11 fallacy types checked

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | CAUTION | Pooled CQR mean is close to 90%, while LSTM panels are all below nominal; model/subset strata are therefore reported explicitly. |
| Ecological fallacy | CAUTION | Cell- or model-level means cannot be converted into claims for individual engines or windows. Engine-level summaries remain the inference unit. |
| Berkson's paradox | NOTE | The experiment covers a deliberately selected C-MAPSS protocol/model grid, not a random sample of all prognostics systems. Field-wide prevalence is not inferred. |
| Collider bias | NOTE | No adjusted regression or post-outcome covariate control is used; no collider path was introduced in the registered contrasts. |
| Base-rate neglect | NOTE | Coverage is a direct event rate rather than diagnostic PPV/NPV; engine and window denominators are reported. |
| Regression to the mean | NOTE | No arm was selected because of an extreme first measurement; all frozen cells and contrasts are retained. |
| Survivorship bias | NOTE | There are 0 failed/missing UQ cells; no completed-only subset replaces the registered 280-cell denominator. |
| Look-elsewhere effect | CAUTION | All 56 registered aggregate comparisons are retained, and the 3 pp reference is descriptive; no selective significance narrative is allowed. |
| Garden of forking paths | NOTE | Configuration, formulas, estimands, missing policy, schema, and cell register were hash-frozen before the first test interval. Amendment `uq_ref_v1_a1` occurred with 280/280 cells pending and no UQ results seen. |
| Correlation != causation | CAUTION | Protocol contrasts are controlled computational comparisons, but conclusions remain implementation-specific; method-bundle differences are not attributed solely to conformalization. |
| Reverse causality | NOTE | Temporal causal direction is not an estimand in this deterministic benchmark. |

## Reproducibility

- **Method:** deterministic source-to-output recomputation plus closed-register audit
- **Verdict:** REPRODUCIBLE

Every saved calibration-score and test-interval table was regenerated from its
frozen source Parquet tables during cell validation. The register contains 280 unique
completed cells, each with source hashes, registration/schema/implementation hashes,
metadata, calibration scores, intervals, and summaries. Aggregate outputs are
generated by `scripts/uq_reference_arm.py` and carry deterministic SHA-256 hashes in
`UQ_REFERENCE_SUMMARY.json`.

