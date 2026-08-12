# Post-grid Statistical Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: `postgrid_analysis_v1`
- Frozen Plan SHA-256: `8d15454a87ccef36135225c83cc145d439cfa26806cb13418a4903097b76d823`

## Validation Report

- **Source:** 120 registered Kill Test cells and 160 validated Unified Grid cells
- **Overall Confidence:** CAUTION
- **Reason:** all registered cells and exact small-sample tests are reproducible, but only five computational seeds are available and this inferential plan was frozen after descriptive outcomes were known.

### Statistical Findings

| Finding | Method | Result | Interpretation boundary |
|---|---|---|---|
| Largest Kill Test contrast | Five paired seeds; exact sign-flip; Holm family of 24 | `FD004`, `lightgbm`, `common_14`: mean difference `20.5323` RMSE; rank-biserial `1.000`; raw p `0.0625`; Holm p `1.0000` | Effect magnitude is descriptive for the frozen design; five-seed exact p-values cannot fall below 0.0625. |
| Kill Test multiplicity | Holm FWER | `0/24` contrasts rejected at 0.05 | Lack of rejection is not evidence of negligible protocol effects. |
| Unified rank concordance range | Exact Friedman permutation; Holm family of 8 | W `0.712` (`FD002`, `quantile`) to `1.000` (`FD003`, `quantile`); `8/8` corrected rejections | W describes stability of model order across the five seeds, not model quality. |
| RMSE/NASA winner conflicts | Complete within-context rankings | `7/80` contexts selected different best models | NASA Score's asymmetric error penalty can change architecture selection. |
| Point/quantile q50 RMSE | Descriptive paired counts only | `13/80` quantile cells had lower RMSE | No superiority test: objectives differ (pinball versus MSE). |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Small replicate count | Five seeds yield coarse exact randomization p-values and unstable percentile intervals. | All inferential outputs |
| Analysis timing | The plan predates new inference but follows the registered PASS and descriptive means. | Confirmatory language |
| Objective mismatch | Point and quantile models optimize different losses. | Point/quantile comparisons |
| Nonlinear metric | NASA Score is asymmetric and highly sensitive to late-prediction outliers. | Metric-rank conflicts |
| Scope | Subsets, budgets, seeds, and architectures are fixed rather than sampled from a target population. | External generalization |

### Fallacy Scan

- **Coverage:** 11/11 statistical fallacy types checked

| Fallacy | Severity | Finding and handling |
|---|---|---|
| Simpson's paradox | CAUTION | No pooled subset effect is used; all primary effects and rankings remain subset-specific. |
| Ecological fallacy | NOTE | Run-level endpoint summaries are not used to infer window-level or individual-engine mechanisms. |
| Berkson's paradox | NOTE | C-MAPSS is a benchmark-selected corpus; conclusions are restricted to the frozen benchmark. |
| Collider bias | NOTE | No covariate-adjusted causal regression is fitted. |
| Base-rate neglect | NOTE | No diagnostic sensitivity, specificity, PPV, or NPV claim is made. |
| Regression to the mean | NOTE | No group is selected for analysis because of an extreme observed seed result. |
| Survivorship bias | NOTE | The full 120+160 registered cells completed; failed cells were not excluded. |
| Look-elsewhere effect | CAUTION | All 24 and eight planned test families are reported with Holm correction; descriptive diagnostics are labeled. |
| Garden of forking paths | CAUTION | The post-grid plan and input hashes are frozen and disclosed, but model outcomes were already descriptively known. |
| Correlation does not imply causation | CAUTION | Protocol transitions are controlled computational contrasts, but claims remain design-specific rather than field-wide causal claims. |
| Reverse causality | NOTE | Temporal or observational causal direction is not analyzed. |

### Reproducibility

- **Method:** deterministic regeneration from frozen seed-level Parquet inputs
- **Verdict:** REPRODUCIBLE
- **Input hash checks:** 4/4 passed
- **Registered cell closure:** 120/120 Kill Test and 160/160 Unified Grid
- **Automatic imputation or rerun:** none
