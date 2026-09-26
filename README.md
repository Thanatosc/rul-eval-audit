# RUL Evaluation Audit

Software version 0.2.0 supplies the corrective code and aggregate evidence for
**Training adequacy and evaluation populations in remaining useful life benchmarks**.
The companion generated-artifact dataset version is 1.1.0.

## Version identifiers

- Software v0.2.0: <https://doi.org/10.5281/zenodo.22980668>
- Generated-artifact dataset v1.1.0: <https://doi.org/10.5281/zenodo.22980776>
- GitHub release: <https://github.com/Thanatosc/rul-eval-audit/releases/tag/v0.2.0>

Use these version-specific identifiers for the corrective study. Earlier
version identifiers below continue to identify the historical experiments.

## Current evidence

- 96 primary corrective fits: 60 core fits and 36 target/input factorial fits.
- Two controlled numerical-scale diagnostics, with the original weak FD002 LSTM
  predictions reproduced exactly before changing the training scale.
- Nine additional budget checks covering every primary final-cap flag. Their
  original prediction prefixes were reproduced exactly; all extended fits stopped
  by validation patience before the threefold cap.
- Sensor-noise/bias checks on six fixed reference predictors: 72 perturbed
  test-prediction tables, error-displacement accounting and 600 inference timings.
- Engine-paired error assessment under two common scoring truths and three
  prediction populations, with separate partition/initialization sensitivity.
- Literature coding for the retained purposive 19-paper model/protocol corpus,
  plus six contextual anchors, and empirical interval summaries by life stage.

The budget extension preserves all 24 endpoint-to-window training-target
preferences on raw truth and the RMSE/NASA winning sets in all 144 core contexts.
The separate sensor check changes the FD004 reference winner under the prescribed
range-scaled perturbations; it does not establish field-noise robustness.
This is an exploratory assessment of one simulated benchmark. It does not
establish general convergence, a new predictive architecture, deployment coverage
or maintenance-cost savings. Read Supplements S1, S2, S4 and S5 for the design and
all limitations.

## Reproduce the corrective analysis

`revisions/20260926/` preserves the scientific scripts, specifications, complete
aggregate tables, figures and supplements. Historical root-level code and tables
are retained for provenance. Run the frozen corrective scripts from their
original two-directory layout, created from the companion dataset:

```text
python scripts/prepare_corrective_workspace.py --dataset rul-eval-audit-corrective-results-v1.1.0.zip --output reproduction-workspace
cd reproduction-workspace/post_rejection_20260926/transfer_revision
python -X utf8 analyze_repair.py
python -X utf8 explain_results.py
python -X utf8 budget_sensitivity/analyze_budget_checks.py
python -X utf8 budget_sensitivity/build_budget_supplement.py
python -X utf8 review_followups/analyze_review_checks.py
python -X utf8 review_followups/explain_sensor_sensitivity.py
python -X utf8 review_followups/build_review_supplement.py
```

Use the versions in `revisions/20260926/ENVIRONMENT_SNAPSHOT.json`. The
materializer checks every payload digest and refuses to overwrite differing
files. It needs only the local dataset archive. These analysis commands consume
supplied predictions and perform no model training. Model reload verification
and fresh fitting additionally require the official NASA data and recorded
training environment; see S2, S4 and S5. Raw NASA sensor data are not redistributed.

## What changed from the historical release

Historical software v0.1.2 is archived at
<https://doi.org/10.5281/zenodo.21915989> and historical data v1.0.2 at
<https://doi.org/10.5281/zenodo.21915990>. Those version-specific DOIs refer to the
earlier experiments, not these corrective fits. They remain available for audit.

The corrective study supersedes architectural interpretations of the archived
nearly constant baseline, improvement claims based on comparing different native
truths, and inference that treats seed-labelled deterministic-tree repetitions as
independent fits. The historical outputs are retained as historical evidence;
the current conclusions use common truths, repaired training and engine-paired
analysis. See `CHANGELOG.md` and `docs/CORRECTIVE_RELEASE.md`.

## Data and licensing

Software is MIT licensed. The separately deposited generated dataset is licensed
CC BY 4.0 for the author's copyrightable contributions; this does not claim or
relicense NASA-origin facts. It contains 107 fitted-run records, 321 fit prediction
tables, 72 fixed-model sensor-perturbation prediction tables, and 20 historical test-prediction inputs required by the
corrective analyses. The complete earlier 120/160/280-cell archives remain in
dataset v1.0.2; they are not duplicated in their entirety in v1.1.0.

Literature full texts, raw NASA sensor files, manuscripts, submission letters,
credentials and internal author guidance are excluded. Open code and derived
data do not determine whether the journal article is published open access.
