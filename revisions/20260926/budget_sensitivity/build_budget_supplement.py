"""Write S4 directly from the completed and independently verified budget arm."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ANALYSIS=HERE/'analysis'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join(
        '| '+' | '.join(str(v) for v in row)+' |' for row in rows)


def main():
    summary=json.loads((ANALYSIS/'BUDGET_SUMMARY.json').read_text())
    verify=json.loads((HERE/'BUDGET_VERIFICATION.json').read_text())
    assert summary['status']=='complete' and verify['status']=='passed'
    training=pd.read_csv(ANALYSIS/'training_budget_comparison.csv')
    paired=pd.read_csv(ANALYSIS/'paired_endpoint_budget_changes.csv')
    populations=pd.read_csv(ANALYSIS/'sensitivity_target_population_summary.csv')
    aliases={row.run_id:f'B{i}' for i,row in enumerate(training.itertuples(),1)}
    settings=[]; budgets=[]; endpoints=[]
    for row in training.itertuples():
        bid=aliases[row.run_id]
        settings.append([bid,row.subset,row.split_seed,'LSTM' if row.model=='lstm' else 'LightGBM',
            'raw' if row.target=='linear_uncapped' else 'capped125',21 if row.sensors=='all_21' else 14])
        budgets.append([bid,f'{row.old_selected}/{row.old_evaluated}',f'{row.new_selected}/{row.new_evaluated}',
            f'{row.old_val_rmse:.4f}',f'{row.new_val_rmse:.4f}',f'{row.validation_difference:+.4f}',
            'yes' if row.remaining_cap_flag else 'no'])
    for row in paired.itertuples():
        endpoints.append([aliases[row.run_id],row.truth,f'{row.original_rmse:.4f}',f'{row.rmse:.4f}',
            f'{row.rmse_difference:+.4f} [{row.rmse_ci_lower:+.4f}, {row.rmse_ci_upper:+.4f}]',
            f'{row.nasa_mean_difference:+.4f} [{row.nasa_mean_ci_lower:+.4f}, {row.nasa_mean_ci_upper:+.4f}]'])
    p_rows=[[r.truth,r.population,f'{r.difference_min:+.3f} to {r.difference_max:+.3f}',
             r.capped_trained_lower_rmse,r.raw_trained_lower_rmse] for r in populations.itertuples()]
    cap=paired[paired.truth=='capped125'].rmse_difference
    raw=paired[paired.truth=='raw'].rmse_difference
    names={
        'training_budget_comparison.csv':'Nine controls, original/extended selected iterations, validation changes and remaining flags.',
        'new_run_score_changes.csv':'54 scores and original-versus-extended differences: nine fits, two truths, three populations.',
        'paired_endpoint_budget_changes.csv':'18 endpoint comparisons with paired-engine RMSE and mean NASA-loss intervals.',
        'sensitivity_all_run_scores.csv':'576 scores from the primary grid with only the nine flagged fits substituted.',
        'sensitivity_metric_winner_contexts.csv':'144 contexts with RMSE and NASA winning sets.',
        'metric_winner_changes.csv':'Original and substituted winner sets, including unchanged contexts.',
        'sensitivity_target_population_contrasts.csv':'144 target contrasts from 24 fitted pairs on two truths and three populations.',
        'sensitivity_target_population_summary.csv':'Six summaries of target-comparison directions and ranges.',
        'target_preference_changes.csv':'All 144 original and substituted training-target contrasts.',
        'sensitivity_core_scores.csv':'216 core summaries; endpoint intervals included.',
        'sensitivity_paired_factor_contrasts.csv':'96 common-truth endpoint target/input contrasts and paired-engine intervals.',
        'sensitivity_uq_per_engine_stage.parquet':'Recalibrated coverage, width and interval-score contributions by engine and stage.',
        'sensitivity_uq_stage_summary.csv':'108 stage summaries under the substituted core fits.'}
    document=f'''# Supplement S4. Bounded training-budget sensitivity

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

{table(['ID','Subset','Partition seed','Model','Training target','Sensors'],settings)}

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
unchanged. The {summary['remaining_cap_flags']} remaining cap flags refer to this
extended arm; the nine original flags remain in the preserved primary records.
Validation stopping is a practical selection rule and does not establish global
convergence or optimal hyperparameters.

{table(['ID','Original selected/evaluated','Extended selected/evaluated','Original validation RMSE','Extended validation RMSE','Change','Cap flag'],budgets)}

## S4.3 Paired changes in official-endpoint errors

Differences below are extended minus original; a negative value favors the
extended fit. Each interval uses 5,000 paired bootstrap samples of complete
official test engines, with generator seed 20260926. The intervals condition on
the two validation-selected fits and are marginal, descriptive intervals. They
do not constitute equivalence tests or simultaneous family-wise guarantees.

The endpoint RMSE changes range from {cap.min():+.4f} to {cap.max():+.4f} cycles
on capped truth and {raw.min():+.4f} to {raw.max():+.4f} cycles on raw truth.
Validation gains do not require test-error gains: several endpoint errors rose
slightly. B2's zero-width difference interval reflects identical predictions.

{table(['ID','Scoring truth','Original RMSE','Extended RMSE','RMSE change [95% interval]','Mean NASA change [95% interval]'],endpoints)}

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

{table(['Scoring truth','Population','Difference range (cycles)','Capped training lower','Raw training lower'],p_rows)}

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
prediction prefixes was {summary['maximum_prefix_prediction_error']:.1f}; the
largest checked training/validation-curve difference was
{summary['maximum_prefix_curve_error']:.3g}. Source, split, scaler and input-data
hashes were checked by the runner.

Independent verification reloaded all {verify['new_models_reloaded']} extended
models and reproduced {verify['new_prediction_rows']:,} rows in 27 prediction
tables. It separately recomputed 54 truth/population scores and 36 paired
endpoint intervals. Maximum reload error was {verify['maximum_reload_error']:.3g};
the maximum RMSE absolute/NASA relative score discrepancy was
{verify['maximum_score_error']:.3g}. The machine-readable check is
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

{table(['File','Contents'],[[f'`{name}`',text] for name,text in names.items()])}
'''
    (BASE/'supplements/SUPPLEMENT_S4.md').write_text(document,encoding='utf-8')
    note={
        'material_passport':{'source':'Completed original 96-fit grid and separately frozen nine-fit budget extension',
            'analysis_scope':'Exploratory benchmark sensitivity, no new external sample',
            'integrity_verification':'BUDGET_VERIFICATION.json'},
        'verification_status':'VERIFIED','fallacy_types_checked':11,
        'checks':[
            {'type':'Simpson/aggregation reversal','assessment':'Endpoint/window and life-stage weighting shown separately; reversal is the measured finding.'},
            {'type':'Ecological inference','assessment':'No fleet or individual maintenance benefit inferred from aggregate benchmark scores.'},
            {'type':'Selection/Berkson','assessment':'All nine validation-cap flags included; inference limited to these configurations.'},
            {'type':'Collider adjustment','assessment':'No causal covariate adjustment; fixed budget controls with descriptive conditional inference.'},
            {'type':'Base rates','assessment':'Prediction populations and stage engine counts specified; no diagnostic PPV claim.'},
            {'type':'Regression to mean','assessment':'Controls selected by validation-cap flag, not worst test result; no universal improvement claim.'},
            {'type':'Survivorship','assessment':'Nine of nine planned fits completed and retained, including unchanged or worse test scores.'},
            {'type':'Look elsewhere','assessment':'All 54 new scores and 144 target/ranking contexts retained; no significance selection.'},
            {'type':'Forking paths','assessment':'Budget specification frozen before additional results, while prior test reuse is disclosed as exploratory.'},
            {'type':'Correlation/causation','assessment':'Budget-only computational intervention checked by exact prefixes; no physical or learning-mechanism causality inferred.'},
            {'type':'Reverse causality','assessment':'Budget and validation selection precede new test scoring; no empirical causal direction claimed.'}],
        'interpretation':'Specified budget robustness; no equivalence, global-convergence or deployment-validity claim.'}
    (HERE/'BUDGET_INTERPRETATION_AUDIT.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf-8')
    print('Wrote Supplement S4 and 11-item interpretation audit.',flush=True)


if __name__=='__main__':
    main()
