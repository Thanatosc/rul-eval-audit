"""Analyze the additional budget arm without changing the primary 96-fit grid."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / 'analysis'
sys.path.insert(0, str(BASE))
from analyze_repair import TRUTHS, POPS, BOOT, unit_losses, estimate, boot_metric, interval, mean_loss, residual_uq, bootstrap_indices

KEYS = ['subset', 'split_seed', 'model', 'init_seed', 'target', 'sensors']
CONTEXT = ['subset', 'split_seed', 'init_seed', 'truth', 'population']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def ranking(scores):
    lookup = {(tuple(row[k] for k in KEYS), row.truth, row.population): row
              for _, row in scores.iterrows()}
    rows = []
    for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
        for split in (42, 137, 271):
            for init in (11, 23):
                for truth in TRUTHS:
                    for population in POPS:
                        values = []
                        for model in ('lstm', 'cnn_1d', 'lightgbm'):
                            key = (subset, split, model, 11 if model=='lightgbm' else init,
                                   'piecewise_125', 'common_14')
                            row = lookup[(key, truth, population)]
                            values.append((model, row.rmse, row.nasa_mean))
                        rm, ns = min(x[1] for x in values), min(x[2] for x in values)
                        wr = {x[0] for x in values if abs(x[1]-rm)<1e-10}
                        wn = {x[0] for x in values if abs(x[2]-ns)<1e-10}
                        rows.append(dict(subset=subset, split_seed=split, init_seed=init,
                            truth=truth, population=population, rmse_winner='|'.join(sorted(wr)),
                            nasa_winner='|'.join(sorted(wn)), conflict=not bool(wr & wn)))
    return pd.DataFrame(rows)


def target_contrasts(scores):
    selected = scores[(scores.subset.isin(['FD001','FD004'])) &
                      (scores.model.isin(['lstm','lightgbm'])) & (scores.init_seed==11)]
    table = selected.pivot(index=['subset','split_seed','model','sensors','truth','population'],
                           columns='target', values='rmse').reset_index()
    table.columns.name = None
    table['raw_minus_capped_rmse'] = table.linear_uncapped-table.piecewise_125
    assert len(table)==144
    return table


def population_winner_changes(ranks, truth):
    chosen = ranks[ranks.truth==truth]
    wide = chosen.pivot(index=['subset','split_seed','init_seed'], columns='population', values='rmse_winner')
    return {pop:int((wide[pop]!=wide.endpoint).sum()) for pop in POPS if pop!='endpoint'}


def main():
    spec = read_json(HERE/'BUDGET_SPEC.json')
    assert spec['new_fits']==9
    assert sha(BASE/'train_repair.py')==spec['source_training_script_sha256']
    assert sha(BASE/'ANALYSIS_SPEC.md')==spec['source_spec_sha256']
    assert sha(BASE/'analysis/training_quality.csv')==spec['source_quality_sha256']
    controls = {entry['run_id']:entry for entry in spec['controls']}
    assert len(controls)==9
    primary_files = sorted((BASE/'runs').glob('*/meta.json'))
    assert len(primary_files)==96
    primary_hashes = {f.relative_to(BASE).as_posix():sha(f) for f in primary_files}
    primary_hashes.update({name:sha(BASE/name) for name in
        ('train_repair.py','ANALYSIS_SPEC.md','analysis/all_run_scores.csv','analysis/metric_winner_contexts.csv')})
    OUT.mkdir(exist_ok=True)
    scores, new_scores, training, paired, prediction_records = [], [], [], [], []
    losses, lookup = {}, {}
    core = defaultdict(list)
    stages = []
    replaced = []
    for meta_file in primary_files:
        old = read_json(meta_file)
        assert old['status']=='completed'
        run_id = old['run_id']
        selected_path = meta_file.parent
        meta = old
        changed = run_id in controls
        if changed:
            entry = controls[run_id]
            assert sha(meta_file)==entry['source_meta_sha256']
            assert sha(meta_file.parent/'training_curve.csv')==entry['source_curve_sha256']
            selected_path = HERE/'runs'/run_id
            meta = read_json(selected_path/'meta.json')
            assert meta['status']=='completed', run_id
            assert meta['budget_spec_sha256']==sha(HERE/'BUDGET_SPEC.json')
            assert meta['script_sha256']==sha(HERE/'run_budget_checks.py')
            assert all(meta[k]==old[k] for k in KEYS)
            for name in ('val','calib','test'):
                path = selected_path/f'preds_{name}.parquet'
                frame = pd.read_parquet(path)
                reference = pd.read_parquet(meta_file.parent/f'preds_{name}.parquet')
                assert np.array_equal(frame[['unit_id','cycle','raw_rul','true_rul']].to_numpy(),
                                      reference[['unit_id','cycle','raw_rul','true_rul']].to_numpy())
                assert np.isfinite(frame.to_numpy()).all()
                assert not frame.duplicated(['unit_id','cycle']).any()
                prediction_records.append(dict(run_id=run_id, split=name, rows=len(frame),
                    engines=int(frame.unit_id.nunique()), sha256=sha(path)))
            t, o = meta['training'], old['training']
            step = 'best_iteration' if meta['model']=='lightgbm' else 'best_epoch'
            done = 'iterations_evaluated' if meta['model']=='lightgbm' else 'epochs_completed'
            training.append(dict(run_id=run_id, **{k:old[k] for k in KEYS},
                old_selected=o[step], new_selected=t[step], old_evaluated=o[done], new_evaluated=t[done],
                old_val_rmse=o['best_val_rmse'], new_val_rmse=t['best_val_rmse'],
                validation_difference=t['best_val_rmse']-o['best_val_rmse'],
                remaining_cap_flag=t['cap_with_recent_best'], prefix_curve_max_abs_error=t['prefix_validation_max_abs_error'],
                prefix_prediction_max_abs_error=max(r['maximum_absolute_prediction_difference'] for r in meta['prefix_prediction_checks']),
                training_seconds=t['training_seconds']))
            replaced.append(run_id)
        frame = pd.read_parquet(selected_path/'preds_test.parquet')
        record = {k:old[k] for k in ['run_id','arm']+KEYS}
        record.update(analysis_run_id=meta['run_id'], budget_extended=changed)
        lookup[tuple(old[k] for k in KEYS)] = run_id
        old_frame = pd.read_parquet(meta_file.parent/'preds_test.parquet') if changed else None
        for truth in TRUTHS:
            for population in POPS:
                loss = unit_losses(frame, truth, population)
                losses[(run_id,truth,population)] = loss
                row = dict(**record,truth=truth,population=population,**estimate(loss))
                scores.append(row)
                if changed:
                    prior_loss = unit_losses(old_frame,truth,population)
                    previous = estimate(prior_loss)
                    new_scores.append(dict(**row, original_rmse=previous['rmse'], original_nasa_mean=previous['nasa_mean'],
                        rmse_difference=row['rmse']-previous['rmse'], nasa_mean_difference=row['nasa_mean']-previous['nasa_mean']))
                    if population=='endpoint':
                        assert loss.index.equals(prior_loss.index)
                        rn, nn = boot_metric(loss)
                        ro, no = boot_metric(prior_loss)
                        ri, ni = interval(rn-ro), interval(nn-no)
                        paired.append(dict(**row,original_rmse=previous['rmse'],rmse_difference=row['rmse']-previous['rmse'],
                            rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],original_nasa_mean=previous['nasa_mean'],
                            nasa_mean_difference=row['nasa_mean']-previous['nasa_mean'],
                            nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1]))
                if old['arm']=='core':
                    core[(old['subset'],old['split_seed'],old['model'],truth,population)].append(loss)
        if old['arm']=='core':
            stages.append(residual_uq(old,frame,selected_path))
    assert len(replaced)==9 and len(prediction_records)==27
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT/'sensitivity_all_run_scores.csv',index=False)
    pd.DataFrame(new_scores).to_csv(OUT/'new_run_score_changes.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_endpoint_budget_changes.csv',index=False)
    training = pd.DataFrame(training)
    training.to_csv(OUT/'training_budget_comparison.csv',index=False)
    save_json(OUT/'NEW_PREDICTION_MANIFEST.json',prediction_records)

    ranks = ranking(scores)
    ranks.to_csv(OUT/'sensitivity_metric_winner_contexts.csv',index=False)
    old_ranks = pd.read_csv(BASE/'analysis/metric_winner_contexts.csv')
    rank_changes = old_ranks.merge(ranks,on=CONTEXT,suffixes=('_original','_sensitivity'),validate='one_to_one')
    for metric in ('rmse','nasa'):
        rank_changes[metric+'_winner_changed'] = rank_changes[metric+'_winner_original']!=rank_changes[metric+'_winner_sensitivity']
    rank_changes.to_csv(OUT/'metric_winner_changes.csv',index=False)
    targets = target_contrasts(scores)
    targets.to_csv(OUT/'sensitivity_target_population_contrasts.csv',index=False)
    target_summary = []
    for (truth,pop), group in targets.groupby(['truth','population']):
        d = group.raw_minus_capped_rmse
        assert len(d)==24
        target_summary.append(dict(truth=truth,population=pop,comparisons=len(d),difference_min=float(d.min()),
            difference_max=float(d.max()),raw_trained_lower_rmse=int((d<-1e-10).sum()),
            capped_trained_lower_rmse=int((d>1e-10).sum()),ties=int((d.abs()<=1e-10).sum())))
    pd.DataFrame(target_summary).to_csv(OUT/'sensitivity_target_population_summary.csv',index=False)
    original_targets = target_contrasts(pd.read_csv(BASE/'analysis/all_run_scores.csv'))
    target_changes = original_targets.merge(targets,on=['subset','split_seed','model','sensors','truth','population'],
        suffixes=('_original','_sensitivity'),validate='one_to_one')
    target_changes['preference_changed'] = np.sign(target_changes.raw_minus_capped_rmse_original)!=np.sign(target_changes.raw_minus_capped_rmse_sensitivity)
    target_changes.to_csv(OUT/'target_preference_changes.csv',index=False)

    core_rows = []
    for (subset,split,model,truth,pop), values in core.items():
        assert len(values)==(1 if model=='lightgbm' else 2)
        loss=mean_loss(values)
        row=dict(subset=subset,split_seed=split,model=model,truth=truth,population=pop,
            initialization_records=len(values),**estimate(loss))
        if pop=='endpoint':
            rm,ns=boot_metric(loss); ri,ni=interval(rm),interval(ns)
            row.update(rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1])
        core_rows.append(row)
    pd.DataFrame(core_rows).to_csv(OUT/'sensitivity_core_scores.csv',index=False)
    factor_rows = []
    for subset in ('FD001','FD004'):
        for split in (42,137,271):
            for model in ('lstm','lightgbm'):
                for kind,fixed_values in (('training_target',('common_14','all_21')),
                                           ('sensor_input',('piecewise_125','linear_uncapped'))):
                    for fixed in fixed_values:
                        if kind=='training_target':
                            ka=(subset,split,model,11,'piecewise_125',fixed)
                            kb=(subset,split,model,11,'linear_uncapped',fixed)
                        else:
                            ka=(subset,split,model,11,fixed,'common_14')
                            kb=(subset,split,model,11,fixed,'all_21')
                        for truth in TRUTHS:
                            a,b=losses[(lookup[ka],truth,'endpoint')],losses[(lookup[kb],truth,'endpoint')]
                            assert a.index.equals(b.index)
                            ar,an=boot_metric(a); br,bn=boot_metric(b)
                            ri,ni=interval(br-ar),interval(bn-an)
                            ea,eb=estimate(a),estimate(b)
                            factor_rows.append(dict(subset=subset,split_seed=split,model=model,init_seed=11,factor=kind,
                                fixed_level=fixed,truth=truth,rmse_difference=eb['rmse']-ea['rmse'],
                                rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],nasa_mean_difference=eb['nasa_mean']-ea['nasa_mean'],
                                nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1],engines=len(a)))
    pd.DataFrame(factor_rows).to_csv(OUT/'sensitivity_paired_factor_contrasts.csv',index=False)
    qu=pd.concat(stages,ignore_index=True)
    qu.to_parquet(OUT/'sensitivity_uq_per_engine_stage.parquet',index=False)
    uq_rows=[]
    for (subset,split,model,stage), group in qu.groupby(['subset','split_seed','model','stage']):
        per=group.groupby('unit_id')[['coverage','width','interval_score','windows']].mean()
        row=dict(subset=subset,split_seed=split,model=model,stage=stage,engines=len(per))
        for column in ('coverage','width','interval_score'):
            values=per[column].to_numpy(); ci=interval(values[bootstrap_indices(len(per))].mean(axis=1))
            row.update({column:float(values.mean()),column+'_ci_lower':ci[0],column+'_ci_upper':ci[1]})
        uq_rows.append(row)
    pd.DataFrame(uq_rows).to_csv(OUT/'sensitivity_uq_stage_summary.csv',index=False)
    assert all(sha(BASE/name)==value for name,value in primary_hashes.items())
    primary_filter = (rank_changes.truth=='capped125') & (rank_changes.population=='endpoint')
    primary_changes=rank_changes[primary_filter]
    summary=dict(status='complete',recorded_at=datetime.now(timezone.utc).isoformat(),
        original_grid_fits=96,diagnostic_fits=2,new_budget_fits=9,total_fits=107,
        new_prediction_tables=27,new_prediction_rows=sum(r['rows'] for r in prediction_records),
        source_grid_preserved=True,budget_spec_sha256=sha(HERE/'BUDGET_SPEC.json'),
        remaining_cap_flags=int(training.remaining_cap_flag.sum()),
        remaining_flag_run_ids=training.loc[training.remaining_cap_flag,'run_id'].tolist(),
        all_validation_scores_nonincreasing=bool((training.validation_difference<=1e-7).all()),
        maximum_prefix_curve_error=float(training.prefix_curve_max_abs_error.max()),
        maximum_prefix_prediction_error=float(training.prefix_prediction_max_abs_error.max()),
        primary_rmse_winner_changes=int(primary_changes.rmse_winner_changed.sum()),
        primary_nasa_winner_changes=int(primary_changes.nasa_winner_changed.sum()),
        primary_metric_conflicts_original=int(primary_changes.conflict_original.sum()),
        primary_metric_conflicts_sensitivity=int(primary_changes.conflict_sensitivity.sum()),
        all_context_rmse_winner_changes=int(rank_changes.rmse_winner_changed.sum()),
        all_context_nasa_winner_changes=int(rank_changes.nasa_winner_changed.sum()),
        primary_population_winner_changes_original=population_winner_changes(old_ranks,'capped125'),
        primary_population_winner_changes_sensitivity=population_winner_changes(ranks,'capped125'),
        target_preference_changes=int(target_changes.preference_changed.sum()),
        target_population_summary=target_summary,bootstrap_replicates=BOOT,
        primary_file_hashes=primary_hashes,
        caveats=['Budget extension is exploratory on previously examined official test data.',
                 'The sensitivity grid substitutes nine fits; it is not another independent 96-fit experiment.',
                 'Tree predictions appear in both neural initialization contexts only as a shared reference.',
                 'Engine bootstrap intervals condition on the selected fits and, for coverage, calibration scores.',
                 'A finite budget and validation early stopping do not establish global convergence.'])
    save_json(OUT/'BUDGET_SUMMARY.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='primary_file_hashes'},indent=2),flush=True)


if __name__=='__main__':
    main()
