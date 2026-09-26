"""Describe the complete fixed-model sensor/latency checks, without fitting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUT=HERE/'analysis'
sys.path.insert(0,str(BASE))
from analyze_repair import unit_losses,estimate,mean_loss,boot_metric,interval,TRUTHS,POPS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    spec=read(HERE/'REVIEW_CHECK_SPEC.json')
    assert len(spec['models'])==6
    OUT.mkdir(exist_ok=True)
    realizations=[]; summaries=[]; latency=[]; raw_latency=[]; models=[]
    checks=[]; perturbation_hashes=defaultdict(set)
    for entry in spec['models']:
        cell=entry['cell']
        name=entry['source_directory'].rsplit('/',1)[1]
        source=BASE/entry['source_directory']
        out=HERE/'model_checks'/name
        meta=read(out/'meta.json')
        assert meta['status']=='completed' and meta['spec_sha256']==sha(HERE/'REVIEW_CHECK_SPEC.json')
        assert meta['script_sha256']==sha(HERE/'run_review_checks.py')
        assert sha(source/'meta.json')==entry['source_meta_sha256']
        assert len(meta['perturbations'])==12 and meta['training_performed'] is False
        clean=pd.read_parquet(source/'preds_test.parquet')
        assert sha(source/'preds_test.parquet')==entry['clean_test_sha256']
        descriptor={k:cell[k] for k in ('subset','model','split_seed','init_seed')}
        models.append(dict(**descriptor,source_directory=entry['source_directory'],checkpoint_bytes=meta['checkpoint_bytes'],
            clean_prediction_max_abs_error=meta['clean_prediction_max_abs_error'],**meta['complexity']))
        samples=pd.read_csv(out/'latency_samples.csv')
        assert len(samples)==100 and (samples.latency_ms>0).all()
        for key,value in descriptor.items():
            samples[key]=value
        raw_latency.append(samples)
        for row in meta['latency']:
            observed=samples[samples.batch==row['batch']].latency_ms
            assert len(observed)==50
            assert abs(np.median(observed)-row['median_batch_ms'])<1e-9
            assert abs(np.quantile(observed,.95)-row['p95_batch_ms'])<1e-9
            latency.append(dict(**descriptor,device='CPU (one thread)' if cell['model']=='lightgbm' else 'RTX 4060 Laptop GPU',
                checkpoint_bytes=meta['checkpoint_bytes'],**meta['complexity'],**row))
        baseline={}
        for truth in TRUTHS:
            for population in POPS:
                loss=unit_losses(clean,truth,population)
                baseline[(truth,population)]=loss
                metrics=estimate(loss); rb,nb=boot_metric(loss); ri,ni=interval(rb),interval(nb)
                row=dict(**descriptor,kind='clean',amplitude=0.,truth=truth,population=population,
                    corruption_realizations=0,**metrics,rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],
                    nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1],rmse_difference=0.,
                    rmse_difference_ci_lower=0.,rmse_difference_ci_upper=0.,nasa_mean_difference=0.,
                    nasa_difference_ci_lower=0.,nasa_difference_ci_upper=0.)
                summaries.append(row)
                realizations.append(dict(**descriptor,kind='clean',amplitude=0.,seed=0,truth=truth,population=population,**metrics))
        groups=defaultdict(list)
        for record in meta['perturbations']:
            file=out/record['prediction_file']
            assert sha(file)==record['prediction_sha256']
            frame=pd.read_parquet(file)
            assert len(frame)==record['rows'] and np.isfinite(frame.to_numpy()).all()
            pd.testing.assert_frame_equal(frame[['unit_id','cycle','raw_rul','true_rul']],clean[['unit_id','cycle','raw_rul','true_rul']])
            perturbation_hashes[(cell['subset'],record['kind'],record['amplitude'],record['seed'])].add(record['perturbation_sha256'])
            for truth in TRUTHS:
                for population in POPS:
                    loss=unit_losses(frame,truth,population)
                    groups[(record['kind'],record['amplitude'],truth,population)].append(loss)
                    realizations.append(dict(**descriptor,kind=record['kind'],amplitude=record['amplitude'],
                        seed=record['seed'],truth=truth,population=population,**estimate(loss)))
            checks.append(dict(path=file.relative_to(BASE).as_posix(),rows=len(frame),sha256=sha(file)))
        for (kind,amplitude,truth,population),values in groups.items():
            assert len(values)==3
            combined=mean_loss(values)
            base=baseline[(truth,population)]
            assert base.index.equals(combined.index)
            metrics=estimate(combined); original=estimate(base)
            rn,nn=boot_metric(combined); rb,nb=boot_metric(base)
            ri,ni=interval(rn),interval(nn); dr,dn=interval(rn-rb),interval(nn-nb)
            summaries.append(dict(**descriptor,kind=kind,amplitude=amplitude,truth=truth,population=population,
                corruption_realizations=3,**metrics,rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],
                nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1],rmse_difference=metrics['rmse']-original['rmse'],
                rmse_difference_ci_lower=dr[0],rmse_difference_ci_upper=dr[1],
                nasa_mean_difference=metrics['nasa_mean']-original['nasa_mean'],nasa_difference_ci_lower=dn[0],nasa_difference_ci_upper=dn[1]))
    assert len(checks)==72 and len(perturbation_hashes)==24 and all(len(v)==1 for v in perturbation_hashes.values())
    scores=pd.DataFrame(realizations); summary_scores=pd.DataFrame(summaries)
    assert len(scores)==468 and len(summary_scores)==180
    scores.to_csv(OUT/'all_realization_scores.csv',index=False)
    summary_scores.to_csv(OUT/'paired_sensitivity.csv',index=False)
    pd.DataFrame(latency).to_csv(OUT/'latency_summary.csv',index=False)
    pd.concat(raw_latency,ignore_index=True).to_csv(OUT/'latency_samples.csv',index=False)
    pd.DataFrame(models).to_csv(OUT/'model_complexity.csv',index=False)
    winners=[]
    for (subset,kind,amplitude,truth,pop),group in summary_scores.groupby(['subset','kind','amplitude','truth','population']):
        assert len(group)==3
        rmse=group.rmse.min(); nasa=group.nasa_mean.min()
        winners.append(dict(subset=subset,kind=kind,amplitude=amplitude,truth=truth,population=pop,
            rmse_winner='|'.join(sorted(group.loc[(group.rmse-rmse).abs()<1e-10,'model'])),
            nasa_winner='|'.join(sorted(group.loc[(group.nasa_mean-nasa).abs()<1e-10,'model']))))
    ranks=pd.DataFrame(winners)
    clean=ranks[ranks.kind=='clean'][['subset','truth','population','rmse_winner','nasa_winner']]
    comparison=ranks.merge(clean,on=['subset','truth','population'],suffixes=('','_clean'),validate='many_to_one')
    comparison['rmse_winner_changed']=comparison.rmse_winner!=comparison.rmse_winner_clean
    comparison['nasa_winner_changed']=comparison.nasa_winner!=comparison.nasa_winner_clean
    comparison.to_csv(OUT/'model_winners.csv',index=False)
    first='core__fd001__s42__lstm__i11__piecewise_125__common_14__scale125'
    failed=HERE/'failed_attempts/metadata_duplicate_rows'/first/'preds_test__gaussian__a0.01__seed101.parquet'
    final=HERE/'model_checks'/first/failed.name
    expected_partial=read(HERE/'EXECUTION_CORRECTION.json')['failed_partial_prediction_sha256']
    assert sha(final)==expected_partial
    if failed.exists():
        assert sha(failed)==expected_partial
    endpoint=summary_scores[(summary_scores.truth=='capped125')&(summary_scores.population=='endpoint')]
    primary_ranks=comparison[(comparison.truth=='capped125')&(comparison.population=='endpoint')&(comparison.kind!='clean')]
    result={'status':'complete','new_fits':0,'fixed_predictors':6,'perturbed_prediction_tables':72,
        'perturbed_prediction_rows':sum(x['rows'] for x in checks),'realization_score_rows':len(scores),
        'paired_summary_rows':len(summary_scores),'winner_contexts':len(comparison),
        'latency_settings':len(latency),'measured_latency_calls':sum(r['repeats'] for r in latency),
        'shared_perturbation_draws_verified':24,'max_clean_prediction_error':max(x['clean_prediction_max_abs_error'] for x in models),
        'first_failed_partial_prediction_matches_completed_attempt':True,
        'primary_perturbation_rmse_winner_changes':int(primary_ranks.rmse_winner_changed.sum()),
        'primary_perturbation_nasa_winner_changes':int(primary_ranks.nasa_winner_changed.sum()),
        'primary_perturbation_contexts':len(primary_ranks),
        'five_percent_gaussian_endpoint':endpoint[(endpoint.kind=='gaussian')&(endpoint.amplitude==.05)].to_dict(orient='records'),
        'three_percent_bias_endpoint':endpoint[endpoint.kind=='engine_bias'].to_dict(orient='records'),
        'spec_sha256':sha(HERE/'REVIEW_CHECK_SPEC.json'),'prediction_files':checks,
        'caveats':spec['caveats']}
    (OUT/'REVIEW_CHECK_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('prediction_files','five_percent_gaussian_endpoint','three_percent_bias_endpoint')},indent=2),flush=True)
    print(endpoint[['subset','model','kind','amplitude','rmse','rmse_difference']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
