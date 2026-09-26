"""Independently rebuild perturbations, reload weights and recompute summaries."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import train_repair as original


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def per_engine(frame,truth,population):
    f=frame.sort_values(['unit_id','cycle'])
    if population=='endpoint':
        f=f.drop_duplicates('unit_id',keep='last')
    y=f.raw_rul.to_numpy(float)
    if truth=='capped125':
        y=np.minimum(y,125.)
    e=f.pred_rul.to_numpy(float)-y
    a=pd.DataFrame({'engine':f.unit_id.to_numpy(),'squared':e*e,
        'nasa':np.expm1(np.where(e<0,-e/13.,e/10.)),'count':np.ones(len(f))})
    summed=a.groupby('engine',sort=True).sum().to_numpy(float)
    if population!='pooled_windows':
        summed[:,:2]/=summed[:,2,None]
        summed[:,2]=1.
    return summed


def point(a):
    return np.asarray([np.sqrt(a[:,0].sum()/a[:,2].sum()),a[:,1].sum()/a[:,2].sum()])


def boot(a):
    ix=np.random.default_rng(20260926).integers(0,len(a),size=(5000,len(a)),dtype=np.int32)
    draw=a[ix].sum(axis=1)
    return np.column_stack([np.sqrt(draw[:,0]/draw[:,2]),draw[:,1]/draw[:,2]])


def main():
    torch.set_num_threads(1)
    spec=read(HERE/'REVIEW_CHECK_SPEC.json')
    rows=pd.read_csv(HERE/'analysis/all_realization_scores.csv')
    sums=pd.read_csv(HERE/'analysis/paired_sensitivity.csv')
    max_prediction=0.; max_score=0.; max_interval=0.; checked=0; nrows=0
    per_realization={}
    def compare(a,b):
        nonlocal max_score
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))/np.maximum(1.,np.abs(b))))
        max_score=max(max_score,error)
        assert error<1e-10,error
    for entry in spec['models']:
        cell=original.Cell(**entry['cell'])
        source=BASE/entry['source_directory']
        folder=HERE/'model_checks'/cell.run_id
        meta=read(folder/'meta.json')
        assert meta['status']=='completed' and meta['spec_sha256']==sha(HERE/'REVIEW_CHECK_SPEC.json')
        assert meta['script_sha256']==sha(HERE/'run_review_checks.py')
        assert sha(source/'meta.json')==entry['source_meta_sha256']
        weights=source/('model.txt' if cell.model=='lightgbm' else 'model.pt')
        assert sha(weights)==entry['model_sha256']
        windows,frames,split,scaler=original.prepare(cell)
        assert np.all(scaler.data_max_>scaler.data_min_),'Noise fraction requires nonzero training sensor ranges.'
        columns=original.sensor_columns(cell.sensors)
        normalized=original.transform_features(frames['test'],scaler,columns)
        clean=pd.read_parquet(source/'preds_test.parquet')
        if cell.model=='lightgbm':
            model=lgb.Booster(model_file=str(weights))
            predict=lambda x:model.predict(x.reshape(len(x),-1),num_threads=1)
        else:
            original.set_deterministic_seed(cell.init_seed)
            model=original.build_neural_model(cell.model,len(columns)).to('cuda:0')
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)); model.eval()
            predict=lambda x:original.predict_neural(model,x,batch_size=4096).astype(float)*125.
        for truth in ('capped125','raw'):
            for population in ('endpoint','equal_engine_windows','pooled_windows'):
                a=per_engine(clean,truth,population)
                per_realization[(cell.subset,cell.model,'clean',0.,0,truth,population)]=a
                wanted=rows[(rows.subset==cell.subset)&(rows.model==cell.model)&(rows.kind=='clean')&
                    (rows.truth==truth)&(rows.population==population)].iloc[0]
                compare(point(a),[wanted.rmse,wanted.nasa_mean])
        for record in meta['perturbations']:
            rng=np.random.default_rng(record['seed'])
            if record['kind']=='gaussian':
                shift=rng.normal(size=(len(normalized),len(columns)))*record['amplitude']
            else:
                units=np.sort(normalized.unit_id.unique())
                signs=rng.choice(np.asarray([-1.,1.]),size=(len(units),len(columns)))
                shift=signs[np.searchsorted(units,normalized.unit_id.to_numpy())]*record['amplitude']
            assert hashlib.sha256(shift.astype('<f8').tobytes()).hexdigest()==record['perturbation_sha256']
            changed=normalized.copy()
            changed.loc[:,columns]=normalized.loc[:,columns].to_numpy()+shift
            noisy=original._windows_with_short_endpoint_support(changed,columns,window_size=30,stride=1)
            assert np.array_equal(noisy.unit_ids,windows['test'].unit_ids)
            adjacent=np.flatnonzero((np.diff(noisy.unit_ids)==0)&(np.diff(noisy.cycles)==1))
            sampled=adjacent[::max(1,len(adjacent)//100)]
            assert np.array_equal(noisy.features[sampled,1:,:],noisy.features[sampled+1,:-1,:])
            file=folder/record['prediction_file']
            assert sha(file)==record['prediction_sha256']
            frame=pd.read_parquet(file)
            pd.testing.assert_frame_equal(frame[['unit_id','cycle','raw_rul','true_rul']],clean[['unit_id','cycle','raw_rul','true_rul']])
            actual=predict(noisy.features)
            error=float(np.max(np.abs(actual-frame.pred_rul.to_numpy())))
            max_prediction=max(max_prediction,error)
            assert error<1e-7,(cell.run_id,record['scenario'],error)
            checked+=1; nrows+=len(frame)
            for truth in ('capped125','raw'):
                for population in ('endpoint','equal_engine_windows','pooled_windows'):
                    a=per_engine(frame,truth,population)
                    per_realization[(cell.subset,cell.model,record['kind'],record['amplitude'],record['seed'],truth,population)]=a
                    wanted=rows[(rows.subset==cell.subset)&(rows.model==cell.model)&(rows.kind==record['kind'])&
                        (rows.amplitude==record['amplitude'])&(rows.seed==record['seed'])&
                        (rows.truth==truth)&(rows.population==population)].iloc[0]
                    compare(point(a),[wanted.rmse,wanted.nasa_mean])
        print('REPLAY VERIFIED '+cell.run_id,flush=True)
        del model
        torch.cuda.empty_cache()
    for row in sums.itertuples():
        clean=per_realization[(row.subset,row.model,'clean',0.,0,row.truth,row.population)]
        current=clean if row.kind=='clean' else np.mean([per_realization[(row.subset,row.model,row.kind,row.amplitude,seed,row.truth,row.population)] for seed in spec['perturbation_seeds']],axis=0)
        compare(point(current),[row.rmse,row.nasa_mean])
        compare(point(current)-point(clean),[row.rmse_difference,row.nasa_mean_difference])
        uncertainty=boot(current)-boot(clean)
        ci=np.quantile(uncertainty,[.025,.975],axis=0)
        expected=np.asarray([[row.rmse_difference_ci_lower,row.nasa_difference_ci_lower],
                             [row.rmse_difference_ci_upper,row.nasa_difference_ci_upper]])
        er=float(np.max(np.abs(ci-expected)/np.maximum(1.,np.abs(expected))))
        max_interval=max(max_interval,er)
        assert er<1e-9,(row.subset,row.model,row.kind,er)
    assert checked==72 and len(rows)==468 and len(sums)==180
    result={'status':'passed','fixed_models_reloaded':6,'perturbed_tables_replayed':checked,
        'perturbed_prediction_rows':nrows,'maximum_prediction_replay_error':max_prediction,
        'realization_scores_independently_recomputed':len(rows),'summary_scores_independently_recomputed':len(sums),
        'paired_intervals_independently_recomputed':288,'maximum_normalized_score_error':max_score,
        'maximum_normalized_interval_error':max_interval,'all_sensor_training_ranges_positive':True,
        'overlapping_observation_consistency_checked':True,'spec_sha256':sha(HERE/'REVIEW_CHECK_SPEC.json'),
        'analysis_script_sha256':sha(HERE/'analyze_review_checks.py'),'verification_script_sha256':sha(__file__),
        'scope':'Computational replay and aggregation; not a validation of the chosen noise amplitudes against field measurements.'}
    (HERE/'REVIEW_CHECK_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
