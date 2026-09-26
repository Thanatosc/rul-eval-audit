"""Independently recompute budget scores and reload all nine saved models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import train_repair as original


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(frame, truth, population):
    work=frame.sort_values(['unit_id','cycle']).copy()
    if population=='endpoint':
        work=work.drop_duplicates('unit_id',keep='last')
    target=work.raw_rul.to_numpy(float)
    if truth=='capped125':
        target=np.clip(target,None,125.)
    error=work.pred_rul.to_numpy(float)-target
    work['squared']=np.square(error)
    work['nasa_loss']=np.expm1(np.where(error<0,-error/13.,error/10.))
    if population=='equal_engine_windows':
        units=work.groupby('unit_id')[['squared','nasa_loss']].mean()
        return np.sqrt(units.squared.mean()),units.nasa_loss.mean()
    return np.sqrt(work.squared.mean()),work.nasa_loss.mean()


def main():
    torch.set_num_threads(1)
    spec=read(HERE/'BUDGET_SPEC.json')
    summary=read(HERE/'analysis/BUDGET_SUMMARY.json')
    scores=pd.read_csv(HERE/'analysis/new_run_score_changes.csv')
    paired=pd.read_csv(HERE/'analysis/paired_endpoint_budget_changes.csv')
    assert len(scores)==54 and len(paired)==18 and summary['status']=='complete'
    records=[]
    score_error=0.
    reload_error=0.
    row_count=0
    interval_error=0.
    for entry in spec['controls']:
        name=entry['run_id']
        path=HERE/'runs'/name
        control=BASE/'runs'/name
        meta=read(path/'meta.json')
        assert meta['status']=='completed'
        assert meta['budget_spec_sha256']==sha(HERE/'BUDGET_SPEC.json')
        assert meta['script_sha256']==sha(HERE/'run_budget_checks.py')
        assert meta['source_meta_sha256']==sha(control/'meta.json')==entry['source_meta_sha256']
        assert sha(control/'training_curve.csv')==entry['source_curve_sha256']
        cell=original.Cell(**entry['cell'])
        data,frames,split,scaler=original.prepare(cell)
        assert split==read(path/'split.json')==read(control/'split.json')
        curve=pd.read_csv(path/'training_curve.csv')
        old_curve=pd.read_csv(control/'training_curve.csv')
        assert len(curve)>=len(old_curve) and int(curve.selected.sum())==1
        if cell.model=='lightgbm':
            model=lgb.Booster(model_file=str(path/'model.txt'))
            valcurve=curve.rmse
            np.testing.assert_allclose(curve.rmse.iloc[:len(old_curve)],old_curve.rmse,rtol=0,atol=1e-10)
        else:
            original.set_deterministic_seed(cell.init_seed)
            model=original.build_neural_model(cell.model,data['train'].features.shape[-1]).to('cuda:0')
            model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True))
            old_weights=torch.load(control/'model.pt',map_location='cpu',weights_only=True)
            prefix_weights=torch.load(path/'prefix_model.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(old_weights[k],prefix_weights[k]) for k in old_weights)
            valcurve=curve.val_rmse_cycles
            for col in ('train_batch_mse_cycles','val_rmse_cycles','best_val_rmse_cycles','learning_rate'):
                np.testing.assert_allclose(curve[col].iloc[:len(old_curve)],old_curve[col],rtol=0,atol=1e-8)
        chosen=float(valcurve[curve.selected].iloc[0])
        assert abs(chosen-float(valcurve.min()))<1e-7
        assert abs(chosen-meta['training']['best_val_rmse'])<1e-7
        for part in ('val','calib','test'):
            file=path/f'preds_{part}.parquet'
            frame=pd.read_parquet(file)
            previous=pd.read_parquet(control/f'preds_{part}.parquet')
            np.testing.assert_array_equal(frame[['unit_id','cycle','raw_rul','true_rul']].to_numpy(),
                                          previous[['unit_id','cycle','raw_rul','true_rul']].to_numpy())
            assert np.isfinite(frame.to_numpy()).all() and not frame.duplicated(['unit_id','cycle']).any()
            features=data[part].features
            if cell.model=='lightgbm':
                prediction=model.predict(features.reshape(len(features),-1),num_threads=1)
            else:
                prediction=original.predict_neural(model,features,batch_size=4096).astype(float)*cell.target_scale
            error=float(np.max(np.abs(prediction-frame.pred_rul.to_numpy())))
            reload_error=max(reload_error,error)
            assert error<1e-7,(name,part,error)
            row_count+=len(frame)
            records.append(dict(run_id=name,split=part,rows=len(frame),sha256=sha(file),reload_max_abs_error=error))
            if part=='test':
                for truth in ('capped125','raw'):
                    for population in ('endpoint','equal_engine_windows','pooled_windows'):
                        rmse,nasa=aggregate(frame,truth,population)
                        row=scores[(scores.run_id==name)&(scores.truth==truth)&(scores.population==population)].iloc[0]
                        er=max(abs(rmse-row.rmse),abs(nasa-row.nasa_mean)/max(1.,abs(nasa)))
                        score_error=max(score_error,er)
                        assert er<1e-8,(name,truth,population,er)
                    ep=frame.sort_values(['unit_id','cycle']).drop_duplicates('unit_id',keep='last')
                    old=previous.sort_values(['unit_id','cycle']).drop_duplicates('unit_id',keep='last')
                    assert np.array_equal(ep.unit_id,old.unit_id)
                    y=ep.raw_rul.to_numpy(float)
                    if truth=='capped125':
                        y=np.minimum(y,125.)
                    en,eo=ep.pred_rul.to_numpy()-y,old.pred_rul.to_numpy()-y
                    ix=np.random.default_rng(20260926).integers(0,len(ep),size=(5000,len(ep)),dtype=np.int32)
                    rm=np.sqrt(np.square(en)[ix].mean(axis=1))-np.sqrt(np.square(eo)[ix].mean(axis=1))
                    nn=np.expm1(np.where(en<0,-en/13.,en/10.))
                    no=np.expm1(np.where(eo<0,-eo/13.,eo/10.))
                    nd=nn[ix].mean(axis=1)-no[ix].mean(axis=1)
                    expected=paired[(paired.run_id==name)&(paired.truth==truth)].iloc[0]
                    for values,lo,hi in ((rm,'rmse_ci_lower','rmse_ci_upper'),(nd,'nasa_mean_ci_lower','nasa_mean_ci_upper')):
                        ci=np.quantile(values,[.025,.975])
                        e=float(np.max(np.abs(ci-np.asarray([expected[lo],expected[hi]]))))
                        interval_error=max(interval_error,e)
                        assert e<1e-8,(name,truth,lo,e)
        print('VERIFIED '+name,flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name,value in summary['primary_file_hashes'].items():
        assert sha(BASE/name)==value
    assert row_count==summary['new_prediction_rows']
    result=dict(status='passed',recorded_at=datetime.now(timezone.utc).isoformat(),new_models_reloaded=9,
        new_prediction_tables=len(records),new_prediction_rows=row_count,independently_recomputed_scores=54,
        paired_endpoint_intervals_recomputed=36,maximum_score_error=score_error,maximum_reload_error=reload_error,
        maximum_interval_error=interval_error,original_grid_metadata_and_analysis_unchanged=True,
        script_sha256=sha(__file__),analysis_script_sha256=sha(HERE/'analyze_budget_checks.py'),
        budget_spec_sha256=sha(HERE/'BUDGET_SPEC.json'),prediction_records=records,
        scope='Numerical and artifact verification; not evidence of generalization or global convergence.')
    (HERE/'BUDGET_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='prediction_records'},indent=2),flush=True)


if __name__=='__main__':
    main()
