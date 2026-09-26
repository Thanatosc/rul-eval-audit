"""Budget-only sensitivity checks for the nine previously flagged fits.

The original fits are read-only controls. Extended fits restart from the same
initialization and retain the original selection/stopping rules. Prefix checks
test that the old training trajectory and predictions are reproduced.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import train_repair as original
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset

SPEC = HERE / 'BUDGET_SPEC.json'
NEURAL_CAP = 450
TREE_CAP = 4500


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def freeze():
    quality = BASE / 'analysis/training_quality.csv'
    with quality.open(encoding='utf-8', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['cap_with_recent_best'].lower()=='true']
    assert len(rows)==9
    source_records=[]
    for row in sorted(rows, key=lambda r:r['run_id']):
        folder = BASE / 'runs' / row['run_id']
        meta=json.loads((folder/'meta.json').read_text(encoding='utf-8'))
        assert meta['status']=='completed' and meta['training']['cap_with_recent_best']
        source_records.append({'run_id':row['run_id'],'model':meta['model'],
            'source_meta_sha256':sha(folder/'meta.json'),'source_curve_sha256':sha(folder/'training_curve.csv'),
            'cell':{f.name:meta[f.name] for f in fields(original.Cell)}})
    record={'recorded_at':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_additional_budget_results',
        'purpose':'Bounded training-budget sensitivity, not an assertion of global convergence or a new independent test population.',
        'selection':'All cap_with_recent_best flags in the completed 96-fit training-quality table; no selection by test performance.',
        'historical_test_results_previously_known':True,'new_fits':9,'neural_max_epochs':NEURAL_CAP,'tree_max_iterations':TREE_CAP,
        'held_fixed':['architecture','input sensors','training target and numerical scale','engine split','initialization','optimizer/learning rate schedule','validation objective','early stopping patience','preprocessing','official test data'],
        'restart_reason':'Optimizer states were not retained for the original neural fit. Restarting from the same initialization avoids an optimizer-reset confound.',
        'prefix_check':'Reproduce the original validation trajectory and all three original prediction tables at the original selected checkpoint/iteration.',
        'selection_of_extended_checkpoint':'Lowest validation RMSE; official test predictions do not select an iteration or budget.',
        'analysis':['Six truth/population scores for each extended fit','Paired engine uncertainty for new-minus-original endpoint errors','Replace only flagged fits in a sensitivity view of the 96-cell grid','Recompute all target/population contrasts and metric winners','Retain and report any flags at the new cap'],
        'primary_grid_policy':'Keep the original 96-fit grid and frozen specification unchanged; report this extra arm separately.',
        'source_quality_sha256':sha(quality),'source_training_script_sha256':sha(BASE/'train_repair.py'),
        'source_spec_sha256':sha(BASE/'ANALYSIS_SPEC.md'),'controls':source_records}
    if SPEC.exists():
        old=json.loads(SPEC.read_text(encoding='utf-8'))
        assert old['controls']==record['controls'] and old['neural_max_epochs']==NEURAL_CAP and old['tree_max_iterations']==TREE_CAP
        print('Existing frozen budget specification retained.',flush=True)
    else:
        HERE.mkdir(parents=True,exist_ok=True)
        dump(SPEC,record)
        print('Froze nine controls and threefold budget caps.',flush=True)


def neural_fit(cell, windows, out, control):
    original.set_deterministic_seed(cell.init_seed)
    torch.cuda.set_device(0)
    model=original.build_neural_model(cell.model,windows['train'].features.shape[-1]).to('cuda:0')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-5)
    generator=torch.Generator().manual_seed(cell.init_seed)
    dataset=TensorDataset(torch.from_numpy(windows['train'].features),torch.as_tensor(windows['train'].labels/cell.target_scale,dtype=torch.float32))
    loader=DataLoader(dataset,batch_size=256,shuffle=True,generator=generator,pin_memory=True)
    old_curve=pd.read_csv(control/'training_curve.csv')
    old_state=torch.load(control/'model.pt',map_location='cpu',weights_only=True)
    old_cap=len(old_curve)
    best,best_epoch,best_state=float('inf'),0,None
    stale,history=0,[]
    start=time.perf_counter()
    prefix_curve_error=0.
    for epoch in range(1,NEURAL_CAP+1):
        model.train()
        total=torch.zeros((),device='cuda:0')
        for bx,by in loader:
            bx,by=bx.to('cuda:0'),by.to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            prediction=model(bx).squeeze(1)
            loss=torch.nn.functional.mse_loss(prediction,by)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
            total+=loss.detach()*len(by)
        vp=original.predict_neural(model,windows['val'].features,batch_size=4096).astype(float)*cell.target_scale
        val=float(np.sqrt(np.mean((vp-windows['val'].labels)**2)))
        if val<best-1e-9:
            best,best_epoch=val,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale=0
        else:
            stale+=1
        row=dict(epoch=epoch,train_batch_mse_cycles=float(total.cpu())/len(dataset)*cell.target_scale**2,
                 val_rmse_cycles=val,best_val_rmse_cycles=best,learning_rate=optimizer.param_groups[0]['lr'],
                 elapsed_seconds=time.perf_counter()-start,selected=False)
        history.append(row)
        if epoch<=old_cap:
            for key in ('train_batch_mse_cycles','val_rmse_cycles','best_val_rmse_cycles','learning_rate'):
                delta=abs(row[key]-float(old_curve.iloc[epoch-1][key]))
                prefix_curve_error=max(prefix_curve_error,delta)
                assert delta<1e-8,(epoch,key,delta)
        if epoch==old_cap:
            assert all(torch.equal(best_state[k],old_state[k]) for k in old_state)
            torch.save(best_state,out/'prefix_model.pt')
        scheduler.step(val)
        pd.DataFrame(history).to_csv(out/'training_curve.csv',index=False)
        if epoch%20==0:
            print(f'{cell.run_id} epoch={epoch} val={val:.5f} best={best:.5f}',flush=True)
        if stale>=15 and epoch>=20:
            break
    assert epoch>=old_cap and best_state is not None
    history[best_epoch-1]['selected']=True
    pd.DataFrame(history).to_csv(out/'training_curve.csv',index=False)
    torch.save(best_state,out/'model.pt')
    def predict(state):
        model.load_state_dict(state)
        return {name:original.predict_neural(model,w.features,batch_size=4096).astype(float)*cell.target_scale for name,w in windows.items() if name!='train'}
    predictions=predict(best_state)
    prefix=predict(old_state)
    audit={'best_val_rmse':best,'best_epoch':best_epoch,'epochs_completed':epoch,'max_epochs_used':NEURAL_CAP,
           'stopped_early':epoch<NEURAL_CAP,'cap_with_recent_best':epoch==NEURAL_CAP and best_epoch>NEURAL_CAP-15,
           'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-start,
           'prefix_validation_max_abs_error':prefix_curve_error,'prefix_weights_exact_match':True,'device':'cuda:0'}
    del model,optimizer,dataset,loader
    gc.collect(); torch.cuda.empty_cache()
    return predictions,prefix,audit


def tree_fit(cell, windows, out, control):
    x={key:value.features.reshape(len(value.labels),-1) for key,value in windows.items()}
    model=lgb.LGBMRegressor(objective='regression',n_estimators=TREE_CAP,learning_rate=.05,num_leaves=31,
        subsample=1.,colsample_bytree=1.,reg_lambda=0.,random_state=cell.init_seed,n_jobs=1,
        deterministic=True,force_row_wise=True,verbosity=-1)
    history={}
    start=time.perf_counter()
    def progress(env):
        if (env.iteration+1)%500==0:
            value=env.evaluation_result_list[0][2]
            print(f'{cell.run_id} tree={env.iteration+1} val={value:.5f}',flush=True)
    progress.order=20
    model.fit(x['train'],windows['train'].labels,eval_set=[(x['val'],windows['val'].labels)],eval_metric='rmse',
              callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False),lgb.record_evaluation(history),progress])
    curve=pd.DataFrame(history['valid_0'])
    curve['iteration']=np.arange(1,len(curve)+1)
    curve['selected']=curve.iteration==model.best_iteration_
    curve.to_csv(out/'training_curve.csv',index=False)
    old_curve=pd.read_csv(control/'training_curve.csv')
    assert len(curve)>=len(old_curve)
    prefix_error=float(np.max(np.abs(curve.rmse.to_numpy()[:len(old_curve)]-old_curve.rmse.to_numpy())))
    assert prefix_error<1e-10,prefix_error
    old_meta=json.loads((control/'meta.json').read_text(encoding='utf-8'))
    old_iteration=int(old_meta['training']['best_iteration'])
    model.booster_.save_model(str(out/'model.txt'),num_iteration=model.best_iteration_)
    predictions={key:model.predict(value,num_iteration=model.best_iteration_) for key,value in x.items() if key!='train'}
    prefix={key:model.predict(value,num_iteration=old_iteration) for key,value in x.items() if key!='train'}
    audit={'best_val_rmse':float(model.best_score_['valid_0']['rmse']),'best_iteration':model.best_iteration_,
           'iterations_evaluated':len(curve),'max_iterations_used':TREE_CAP,'stopped_early':len(curve)<TREE_CAP,
           'cap_with_recent_best':len(curve)>=TREE_CAP and model.best_iteration_>TREE_CAP-50,
           'training_seconds':time.perf_counter()-start,'prefix_validation_max_abs_error':prefix_error,
           'device':'cpu','deterministic_single_record':True}
    return predictions,prefix,audit


def execute(entry):
    cell=original.Cell(**entry['cell'])
    control=BASE/'runs'/entry['run_id']
    assert sha(control/'meta.json')==entry['source_meta_sha256']
    assert sha(control/'training_curve.csv')==entry['source_curve_sha256']
    old=json.loads((control/'meta.json').read_text(encoding='utf-8'))
    assert sha(original.REPO/'src/rul_audit/models/baselines.py')==old['source_model_sha256']
    for file,value in old['data_sha256'].items():
        assert sha(original.REPO/'data/interim/cmapss'/file)==value
    out=HERE/'runs'/entry['run_id']
    meta_path=out/'meta.json'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text(encoding='utf-8'))
        if previous.get('status')=='completed':
            assert previous['budget_spec_sha256']==sha(SPEC) and previous['script_sha256']==sha(__file__)
            print('REUSE '+entry['run_id'],flush=True)
            return
        raise RuntimeError('Inspect incomplete budget fit before retry: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    meta={**entry['cell'],'source_run_id':entry['run_id'],'run_id':entry['run_id']+'__budget_check',
          'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
          'budget_spec_sha256':sha(SPEC),'source_meta_sha256':entry['source_meta_sha256'],
          'source_training_script_sha256':sha(BASE/'train_repair.py'),'torch_version':torch.__version__,
          'lightgbm_version':lgb.__version__,'cuda':torch.version.cuda,'data_sha256':old['data_sha256'],
          'source_model_sha256':old['source_model_sha256']}
    dump(meta_path,meta)
    try:
        data,frames,split,scaler=original.prepare(cell)
        assert split==json.loads((control/'split.json').read_text(encoding='utf-8'))
        with np.load(control/'scaler.npz') as reference:
            for name in ('min_','scale_','data_min_','data_max_'):
                assert np.array_equal(getattr(scaler,name),reference[name])
        dump(out/'split.json',split)
        np.savez(out/'scaler.npz',min_=scaler.min_,scale_=scaler.scale_,data_min_=scaler.data_min_,data_max_=scaler.data_max_)
        fit=tree_fit if cell.model=='lightgbm' else neural_fit
        predictions,prefix,audit=fit(cell,data,out,control)
        prefix_checks=[]
        for key,prediction in predictions.items():
            w=data[key]
            lookup=frames[key].set_index(['unit_id','cycle']).raw_rul
            raw=lookup.reindex(pd.MultiIndex.from_arrays([w.unit_ids,w.cycles])).to_numpy()
            frame=pd.DataFrame(dict(unit_id=w.unit_ids,cycle=w.cycles,raw_rul=raw,true_rul=w.labels,pred_rul=prediction))
            assert np.isfinite(frame.to_numpy()).all() and not frame.duplicated(['unit_id','cycle']).any()
            reference=pd.read_parquet(control/f'preds_{key}.parquet')
            assert np.array_equal(frame[['unit_id','cycle','raw_rul','true_rul']].to_numpy(),reference[['unit_id','cycle','raw_rul','true_rul']].to_numpy())
            delta=float(np.max(np.abs(prefix[key]-reference.pred_rul.to_numpy())))
            assert delta<1e-7,(key,delta)
            prefix_checks.append({'split':key,'rows':len(frame),'maximum_absolute_prediction_difference':delta})
            frame.to_parquet(out/f'preds_{key}.parquet',index=False)
        validation=float(np.sqrt(np.mean((predictions['val']-data['val'].labels)**2)))
        assert np.isclose(validation,audit['best_val_rmse'],rtol=1e-7,atol=1e-7)
        meta.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),training=audit,
                    prefix_prediction_checks=prefix_checks,prediction_rows={k:len(v) for k,v in predictions.items()})
        dump(meta_path,meta)
        print(f'COMPLETED {entry["run_id"]} validation={validation:.6f}',flush=True)
    except Exception as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        dump(meta_path,meta)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--model',choices=['lstm','lightgbm'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.freeze:
        freeze(); return
    spec=json.loads(SPEC.read_text(encoding='utf-8'))
    assert sha(BASE/'train_repair.py')==spec['source_training_script_sha256']
    assert sha(BASE/'ANALYSIS_SPEC.md')==spec['source_spec_sha256']
    selected=[e for e in spec['controls'] if args.model is None or e['model']==args.model]
    assert 0<=args.shard<args.shards
    selected=selected[args.shard::args.shards]
    if any(e['model']=='lstm' for e in selected):
        assert torch.cuda.is_available()
    print(f'Budget sensitivity: {len(selected)} fits in shard {args.shard}/{args.shards}',flush=True)
    for entry in selected:
        execute(entry)
    print('REQUESTED BUDGET CHECKS COMPLETE',flush=True)


if __name__=='__main__':
    main()
