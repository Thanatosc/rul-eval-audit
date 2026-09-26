"""Bounded sensor-perturbation and inference-cost checks on six fixed predictors.

No model fitting, test-driven selection, or changes to the primary/budget grids.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import train_repair as training

SEEDS=[101,211,307]
SCENARIOS=[('gaussian',.01),('gaussian',.03),('gaussian',.05),('engine_bias',.03)]
SPEC=HERE/'REVIEW_CHECK_SPEC.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def freeze():
    entries=[]
    for subset in ('FD001','FD004'):
        for model in ('lstm','cnn_1d','lightgbm'):
            cell=training.Cell(subset,42,model,11)
            folder=BASE/'runs'/cell.run_id
            extension=BASE/'budget_sensitivity/runs'/cell.run_id
            if extension.exists():
                folder=extension
            meta=read(folder/'meta.json')
            assert meta['status']=='completed' and not meta['training']['cap_with_recent_best']
            model_file=folder/('model.txt' if model=='lightgbm' else 'model.pt')
            entries.append({'cell':asdict(cell),'source_directory':folder.relative_to(BASE).as_posix(),
                'source_meta_sha256':sha(folder/'meta.json'),'model_sha256':sha(model_file),
                'clean_test_sha256':sha(folder/'preds_test.parquet')})
    spec={'recorded_at':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_followup_results',
        'purpose':'Address sensor-noise sensitivity and measured inference cost for representative fixed models.',
        'models':entries,'new_fits':0,'representative_predictors':6,
        'selection':'FD001 and FD004 provide the simpler and multi-condition/multi-fault benchmark settings; partition 42, initialization 11, capped training and 14 sensors are fixed. Use the budget-extended checkpoint when that cell was flagged.',
        'scenarios':[{'kind':k,'amplitude':a} for k,a in SCENARIOS],'perturbation_seeds':SEEDS,
        'perturbation_unit':'Unique sensor observations before window construction; an overlapping observation reuses the same corruption. Short-engine left padding duplicates the corrupted first observation.',
        'scale':'Normalized units of the training-only MinMax scaler. Gaussian standard deviation or fixed absolute bias is the stated fraction of each training sensor range. No clipping.',
        'engine_bias':'Independent random plus/minus signs per test engine and sensor, held constant across that engine trajectory.',
        'pairing':'Within each subset and seed, use identical perturbations across all three predictors; reuse one Gaussian draw across noise amplitudes.',
        'fixed':['model weights','training and validation data','calibration data','training/scoring targets','engine partition','window length and stride','input sensors','scaler'],
        'timing':{'batches':[1,64],'warmup_calls':10,'measured_calls':50,'neural_device':'cuda:0','tree_threads':1,
            'scope':'Warm model forward calls; inputs already on device; exclude loading, preprocessing, host/device transfer and final CPU conversion; synchronize CUDA around timed calls.'},
        'analysis':['All perturbation predictions and six truth/population scores','Paired engine errors averaged over three corruption realizations before bootstrap','Full model winners for the fixed configurations','Measured median/p95 batch latency and prediction throughput'],
        'caveats':['Exploratory use of known benchmark data; no field-calibrated noise model.',
            'Six fixed predictors, one partition and initialization; corruption seeds are not training replications.',
            'No noise-aware retraining, target/architecture tuning, deployment guarantee or real maintenance-cost claim.',
            'CPU and GPU timings describe these implementations and cannot isolate architecture efficiency.'],
        'training_source_sha256':sha(BASE/'train_repair.py'),'primary_spec_sha256':sha(BASE/'ANALYSIS_SPEC.md'),
        'budget_spec_sha256':sha(BASE/'budget_sensitivity/BUDGET_SPEC.json')}
    HERE.mkdir(exist_ok=True)
    if SPEC.exists():
        old=read(SPEC)
        for key in spec:
            if key!='recorded_at':
                assert old[key]==spec[key],key
        print('Existing follow-up specification retained.',flush=True)
    else:
        dump(SPEC,spec)
        print('Froze six fixed models, 72 perturbed prediction tables and latency checks.',flush=True)


def corrupted_frame(frame,columns,kind,amplitude,seed):
    result=frame.copy()
    rng=np.random.default_rng(seed)
    if kind=='gaussian':
        shift=amplitude*rng.standard_normal((len(frame),len(columns)))
    elif kind=='engine_bias':
        units=np.sort(frame.unit_id.unique())
        values=amplitude*rng.choice(np.asarray([-1.,1.]),size=(len(units),len(columns)))
        shift=values[np.searchsorted(units,frame.unit_id.to_numpy())]
    else:
        raise ValueError(kind)
    result.loc[:,columns]=result.loc[:,columns].to_numpy(float)+shift
    signature=hashlib.sha256(shift.astype('<f8').tobytes()).hexdigest()
    return result,{'perturbation_sha256':signature,'sensor_observation_rows':len(frame),'sensors':len(columns),
        'realized_rms_shift':float(np.sqrt(np.mean(shift**2))),'maximum_absolute_shift':float(np.max(np.abs(shift)))}


def timed_calls(model,cell,features):
    rows=[]; samples=[]
    for batch in (1,64):
        x=features[:batch]
        if cell.model=='lightgbm':
            x=x.reshape(batch,-1)
            forward=lambda:model.predict(x,num_threads=1)
            sync=lambda:None
        else:
            x=torch.from_numpy(x).to('cuda:0')
            forward=lambda:model(x)
            sync=lambda:torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(10):
                forward()
            sync()
            times=[]
            for index in range(50):
                sync(); start=time.perf_counter_ns()
                forward(); sync()
                elapsed=(time.perf_counter_ns()-start)/1e6
                times.append(elapsed)
                samples.append(dict(batch=batch,repeat=index+1,latency_ms=elapsed))
        rows.append(dict(batch=batch,repeats=50,median_batch_ms=float(np.median(times)),
            p95_batch_ms=float(np.quantile(times,.95)),mean_batch_ms=float(np.mean(times)),
            predictions_per_second_at_median=float(batch*1000/np.median(times))))
    return rows,samples


def execute(entry):
    cell=training.Cell(**entry['cell'])
    source=BASE/entry['source_directory']
    assert sha(source/'meta.json')==entry['source_meta_sha256']
    assert sha(source/'preds_test.parquet')==entry['clean_test_sha256']
    model_file=source/('model.txt' if cell.model=='lightgbm' else 'model.pt')
    assert sha(model_file)==entry['model_sha256']
    out=HERE/'model_checks'/cell.run_id
    if (out/'meta.json').exists():
        previous=read(out/'meta.json')
        assert previous['status']=='completed','Inspect incomplete follow-up before retry.'
        assert previous['script_sha256']==sha(__file__) and previous['spec_sha256']==sha(SPEC)
        print('REUSE '+cell.run_id,flush=True); return
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(run_id=cell.run_id,**entry,started_at=datetime.now(timezone.utc).isoformat(),status='running',
        script_sha256=sha(__file__),spec_sha256=sha(SPEC),training_performed=False)
    dump(out/'meta.json',meta)
    try:
        windows,frames,split,scaler=training.prepare(cell)
        columns=training.sensor_columns(cell.sensors)
        normalized=training.transform_features(frames['test'],scaler,columns)
        rebuilt=training._windows_with_short_endpoint_support(normalized,columns,window_size=30,stride=1)
        assert np.array_equal(rebuilt.features,windows['test'].features)
        assert split==read(source/'split.json')
        with np.load(source/'scaler.npz') as expected:
            for name in ('min_','scale_','data_min_','data_max_'):
                assert np.array_equal(getattr(scaler,name),expected[name])
        source_meta=read(source/'meta.json')
        for name,digest in source_meta['data_sha256'].items():
            assert sha(training.REPO/'data/interim/cmapss'/name)==digest
        if cell.model=='lightgbm':
            model=lgb.Booster(model_file=str(model_file))
            predict=lambda x:model.predict(x.reshape(len(x),-1),num_threads=1)
            complexity={'trees':model.num_trees(),'features':model.num_feature()}
        else:
            training.set_deterministic_seed(cell.init_seed)
            model=training.build_neural_model(cell.model,len(columns)).to('cuda:0')
            model.load_state_dict(torch.load(model_file,map_location='cpu',weights_only=True)); model.eval()
            predict=lambda x:training.predict_neural(model,x,batch_size=4096).astype(float)*cell.target_scale
            complexity={'parameters':sum(p.numel() for p in model.parameters()),'features':len(columns)}
        clean=pd.read_parquet(source/'preds_test.parquet')
        actual=predict(rebuilt.features)
        baseline_error=float(np.max(np.abs(actual-clean.pred_rul.to_numpy())))
        assert baseline_error<1e-7,(cell.run_id,baseline_error)
        latency,samples=timed_calls(model,cell,rebuilt.features)
        pd.DataFrame(samples).to_csv(out/'latency_samples.csv',index=False)
        dump(out/'latency_summary.json',latency)
        print('CLEAN AND LATENCY VERIFIED '+cell.run_id,flush=True)
        records=[]
        for kind,amplitude in SCENARIOS:
            for seed in SEEDS:
                identifier=f'{kind}__a{amplitude:g}__seed{seed}'
                perturbed,audit=corrupted_frame(normalized,columns,kind,amplitude,seed)
                current=training._windows_with_short_endpoint_support(perturbed,columns,window_size=30,stride=1)
                assert np.array_equal(current.unit_ids,rebuilt.unit_ids) and np.array_equal(current.cycles,rebuilt.cycles)
                assert np.array_equal(current.labels,rebuilt.labels)
                prediction=predict(current.features)
                frame=clean.copy()
                frame['pred_rul']=prediction
                assert np.isfinite(frame.to_numpy()).all()
                file=out/f'preds_test__{identifier}.parquet'
                frame.to_parquet(file,index=False)
                record=dict(scenario=identifier,kind=kind,amplitude=amplitude,seed=seed,prediction_file=file.name,
                    rows=len(frame),prediction_sha256=sha(file),**audit)
                records.append(record)
                print('PERTURBED '+cell.run_id+' '+identifier,flush=True)
        meta.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),
            clean_prediction_max_abs_error=baseline_error,complexity=complexity,
            checkpoint_bytes=model_file.stat().st_size,latency=latency,perturbations=records,
            data_sha256=source_meta['data_sha256'],platform=platform.platform(),processor=platform.processor(),
            torch_version=torch.__version__,lightgbm_version=lgb.__version__,cuda=torch.version.cuda,
            neural_gpu=torch.cuda.get_device_name(0),tree_threads=1)
        dump(out/'meta.json',meta)
        del model,windows,frames,rebuilt
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        dump(out/'meta.json',meta)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.freeze:
        freeze(); return
    spec=read(SPEC)
    assert spec['training_source_sha256']==sha(BASE/'train_repair.py')
    assert spec['primary_spec_sha256']==sha(BASE/'ANALYSIS_SPEC.md')
    assert spec['budget_spec_sha256']==sha(BASE/'budget_sensitivity/BUDGET_SPEC.json')
    assert torch.cuda.is_available()
    for entry in spec['models']:
        execute(entry)
    print('ALL SIX FIXED-MODEL REVIEW CHECKS COMPLETE',flush=True)


if __name__=='__main__':
    main()
