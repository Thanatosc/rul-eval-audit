"""Post hoc accounting of endpoint error changes after the frozen sensor checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def endpoint(frame):
    return frame.sort_values(['unit_id','cycle']).drop_duplicates('unit_id',keep='last').set_index('unit_id')


def main():
    rows=[]; largest=0.
    for entry in read(HERE/'REVIEW_CHECK_SPEC.json')['models']:
        cell=entry['cell']; source=BASE/entry['source_directory']
        name=entry['source_directory'].rsplit('/',1)[1]
        clean=endpoint(pd.read_parquet(source/'preds_test.parquet'))
        y=np.minimum(clean.raw_rul.to_numpy(float),125.)
        original_error=clean.pred_rul.to_numpy()-y
        for record in read(HERE/'model_checks'/name/'meta.json')['perturbations']:
            changed=endpoint(pd.read_parquet(HERE/'model_checks'/name/record['prediction_file']))
            assert clean.index.equals(changed.index)
            displacement=changed.pred_rul.to_numpy()-clean.pred_rul.to_numpy()
            cross=2*original_error*displacement
            squared=displacement**2
            error=((changed.pred_rul.to_numpy()-y)**2-original_error**2)-cross-squared
            largest=max(largest,float(np.max(np.abs(error))))
            assert np.max(np.abs(error))<1e-8
            rows.append(dict(subset=cell['subset'],model=cell['model'],kind=record['kind'],amplitude=record['amplitude'],
                seed=record['seed'],engines=len(clean),mean_prediction_displacement=float(displacement.mean()),
                prediction_displacement_mse=float(squared.mean()),error_cross_term=float(cross.mean()),
                endpoint_mse_increase=float(cross.mean()+squared.mean())))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'analysis/endpoint_displacement_realizations.csv',index=False)
    combined=frame.groupby(['subset','model','kind','amplitude'],as_index=False)[
        ['mean_prediction_displacement','prediction_displacement_mse','error_cross_term','endpoint_mse_increase']].mean()
    combined['rms_prediction_displacement']=np.sqrt(combined.prediction_displacement_mse)
    combined.to_csv(HERE/'analysis/endpoint_displacement_summary.csv',index=False)
    report={'status':'verified','design_status':'Post hoc descriptive attribution after inspecting the completed frozen perturbation checks.',
        'identity':'MSE_perturbed - MSE_clean = 2 mean(clean_error * prediction_displacement) + mean(prediction_displacement^2)',
        'maximum_per_engine_identity_error':largest,'realization_rows':len(frame),'summary_rows':len(combined),
        'scope':'Exact prediction/error accounting; no isolation of internal learning dynamics or proof of why an architecture is sensitive.',
        'source_score_sha256':hashlib.sha256((HERE/'analysis/paired_sensitivity.csv').read_bytes()).hexdigest()}
    (HERE/'analysis/SENSOR_EXPLANATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(combined[(combined.subset=='FD004')&(combined.kind=='gaussian')&(combined.amplitude==.01)].to_string(index=False))


if __name__=='__main__':
    main()
