"""Independent checks of weighting, target transformation, and saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_repair import unit_losses, estimate, mean_loss, boot_metric

HERE=Path(__file__).resolve().parent

def main():
    frame=pd.DataFrame(dict(unit_id=[1,1,2],cycle=[30,31,30],raw_rul=[150.,149.,10.],pred_rul=[100.,120.,30.]))
    expected={
        ("capped125","endpoint"): (25.+400.)/2,
        ("capped125","equal_engine_windows"): ((625.+25.)/2+400.)/2,
        ("capped125","pooled_windows"): (625.+25.+400.)/3,
        ("raw","endpoint"): (29.**2+20.**2)/2,
        ("raw","equal_engine_windows"): ((50.**2+29.**2)/2+20.**2)/2,
        ("raw","pooled_windows"): (50.**2+29.**2+20.**2)/3}
    for (truth,pop),mse in expected.items():
        loss=unit_losses(frame,truth,pop)
        assert np.isclose(estimate(loss)["rmse"]**2,mse)
        # Duplicating all windows from unit 1 changes pooled weighting only;
        # engine-equal and endpoint values remain unchanged.
        duplicated=pd.concat([frame,frame[frame.unit_id==1]],ignore_index=True)
        if pop!="pooled_windows":
            assert np.isclose(estimate(unit_losses(duplicated,truth,pop))["rmse"]**2,mse)
        draw,_=boot_metric(loss)
        assert draw.min()>=np.sqrt((loss.se/loss['count']).min())-1e-10
        assert draw.max()<=np.sqrt((loss.se/loss['count']).max())+1e-10
    first=unit_losses(frame,"capped125","endpoint")
    assert np.allclose(mean_loss([first,first]).to_numpy(),first.to_numpy())

    # Independent direct NumPy endpoint recomputation of both diagnostic runs.
    diagnostics=[]
    for file in (HERE/'diagnostics').glob('*/meta.json'):
        meta=json.loads(file.read_text(encoding='utf-8'))
        df=pd.read_parquet(file.parent/'preds_test.parquet')
        last=df.sort_values(['unit_id','cycle']).drop_duplicates('unit_id',keep='last')
        actual=float(np.sqrt(np.mean((last.pred_rul-np.minimum(last.raw_rul,125.))**2)))
        assert np.isclose(actual,meta['endpoint_rmse_native'],atol=1e-10)
        diagnostics.append(dict(scale=meta['target_scale'],rmse=actual,engines=len(last)))
    assert len(diagnostics)==2
    output=dict(weighting_and_truth_cases=len(expected),diagnostics=diagnostics,passed=True)
    (HERE/'ANALYSIS_VERIFICATION.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
