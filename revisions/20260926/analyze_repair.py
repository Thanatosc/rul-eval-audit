"""Engine-level analysis of the corrective grid; no model fitting."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "rul-eval-audit"
TRUTHS = ("capped125", "raw")
POPS = ("endpoint", "equal_engine_windows", "pooled_windows")
BOOT = 5000
INDICES = {}


def bootstrap_indices(n):
    if n not in INDICES:
        INDICES[n] = np.random.default_rng(20260926).integers(0, n, size=(BOOT, n), dtype=np.int32)
    return INDICES[n]


def nasa(error):
    out = np.empty_like(error, dtype=float)
    neg = error < 0
    out[neg] = np.expm1(-error[neg] / 13.)
    out[~neg] = np.expm1(error[~neg] / 10.)
    return out


def unit_losses(frame, truth, population):
    f = frame.sort_values(["unit_id", "cycle"]).copy()
    if population == "endpoint":
        f = f.groupby("unit_id", sort=True).tail(1)
    y = f.raw_rul.to_numpy(float)
    if truth == "capped125":
        y = np.minimum(y, 125.)
    error = f.pred_rul.to_numpy(float) - y
    f["se"], f["nasa"] = error**2, nasa(error)
    grouped = f.groupby("unit_id", sort=True).agg(se=("se", "sum"), nasa=("nasa", "sum"), count=("se", "size"))
    if population != "pooled_windows":
        grouped[["se", "nasa"]] = grouped[["se", "nasa"]].div(grouped["count"], axis=0)
        grouped["count"] = 1.
    assert np.isfinite(grouped.to_numpy()).all()
    return grouped


def estimate(loss):
    return dict(rmse=float(np.sqrt(loss.se.sum() / loss["count"].sum())),
                nasa_mean=float(loss.nasa.sum() / loss["count"].sum()), engines=len(loss))


def mean_loss(items):
    first = items[0]
    assert all(first.index.equals(x.index) for x in items)
    return pd.DataFrame(np.mean([x.to_numpy(float) for x in items], axis=0), index=first.index, columns=first.columns)


def boot_metric(loss):
    indices = bootstrap_indices(len(loss))
    se, cost, den = loss.se.to_numpy(), loss.nasa.to_numpy(), loss["count"].to_numpy()
    sampled_den = den[indices].sum(axis=1)
    return np.sqrt(se[indices].sum(axis=1) / sampled_den), cost[indices].sum(axis=1) / sampled_den


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def residual_uq(meta, frame, path):
    ca = pd.read_parquet(path / "preds_calib.parquet")
    yca = np.minimum(ca.raw_rul.to_numpy(float), 125.)
    score = np.abs(ca.pred_rul.to_numpy(float) - yca)
    order = min(int(np.ceil((len(score) + 1) * .9)), len(score))
    qhat = float(np.partition(score, order-1)[order-1])
    f = frame.copy()
    y = np.minimum(f.raw_rul.to_numpy(float), 125.)
    lower, upper = f.pred_rul.to_numpy() - qhat, f.pred_rul.to_numpy() + qhat
    f["coverage"] = ((y >= lower) & (y <= upper)).astype(float)
    f["width"] = upper - lower
    f["interval_score"] = upper-lower+20*np.maximum(lower-y, 0)+20*np.maximum(y-upper, 0)
    f["stage"] = np.where(f.raw_rul <= 30, "0-30", np.where(f.raw_rul <= 80, "31-80", ">80"))
    unit = f.groupby(["stage", "unit_id"], sort=True).agg(coverage=("coverage", "mean"), width=("width", "mean"),
              interval_score=("interval_score", "mean"), windows=("cycle", "size")).reset_index()
    for key in ("subset", "split_seed", "model", "init_seed", "run_id"):
        unit[key] = meta[key]
    unit["qhat"] = qhat
    unit["calibration_engines"] = ca.unit_id.nunique()
    unit["calibration_windows"] = len(ca)
    return unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    out = HERE / ("interim_analysis" if args.allow_partial else "analysis")
    out.mkdir(exist_ok=True)
    metas = []
    for file in sorted((HERE / "runs").glob("*/meta.json")):
        try:
            m = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if args.allow_partial:
                continue
            raise
        if m["status"] == "completed":
            metas.append((m, file.parent))
    if not args.allow_partial and len(metas) != 96:
        raise RuntimeError(f"Expected all 96 completed runs, found {len(metas)}")
    scores, training, losses, grouped_core, uq, historical_pairs = [], [], {}, defaultdict(list), [], []
    lookup = {}
    for m, path in metas:
        frame = pd.read_parquet(path / "preds_test.parquet")
        expected = {"FD001":100,"FD002":259,"FD003":100,"FD004":248}[m["subset"]]
        assert frame.unit_id.nunique() == expected
        assert not frame.duplicated(["unit_id", "cycle"]).any()
        curve = pd.read_csv(path / "training_curve.csv")
        assert int(curve.selected.sum()) == 1
        meta = {k:m[k] for k in ("run_id", "arm", "subset", "split_seed", "model", "init_seed", "target", "sensors")}
        training.append({**meta, **m["training"], "test_prediction_range":m["prediction_range"]})
        key = (m["subset"], m["split_seed"], m["model"], m["init_seed"], m["target"], m["sensors"])
        assert key not in lookup
        lookup[key] = m["run_id"]
        for truth in TRUTHS:
            for pop in POPS:
                loss = unit_losses(frame, truth, pop)
                losses[(m["run_id"], truth, pop)] = loss
                scores.append({**meta, "truth":truth, "population":pop, **estimate(loss)})
                if m["arm"] == "core":
                    grouped_core[(m["subset"], m["split_seed"], m["model"], truth, pop)].append(loss)
        if m["arm"] == "core":
            uq.append(residual_uq(m, frame, path))
            if m["split_seed"] == 42:
                historic_path=REPO / f"results/runs/unified_v1__{m['subset'].lower()}__{m['model']}__seed{m['init_seed']}__point/preds_test.parquet"
                if historic_path.exists():
                    old=pd.read_parquet(historic_path)
                    old=old.merge(frame[["unit_id","cycle","raw_rul"]],on=["unit_id","cycle"],validate="one_to_one")
                    assert len(old)==len(frame)
                    old_loss=unit_losses(old,"capped125","endpoint")
                    new_loss=losses[(m["run_id"],"capped125","endpoint")]
                    assert old_loss.index.equals(new_loss.index)
                    bo,_=boot_metric(old_loss); bn,_=boot_metric(new_loss)
                    ci=interval(bn-bo)
                    eo,en=estimate(old_loss),estimate(new_loss)
                    historical_pairs.append({**meta,"historical_rmse":eo["rmse"],"repaired_rmse":en["rmse"],
                        "difference":en["rmse"]-eo["rmse"],"difference_ci_lower":ci[0],"difference_ci_upper":ci[1]})
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(out / "all_run_scores.csv", index=False)
    training_frame = pd.DataFrame(training)
    training_frame.to_csv(out / "training_quality.csv", index=False)
    pd.DataFrame(historical_pairs).to_csv(out / "repaired_vs_historical_baselines.csv",index=False)

    core_rows = []
    for (subset, split, model, truth, pop), values in grouped_core.items():
        expected = 1 if model == "lightgbm" else 2
        if len(values) != expected:
            continue
        loss = mean_loss(values)
        rmse, nc = boot_metric(loss)
        ri, ni = interval(rmse), interval(nc)
        core_rows.append(dict(subset=subset, split_seed=split, model=model, truth=truth, population=pop,
            initialization_records=len(values), **estimate(loss), rmse_ci_lower=ri[0], rmse_ci_upper=ri[1],
            nasa_mean_ci_lower=ni[0], nasa_mean_ci_upper=ni[1],
            initialization_rmse_min=min(estimate(v)["rmse"] for v in values),
            initialization_rmse_max=max(estimate(v)["rmse"] for v in values)))
    core = pd.DataFrame(core_rows)
    core.to_csv(out / "core_engine_bootstrap.csv", index=False)

    contrast = []
    for subset in ("FD001", "FD004"):
        for split in (42, 137, 271):
            for model in ("lstm", "lightgbm"):
                for kind, fixed_values in (("training_target", ("common_14", "all_21")),
                                           ("sensor_input", ("piecewise_125", "linear_uncapped"))):
                    for fixed in fixed_values:
                        if kind == "training_target":
                            ka = (subset,split,model,11,"piecewise_125",fixed)
                            kb = (subset,split,model,11,"linear_uncapped",fixed)
                            definition = "raw_trained_minus_capped_trained"
                        else:
                            ka = (subset,split,model,11,fixed,"common_14")
                            kb = (subset,split,model,11,fixed,"all_21")
                            definition = "all21_minus_common14"
                        if ka not in lookup or kb not in lookup:
                            continue
                        for truth in TRUTHS:
                            a, b = losses[(lookup[ka],truth,"endpoint")], losses[(lookup[kb],truth,"endpoint")]
                            assert a.index.equals(b.index)
                            ra, na = boot_metric(a)
                            rb, nb = boot_metric(b)
                            ri, ni = interval(rb-ra), interval(nb-na)
                            ea, eb = estimate(a), estimate(b)
                            contrast.append(dict(subset=subset,split_seed=split,model=model,init_seed=11,
                                factor=kind,fixed_level=fixed,truth=truth,direction=definition,
                                rmse_difference=eb["rmse"]-ea["rmse"],rmse_ci_lower=ri[0],rmse_ci_upper=ri[1],
                                nasa_mean_difference=eb["nasa_mean"]-ea["nasa_mean"],
                                nasa_mean_ci_lower=ni[0],nasa_mean_ci_upper=ni[1],engines=len(a)))
    contrasts = pd.DataFrame(contrast)
    contrasts.to_csv(out / "paired_factor_contrasts.csv", index=False)

    rankings = []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for split in (42, 137, 271):
            for seed in (11, 23):
                for truth in TRUTHS:
                    for pop in POPS:
                        available = []
                        for model in ("lstm", "cnn_1d", "lightgbm"):
                            key=(subset,split,model,11 if model=="lightgbm" else seed,"piecewise_125","common_14")
                            if key in lookup:
                                available.append((model,estimate(losses[(lookup[key],truth,pop)])))
                        if len(available) != 3:
                            continue
                        rm = min(v["rmse"] for _,v in available)
                        ns = min(v["nasa_mean"] for _,v in available)
                        wr = {model for model,v in available if abs(v["rmse"]-rm) < 1e-10}
                        wn = {model for model,v in available if abs(v["nasa_mean"]-ns) < 1e-10}
                        rankings.append(dict(subset=subset,split_seed=split,init_seed=seed,truth=truth,population=pop,
                            rmse_winner="|".join(sorted(wr)),nasa_winner="|".join(sorted(wn)),conflict=not bool(wr & wn)))
    rank = pd.DataFrame(rankings)
    rank.to_csv(out / "metric_winner_contexts.csv", index=False)

    uq_rows = []
    if uq:
        qu = pd.concat(uq, ignore_index=True)
        qu.to_parquet(out / "uq_per_engine_stage.parquet", index=False)
        for (subset,split,model,stage), group in qu.groupby(["subset","split_seed","model","stage"]):
            expected = 1 if model == "lightgbm" else 2
            if group.init_seed.nunique() != expected:
                continue
            per = group.groupby("unit_id")[["coverage","width","interval_score","windows"]].mean()
            idx = bootstrap_indices(len(per))
            row=dict(subset=subset,split_seed=split,model=model,stage=stage,engines=len(per),
                windows=int(per.windows.sum()),calibration_engines=int(group.calibration_engines.iloc[0]),
                calibration_windows=int(group.calibration_windows.iloc[0]),
                initialization_records=expected)
            for col in ("coverage","width","interval_score"):
                values=per[col].to_numpy()
                ci=interval(values[idx].mean(axis=1))
                row.update({col:float(values.mean()),col+"_ci_lower":ci[0],col+"_ci_upper":ci[1]})
            uq_rows.append(row)
    uq_table=pd.DataFrame(uq_rows)
    uq_table.to_csv(out / "uq_stage_engine_bootstrap.csv",index=False)
    # Preserve the complete historical common-truth table without reusing its
    # seed-labelled repetitions for statistical inference.
    historical=pd.read_csv(REPO / "results/common_truth/COMMON_TRUTH_RUN_METRICS.csv")
    historical.to_csv(out / "historical_common_truth_all_runs.csv",index=False)
    pd.read_csv(REPO / "results/common_truth/COMMON_TRUTH_LABEL_CONTRASTS.csv").to_csv(
        out / "historical_common_truth_descriptive_contrasts.csv",index=False)

    primary_rank = rank[(rank.truth=="capped125")&(rank.population=="endpoint")] if len(rank) else rank
    summary=dict(status="interim" if args.allow_partial else "complete",completed_grid_runs=len(metas),
        planned_grid_runs=96,score_rows=len(scores),core_summary_rows=len(core),factor_contrast_rows=len(contrasts),
        primary_winner_contexts=len(primary_rank),
        primary_metric_winner_conflicts=int(primary_rank.conflict.sum()) if len(primary_rank) else 0,
        bootstrap_replicates=BOOT,uq_stage_rows=len(uq_table),
        cap_with_recent_best_count=int(training_frame.cap_with_recent_best.sum()) if len(training_frame) else 0,
        cautions=["Post-rejection exploratory analyses on previously examined benchmark data.",
                  "Bootstrap samples engines conditional on fitted models and calibration scores.",
                  "Repeated engine splits and initialization contexts are not independent devices.",
                  "Tree models are stored once; reused only as a reference when listing seed-specific rankings."])
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
