"""Descriptive decompositions of the specified contrasts; never selects or refits models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_repair import nasa, bootstrap_indices, interval

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"


def endpoint(path):
    frame = pd.read_parquet(path / "preds_test.parquet").sort_values(["unit_id", "cycle"])
    return frame.groupby("unit_id").tail(1).set_index("unit_id").sort_index()


def main():
    status = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["completed_grid_runs"] == 96
    lookup, tails, unit_rows, populations = {}, [], [], []
    for path in sorted((HERE / "runs").iterdir()):
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "completed"
        key = (meta["subset"], meta["split_seed"], meta["model"], meta["init_seed"], meta["target"], meta["sensors"])
        lookup[key] = (meta, endpoint(path))
        if meta["arm"] != "core":
            continue
        f = lookup[key][1]
        truth = np.minimum(f.raw_rul.to_numpy(), 125)
        error = f.pred_rul.to_numpy() - truth
        cost = nasa(error)
        squared = error**2
        k = max(1, int(np.ceil(.05 * len(f))))
        order = np.argsort(cost)[::-1]
        tails.append({k: meta[k] for k in ("run_id", "subset", "split_seed", "model", "init_seed")} | {
            "engines": len(f), "largest_five_percent_n": k,
            "nasa_largest_share": float(cost.max()/cost.sum()),
            "nasa_largest_five_percent_share": float(cost[order[:k]].sum()/cost.sum()),
            "squared_error_largest_five_percent_share": float(np.sort(squared)[-k:].sum()/squared.sum()),
            "late_error_nasa_share": float(cost[error>0].sum()/cost.sum()),
            "nasa_mean": float(cost.mean()), "rmse": float(np.sqrt(squared.mean())),
            "largest_loss_engine": int(f.index[order[0]]), "largest_loss_error": float(error[order[0]])})
        for unit, e, c, s in zip(f.index, error, cost, squared):
            unit_rows.append({"run_id": meta["run_id"], "unit_id": int(unit), "error": float(e),
                              "nasa_loss": float(c), "squared_error": float(s)})
        if meta["model"] == "lightgbm" and meta["split_seed"] == 42:
            all_windows = pd.read_parquet(path / "preds_test.parquet")
            for population in ("endpoint", "equal_engine_windows", "pooled_windows"):
                frame = f.reset_index() if population == "endpoint" else all_windows
                stage = np.where(frame.raw_rul <= 30, "0-30", np.where(frame.raw_rul <= 80, "31-80", ">80"))
                for group in ("0-30", "31-80", ">80"):
                    mask = (stage == group).astype(float)
                    share = (pd.Series(mask, index=frame.index).groupby(frame.unit_id).mean().mean()
                             if population == "equal_engine_windows" else mask.mean())
                    populations.append({"subset": meta["subset"], "population": population, "stage": group,
                                        "weighted_stage_share": float(share), "records": len(frame),
                                        "engines": frame.unit_id.nunique()})
    pd.DataFrame(tails).to_csv(OUT / "endpoint_loss_concentration.csv", index=False)
    pd.DataFrame(unit_rows).to_parquet(OUT / "core_endpoint_loss_contributions.parquet", index=False)
    pd.DataFrame(populations).to_csv(OUT / "population_stage_composition.csv", index=False)
    uq = pd.read_parquet(OUT / "uq_per_engine_stage.parquet")
    measures = ("coverage", "width", "interval_score")
    for name in measures:
        uq[name+"_sum"] = uq[name]*uq.windows
    unit = uq.groupby(["subset", "split_seed", "model", "init_seed", "unit_id"])[
        [name+"_sum" for name in measures]+["windows"]].sum()
    for name in measures:
        unit[name] = unit[name+"_sum"]/unit.windows
    overall = []
    for (subset, split, model), group in unit.reset_index().groupby(["subset", "split_seed", "model"]):
        per = group.groupby("unit_id")[list(measures)].mean()
        idx = bootstrap_indices(len(per))
        row = {"subset": subset, "split_seed": split, "model": model, "engines": len(per)}
        for name in measures:
            values = per[name].to_numpy()
            lo, hi = interval(values[idx].mean(axis=1))
            row.update({name: float(values.mean()), name+"_ci_lower": lo, name+"_ci_upper": hi})
        overall.append(row)
    pd.DataFrame(overall).to_csv(OUT / "uq_overall_engine_bootstrap.csv", index=False)
    decomposition, errors = [], []
    for subset in ("FD001", "FD004"):
        for split in (42, 137, 271):
            for model in ("lstm", "lightgbm"):
                for sensors in ("common_14", "all_21"):
                    _, cap = lookup[(subset, split, model, 11, "piecewise_125", sensors)]
                    _, raw = lookup[(subset, split, model, 11, "linear_uncapped", sensors)]
                    assert raw.index.equals(cap.index)
                    assert np.array_equal(raw.raw_rul, cap.raw_rul)
                    truth_raw = raw.raw_rul.to_numpy()
                    truth_cap = np.minimum(truth_raw, 125)
                    pr, pc = raw.pred_rul.to_numpy(), cap.pred_rul.to_numpy()
                    d_cap = (pr-truth_cap)**2-(pc-truth_cap)**2
                    d_raw = (pr-truth_raw)**2-(pc-truth_raw)**2
                    algebra = -2*(truth_raw-truth_cap)*(pr-pc)
                    numerical_error = float(np.max(np.abs(d_raw-d_cap-algebra)))
                    assert numerical_error < 1e-7
                    errors.append(numerical_error)
                    late = truth_raw <= 125
                    decomposition.append({"subset": subset, "split_seed": split, "model": model, "sensors": sensors,
                        "engines": len(raw), "above_cap_engines": int((~late).sum()),
                        "mse_difference_on_capped_truth": float(d_cap.mean()),
                        "mse_difference_on_raw_truth": float(d_raw.mean()),
                        "within_cap_contribution": float(np.where(late, d_cap, 0).mean()),
                        "above_cap_contribution_capped_truth": float(np.where(~late, d_cap, 0).mean()),
                        "above_cap_contribution_raw_truth": float(np.where(~late, d_raw, 0).mean()),
                        "truth_change_mse_difference": float(algebra.mean()),
                        "identity_max_absolute_error": numerical_error})
    pd.DataFrame(decomposition).to_csv(OUT / "target_contrast_decomposition.csv", index=False)
    scored = pd.read_csv(OUT / "all_run_scores.csv")
    scored = scored[(scored.subset.isin(["FD001", "FD004"])) &
                    (scored.model.isin(["lstm", "lightgbm"])) & (scored.init_seed == 11)]
    target_population = scored.pivot(index=["subset", "split_seed", "model", "sensors", "truth", "population"],
                                     columns="target", values="rmse").reset_index()
    assert len(target_population) == 144
    target_population["raw_minus_capped_rmse"] = target_population.linear_uncapped-target_population.piecewise_125
    target_population.to_csv(OUT / "target_population_contrasts.csv", index=False)
    grouped = []
    for (truth, population), group in target_population.groupby(["truth", "population"]):
        differences = group.raw_minus_capped_rmse
        assert len(differences) == 24
        grouped.append({"truth": truth, "population": population, "contexts": len(differences),
                        "difference_min": float(differences.min()), "difference_max": float(differences.max()),
                        "raw_trained_lower_rmse": int((differences < 0).sum()),
                        "capped_trained_lower_rmse": int((differences > 0).sum())})
    pd.DataFrame(grouped).to_csv(OUT / "target_population_summary.csv", index=False)
    note = {
        "status": "complete", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identity_max_absolute_error": max(errors), "paired_target_contexts": len(decomposition),
        "interpretation": "Post-result descriptive decompositions added during writing. No model selection, new hypothesis test or causal identification.",
        "identity": "D_rawtruth - D_cappedtruth = -2 (raw_truth-capped_truth) (raw_trained_prediction-capped_trained_prediction), for squared-loss contrasts.",
        "tail_definition": "Largest ceiling(0.05*N) NASA-loss endpoints, all core fits retained."
    }
    (OUT / "EXPLANATORY_ANALYSIS.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
