"""Corrective experiments for the transfer revision; historical assets are inputs."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "rul-eval-audit"
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
from rul_audit.models.baselines import build_neural_model, set_deterministic_seed, predict_neural
from rul_audit.data.cmapss import (load_subset, apply_rul_label, create_unit_split,
    split_frame_by_unit, sensor_columns, fit_train_minmax, transform_features)
from rul_audit.experiments.kill_test import _windows_with_short_endpoint_support
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb


@dataclass(frozen=True)
class Cell:
    subset: str
    split_seed: int
    model: str
    init_seed: int
    target: str = "piecewise_125"
    sensors: str = "common_14"
    arm: str = "core"
    target_scale: float = 125.0

    @property
    def run_id(self):
        return (f"{self.arm}__{self.subset.lower()}__s{self.split_seed}__{self.model}"
                f"__i{self.init_seed}__{self.target}__{self.sensors}__scale{self.target_scale:g}")


def cells(arm):
    if arm == "diagnostic":
        return [Cell("FD002", 42, "lstm", 11, arm=arm, target_scale=s) for s in (1., 125.)]
    result = []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for split in (42, 137, 271):
            for model, seed in (("lstm", 11), ("lstm", 23), ("cnn_1d", 11), ("cnn_1d", 23), ("lightgbm", 11)):
                result.append(Cell(subset, split, model, seed))
    for subset in ("FD001", "FD004"):
        for split in (42, 137, 271):
            for target, sensors in (("linear_uncapped", "common_14"),
                                    ("linear_uncapped", "all_21"), ("piecewise_125", "all_21")):
                for model in ("lstm", "lightgbm"):
                    result.append(Cell(subset, split, model, 11, target, sensors, "factor"))
    assert len(result) == 96 and len({c.run_id for c in result}) == 96
    return result


def prepare(cell):
    source = load_subset(REPO / "data/interim/cmapss", cell.subset)
    split = create_unit_split(source.train.unit_id.unique().tolist(), cell.split_seed)
    frames = split_frame_by_unit(apply_rul_label(source.train, cell.target), split)
    frames["test"] = apply_rul_label(source.test, cell.target)
    selected = sensor_columns(cell.sensors)
    scaler = fit_train_minmax(frames["train"], selected)
    windows = {key: _windows_with_short_endpoint_support(
        transform_features(frame, scaler, selected), selected, window_size=30, stride=1)
        for key, frame in frames.items()}
    for a, b in (("train", "val"), ("train", "calib"), ("val", "calib")):
        assert not (set(split[a]) & set(split[b]))
    return windows, frames, split, scaler


def neural_fit(cell, windows, out):
    set_deterministic_seed(cell.init_seed)
    torch.cuda.set_device(0)
    model = build_neural_model(cell.model, windows["train"].features.shape[-1]).to("cuda:0")
    scale = cell.target_scale
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    diagnostic = cell.arm == "diagnostic"
    scheduler = None if diagnostic else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=5, min_lr=1e-5)
    generator = torch.Generator().manual_seed(cell.init_seed)
    dataset = TensorDataset(torch.from_numpy(windows["train"].features),
                            torch.as_tensor(windows["train"].labels / scale, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator, pin_memory=True)
    best, best_epoch, best_state = float("inf"), 0, None
    stale, history, epoch = 0, [], 0
    budget, patience = (50, 8) if diagnostic else (150, 15)
    extended = False
    start = time.perf_counter()
    while epoch < budget:
        epoch += 1
        model.train()
        loss_total = 0.
        for bx, by in loader:
            bx, by = bx.to("cuda:0"), by.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bx).squeeze(1)
            loss = torch.nn.functional.mse_loss(prediction, by)
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(by)
        vp = predict_neural(model, windows["val"].features, batch_size=4096).astype(float) * scale
        val_rmse = float(np.sqrt(np.mean((vp - windows["val"].labels) ** 2)))
        improved = val_rmse < best - 1e-9
        if improved:
            best, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_batch_mse_cycles=loss_total / len(dataset) * scale**2,
                            val_rmse_cycles=val_rmse, best_val_rmse_cycles=best,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            elapsed_seconds=time.perf_counter() - start, selected=False))
        if scheduler:
            scheduler.step(val_rmse)
        # Flush progress and curve so an interrupted job remains diagnosable.
        pd.DataFrame(history).to_csv(out / "training_curve.csv", index=False)
        if epoch % 10 == 0:
            print(f"{cell.run_id} epoch={epoch} val={val_rmse:.4f} best={best:.4f}", flush=True)
        if stale >= patience and (diagnostic or epoch >= 20):
            break
        if epoch == 150 and not diagnostic and best_epoch > 135:
            if history[-16]["best_val_rmse_cycles"] - best >= .1:
                budget, extended = 300, True
                print(f"{cell.run_id} continuing predeclared budget extension", flush=True)
    if best_state is None:
        raise ArithmeticError("No valid checkpoint")
    history[best_epoch - 1]["selected"] = True
    pd.DataFrame(history).to_csv(out / "training_curve.csv", index=False)
    torch.save(best_state, out / "model.pt")
    model.load_state_dict(best_state)
    predicted = {name: predict_neural(model, data.features, batch_size=4096).astype(float) * scale
                 for name, data in windows.items() if name != "train"}
    audit = dict(best_val_rmse=best, best_epoch=best_epoch, epochs_completed=epoch,
                 max_epochs_used=budget, extended=extended, stopped_early=epoch < budget,
                 cap_with_recent_best=epoch == budget and best_epoch > budget - 15,
                 parameters=sum(p.numel() for p in model.parameters()), target_scale=scale,
                 device="cuda:0", training_seconds=time.perf_counter() - start)
    del model, optimizer, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()
    return predicted, audit


def tree_fit(cell, windows, out):
    x = {key: value.features.reshape(len(value.labels), -1) for key, value in windows.items()}
    model = lgb.LGBMRegressor(objective="regression", n_estimators=1500,
        learning_rate=.05, num_leaves=31, subsample=1., colsample_bytree=1.,
        reg_lambda=0., random_state=cell.init_seed, n_jobs=1, deterministic=True,
        force_row_wise=True, verbosity=-1)
    history = {}
    start = time.perf_counter()
    model.fit(x["train"], windows["train"].labels,
              eval_set=[(x["val"], windows["val"].labels)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False),
                         lgb.record_evaluation(history)])
    curve = pd.DataFrame(history["valid_0"])
    curve["iteration"] = np.arange(1, len(curve) + 1)
    curve["selected"] = curve.iteration == model.best_iteration_
    curve.to_csv(out / "training_curve.csv", index=False)
    model.booster_.save_model(str(out / "model.txt"), num_iteration=model.best_iteration_)
    pred = {key: model.predict(values, num_iteration=model.best_iteration_)
            for key, values in x.items() if key != "train"}
    audit = dict(best_iteration=model.best_iteration_, iterations_evaluated=len(curve),
                 best_val_rmse=float(model.best_score_["valid_0"]["rmse"]),
                 cap_with_recent_best=len(curve) >= 1500 and model.best_iteration_ > 1450,
                 device="cpu", training_seconds=time.perf_counter() - start,
                 deterministic_single_record=True)
    return pred, audit


def execute(cell, overwrite_incomplete=False):
    root = HERE / ("diagnostics" if cell.arm == "diagnostic" else "runs")
    out = root / cell.run_id
    meta_file = out / "meta.json"
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        if old.get("status") == "completed":
            return old
        if not overwrite_incomplete:
            raise RuntimeError(f"Incomplete run requires inspection: {out}")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(**asdict(cell), run_id=cell.run_id, status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                specification_sha256=hashlib.sha256((HERE / "ANALYSIS_SPEC.md").read_bytes()).hexdigest(),
                torch_version=torch.__version__, cuda=torch.version.cuda)
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        data, frames, partitions, scaler = prepare(cell)
        (out / "split.json").write_text(json.dumps(partitions, indent=2), encoding="utf-8")
        np.savez(out / "scaler.npz", min_=scaler.min_, scale_=scaler.scale_,
                 data_min_=scaler.data_min_, data_max_=scaler.data_max_)
        pred, audit = tree_fit(cell, data, out) if cell.model == "lightgbm" else neural_fit(cell, data, out)
        for key, prediction in pred.items():
            w = data[key]
            lookup = frames[key].set_index(["unit_id", "cycle"]).raw_rul
            raw = lookup.reindex(pd.MultiIndex.from_arrays([w.unit_ids, w.cycles])).to_numpy()
            result = pd.DataFrame(dict(unit_id=w.unit_ids, cycle=w.cycles, raw_rul=raw,
                                       true_rul=w.labels, pred_rul=prediction))
            assert len(result) == len(prediction) and np.isfinite(result.to_numpy()).all()
            assert not result.duplicated(["unit_id", "cycle"]).any()
            result.to_parquet(out / f"preds_{key}.parquet", index=False)
        validation = pred["val"]
        assert np.isclose(np.sqrt(np.mean((validation - data["val"].labels)**2)), audit["best_val_rmse"], rtol=1e-5, atol=1e-4)
        te = pd.read_parquet(out / "preds_test.parquet").sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)
        meta.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                    training=audit, units={k: len(v.unit_ids) for k, v in data.items()},
                    official_test_engines=int(te.unit_id.nunique()),
                    endpoint_rmse_native=float(np.sqrt(np.mean((te.pred_rul-te.true_rul)**2))),
                    prediction_range=float(np.ptp(pred["test"])))
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"COMPLETED {cell.run_id} val={audit['best_val_rmse']:.4f} test={meta['endpoint_rmse_native']:.4f}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["diagnostic", "grid"], default="grid")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", choices=["lstm", "cnn_1d", "lightgbm"])
    ap.add_argument("--retry-inspected", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This revision specification requires the available CUDA device")
    selected = cells(args.arm)
    if args.model:
        selected = [c for c in selected if c.model == args.model]
    if args.limit:
        selected = selected[:args.limit]
    print(f"Executing {len(selected)} {args.arm} cells (completed cells resume by identity)", flush=True)
    for index, cell in enumerate(selected, 1):
        execute(cell, args.retry_inspected)
        print(f"PROGRESS {index}/{len(selected)}", flush=True)
    print("REQUESTED CELLS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
