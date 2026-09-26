"""Publication figures from saved corrective results; no fitting or data selection."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd

OUT = HERE / "figures"
ANALYSIS = HERE / "analysis"
LABELS = {"lstm": "LSTM", "cnn_1d": "1D CNN", "lightgbm": "LightGBM"}
COLORS = {"lstm": "#0072B2", "cnn_1d": "#D55E00", "lightgbm": "#009E73"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
SOURCES = set()


def read_csv(name):
    file = ANALYSIS / name
    SOURCES.add(file)
    return pd.read_csv(file)


def save(fig, number):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure_{number}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagnostic():
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    for j, (scale, label, color) in enumerate(((1, "Target in cycles", "#D55E00"),
                                              (125, "Target / 125", "#0072B2"))):
        path = HERE / "diagnostics" / f"diagnostic__fd002__s42__lstm__i11__piecewise_125__common_14__scale{scale}"
        for filename in ("meta.json", "training_curve.csv", "preds_test.parquet"):
            SOURCES.add(path / filename)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        curve = pd.read_csv(path / "training_curve.csv")
        axes[0].plot(curve.epoch, curve.val_rmse_cycles, color=color, label=label, linewidth=1.4)
        best = curve.loc[curve.selected].iloc[0]
        axes[0].scatter([best.epoch], [best.val_rmse_cycles], color=color, edgecolors="white", s=40, zorder=4)
        test = pd.read_parquet(path / "preds_test.parquet").sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)
        truth = np.minimum(test.raw_rul.to_numpy(float), 125)
        ax = axes[j+1]
        ax.scatter(truth, test.pred_rul, s=13, alpha=.62, color=color, edgecolors="none")
        ax.plot([0, 150], [0, 150], color="0.4", linestyle="--", linewidth=1)
        ax.set(xlim=(-4, 155), ylim=(-4, 155), xlabel="Capped endpoint truth (cycles)",
               ylabel="Predicted RUL (cycles)", title=f"{'bc'[j]}  {label}")
        ax.text(.04, .95, f"RMSE {meta['endpoint_rmse_native']:.2f}\n259 engines",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Epoch", ylabel="Validation RMSE (cycles)", title="a  Original 50-epoch recipe", xlim=(1, 50))
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(axis="y", alpha=.2)
    fig.tight_layout(w_pad=1.8)
    save(fig, 1)


def target_contrasts():
    df = read_csv("paired_factor_contrasts.csv")
    df = df[df.factor == "training_target"]
    contexts = [(s, m, inputs) for s in ("FD001", "FD004") for m in ("lstm", "lightgbm")
                for inputs in ("common_14", "all_21")]
    labels = [f"{s}  {LABELS[m]}  {'14' if inputs == 'common_14' else '21'} sensors" for s, m, inputs in contexts]
    split_style = [(42, -.21, "#0072B2", "o"), (137, 0, "#D55E00", "s"), (271, .21, "#009E73", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.4), sharey=True)
    for ax, truth, title in zip(axes, ("capped125", "raw"), ("a  Common capped-125 truth", "b  Common raw-RUL truth")):
        for split, offset, color, marker in split_style:
            selected = []
            for subset, model, inputs in contexts:
                row = df[(df.subset == subset) & (df.model == model) & (df.fixed_level == inputs)
                         & (df.split_seed == split) & (df.truth == truth)]
                assert len(row) == 1
                selected.append(row.iloc[0])
            selected = pd.DataFrame(selected)
            x = selected.rmse_difference.to_numpy()
            error = np.maximum(0, np.vstack([x-selected.rmse_ci_lower, selected.rmse_ci_upper-x]))
            ax.errorbar(x, np.arange(len(contexts))+offset, xerr=error, fmt=marker,
                        color=color, markersize=4, elinewidth=.9, capsize=2, label=f"Split {split}")
        ax.axvline(0, color="0.3", linestyle="--", linewidth=.9)
        ax.axhline(3.5, color="0.85", linewidth=.8)
        ax.set(title=title, xlabel="RMSE difference (cycles)\nraw-trained − capped-trained")
        ax.grid(axis="x", alpha=.16)
        ax.set_yticks(np.arange(len(contexts)), labels)
        ax.set_ylim(7.55, -.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.6, -.005))
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2.2)
    save(fig, 2)


def metric_winners():
    df = read_csv("metric_winner_contexts.csv")
    df = df[(df.truth == "capped125") & (df.population == "endpoint")]
    assert len(df) == 24
    fills = {"lstm": "#C6E3F0", "cnn_1d": "#F5D6BD", "lightgbm": "#C9E8DA"}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
    for panel, (ax, subset) in enumerate(zip(axes.flat, ("FD001", "FD002", "FD003", "FD004"))):
        sub = df[df.subset == subset].sort_values(["split_seed", "init_seed"])
        for y, row in enumerate(sub.itertuples()):
            for x, winner in enumerate((row.rmse_winner, row.nasa_winner)):
                ax.add_patch(Rectangle((x-.47, y-.40), .94, .80, facecolor=fills.get(winner, "#EEEEEE"), edgecolor="white"))
                text = " / ".join(LABELS.get(k, k) for k in winner.split("|"))
                ax.text(x, y, text, ha="center", va="center", fontsize=10)
            if row.conflict:
                ax.add_patch(Rectangle((-.49, y-.43), 1.98, .86, facecolor="none", edgecolor="#A51C30", linewidth=1.3))
        ax.set(xticks=[0, 1], xticklabels=["RMSE winner", "NASA-loss winner"],
               yticks=np.arange(6), yticklabels=[f"Split {r.split_seed}, init {r.init_seed}" for r in sub.itertuples()],
               xlim=(-.53, 1.53), ylim=(5.65, -.7), title=f"{'abcd'[panel]}  {subset}")
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, pad=5)
        for spine in ax.spines.values(): spine.set_visible(False)
    handles = [Patch(facecolor="none", edgecolor="#A51C30", label="Disjoint metric winners")]
    fig.legend(handles=handles, loc="lower center", frameon=False, bbox_to_anchor=(.56, .006))
    fig.tight_layout(rect=(0, .05, 1, 1), h_pad=2.3, w_pad=2.2)
    save(fig, 3)


def stage_coverage():
    df = read_csv("uq_stage_engine_bootstrap.csv")
    df = df[df.split_seed == 42]
    stages = ("0-30", "31-80", ">80")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), sharey=True)
    for panel, (ax, subset) in enumerate(zip(axes.flat, ("FD001", "FD002", "FD003", "FD004"))):
        counts = []
        for i, model in enumerate(("lstm", "cnn_1d", "lightgbm")):
            rows = df[(df.subset == subset) & (df.model == model)].set_index("stage").loc[list(stages)]
            y = rows.coverage.to_numpy()
            err = np.maximum(0, np.vstack([y-rows.coverage_ci_lower, rows.coverage_ci_upper-y]))
            ax.errorbar(np.arange(3)+(i-1)*.18, y, yerr=err, fmt=["o", "s", "^"][i],
                        color=COLORS[model], label=LABELS[model], markersize=5, elinewidth=1, capsize=3)
            counts.append(rows.engines.to_numpy())
        assert np.array_equal(counts[0], counts[1]) and np.array_equal(counts[0], counts[2])
        ax.axhline(.9, color="0.35", linestyle="--", linewidth=.9)
        ax.set(ylim=(0, 1.035), xlim=(-.45, 2.45), title=f"{'abcd'[panel]}  {subset}",
               xticks=np.arange(3), xticklabels=[f"{stage.replace('-', '–')}\n(n={n})" for stage,n in zip(stages,counts[0])])
        ax.grid(axis="y", alpha=.16)
        if panel % 2 == 0: ax.set_ylabel("Empirical coverage")
        if panel >= 2: ax.set_xlabel("Raw RUL (cycles); n = engines")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", frameon=False, ncol=3,
               bbox_to_anchor=(.55, -.005))
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=1.6, w_pad=1.8)
    save(fig, 4)


def main():
    summary = json.loads((ANALYSIS / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["completed_grid_runs"] == 96
    OUT.mkdir(exist_ok=True)
    diagnostic()
    target_contrasts()
    metric_winners()
    stage_coverage()
    metadata = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sources": {str(p.relative_to(HERE)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(SOURCES)},
                "png_dpi": 300, "formats": ["png", "pdf", "svg"],
                "resampling": "5000 paired engine replicates from the analysis tables; calibration held fixed"}
    (OUT / "FIGURE_PROVENANCE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved four figures as PNG/PDF/SVG and their source hashes.")


if __name__ == "__main__":
    main()
