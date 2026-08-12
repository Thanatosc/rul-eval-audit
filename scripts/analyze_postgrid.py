"""Run the frozen post-grid mainline statistical analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

ANALYSIS_ID = "postgrid_analysis_v1"
MASTER_SEED = 20260812
BOOTSTRAP_RESAMPLES = 20_000
ALPHA = 0.05
PLAN_SHA256 = "8d15454a87ccef36135225c83cc145d439cfa26806cb13418a4903097b76d823"
INPUT_HASHES = {
    "results/KILL_TEST_RUN_METRICS.parquet": (
        "60e1b19cfa2f0be10640d4431ce400dbd491bb019059d6662a7785822f000a8a"
    ),
    "results/UNIFIED_GRID_RUN_METRICS.parquet": (
        "48e99c2fc935f17b1b92295cceec549d149e21150567ff534c2087ce19130976"
    ),
    "results/KILL_TEST_RMSE_EFFECTS.csv": (
        "e2e3a47bb64c64e6e8d63c700f08c3089eade476c33ae7ae9410f4448c5d05ca"
    ),
    "results/KILL_TEST_RANKING_REVERSALS.csv": (
        "b2b37af08c1bc401d4f7d7a1f880ef167c20d9e6c3888b42a5a077afd93c37a5"
    ),
}
SEEDS = (11, 23, 37, 53, 71)
KILL_DATASETS = ("FD001", "FD004")
KILL_MODELS = ("lstm", "cnn_1d", "lightgbm")
RUL_LABELS = ("piecewise_125", "linear_uncapped")
SENSOR_SETS = ("all_21", "common_14")
UNIFIED_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
UNIFIED_MODELS = ("lstm", "cnn_1d", "transformer", "lightgbm")
OUTPUT_MODES = ("point", "quantile")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm adjusted p-values and alpha=.05 reject decisions."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires finite one-dimensional p-values")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for index, original_index in enumerate(order):
        candidate = (len(values) - index) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[index] = min(running, 1.0)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted < ALPHA


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided randomization p-value for the absolute mean difference."""

    values = np.asarray(differences, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("registered contrasts require exactly five finite seed differences")
    observed = abs(float(values.mean()))
    magnitudes = np.abs(values)
    statistics = [
        abs(float(np.mean(magnitudes * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    exceedances = sum(value >= observed - 1e-12 for value in statistics)
    return exceedances / len(statistics)


def matched_rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return math.nan
    ranks = rankdata(np.abs(nonzero), method="average")
    return float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(data), size=(BOOTSTRAP_RESAMPLES, len(data)))
    means = data[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _friedman_tail_distribution(
    judges: int = 5, objects: int = 4
) -> tuple[Counter[int], int]:
    """Exact distribution keyed by 4*sum((rank_sum-expected)^2)."""

    if judges != 5 or objects != 4:
        raise ValueError("this registered exact distribution is fixed to five by four")
    states: Counter[tuple[int, ...]] = Counter({(0,) * objects: 1})
    rank_permutations = list(permutations(range(1, objects + 1)))
    for _ in range(judges):
        next_states: Counter[tuple[int, ...]] = Counter()
        for totals, count in states.items():
            for permuted in rank_permutations:
                key = tuple(left + right for left, right in zip(totals, permuted, strict=True))
                next_states[key] += count
        states = next_states
    twice_expected = judges * (objects + 1)
    distribution: Counter[int] = Counter()
    for totals, count in states.items():
        statistic_key = sum((2 * value - twice_expected) ** 2 for value in totals)
        distribution[statistic_key] += count
    return distribution, len(rank_permutations) ** judges


def kendalls_w_exact(rank_matrix: np.ndarray) -> tuple[float, float, float]:
    """Kendall's W, Friedman Q, and exact permutation p-value."""

    ranks = np.asarray(rank_matrix, dtype=float)
    if ranks.shape != (5, 4):
        raise ValueError("Kendall analysis requires a five-seed by four-model matrix")
    expected_ranks = np.arange(1.0, 5.0)
    for row in ranks:
        if not np.array_equal(np.sort(row), expected_ranks):
            raise ValueError("exact Friedman analysis requires no within-seed metric ties")
    rank_sums = ranks.sum(axis=0).astype(int)
    centered_sum_squares = float(np.sum((rank_sums - 12.5) ** 2))
    friedman_q = 12.0 * centered_sum_squares / (5 * 4 * 5)
    kendalls_w = friedman_q / (5 * 3)
    observed_key = int(sum((2 * value - 25) ** 2 for value in rank_sums))
    distribution, total = _friedman_tail_distribution()
    tail = sum(count for key, count in distribution.items() if key >= observed_key)
    return float(kendalls_w), float(friedman_q), tail / total


def validate_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan_path = root / "project" / "POSTGRID_ANALYSIS_PLAN.md"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise RuntimeError("post-grid analysis plan differs from its frozen hash")
    for relative, expected in INPUT_HASHES.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise RuntimeError(f"frozen input hash mismatch for {relative}: {observed}")

    kill = pd.read_parquet(root / "results" / "KILL_TEST_RUN_METRICS.parquet")
    unified = pd.read_parquet(root / "results" / "UNIFIED_GRID_RUN_METRICS.parquet")
    reversals = pd.read_csv(root / "results" / "KILL_TEST_RANKING_REVERSALS.csv")
    if len(kill) != 120 or kill["run_id"].nunique() != 120:
        raise RuntimeError("Kill Test input is not the closed 120-cell register")
    if len(unified) != 160 or unified["run_id"].nunique() != 160:
        raise RuntimeError("Unified input is not the closed 160-cell register")
    if not np.isfinite(kill[["rmse", "nasa_score"]].to_numpy()).all():
        raise RuntimeError("Kill Test metrics contain non-finite values")
    if not np.isfinite(unified[["rmse", "nasa_score"]].to_numpy()).all():
        raise RuntimeError("Unified metrics contain non-finite values")

    expected_kill = set(
        product(KILL_DATASETS, KILL_MODELS, RUL_LABELS, SENSOR_SETS, SEEDS)
    )
    observed_kill = set(
        kill[["dataset", "model", "rul_label", "sensor_set", "seed"]]
        .itertuples(index=False, name=None)
    )
    if observed_kill != expected_kill:
        raise RuntimeError("Kill Test factor cells do not close exactly")
    expected_unified = set(product(UNIFIED_SUBSETS, UNIFIED_MODELS, SEEDS, OUTPUT_MODES))
    observed_unified = set(
        unified[["subset", "model", "seed", "output_mode"]]
        .itertuples(index=False, name=None)
    )
    if observed_unified != expected_unified:
        raise RuntimeError("Unified factor cells do not close exactly")
    if len(reversals) != 2:
        raise RuntimeError("registered ranking-reversal table has changed")
    return kill, unified, reversals


def analyze_kill_contrasts(
    kill: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    indexed = kill.set_index(["dataset", "model", "rul_label", "sensor_set", "seed"])
    rows: list[dict[str, Any]] = []

    def add_row(
        effect_type: str,
        dataset: str,
        model: str,
        held_level: str,
        contrast: str,
        differences: np.ndarray,
    ) -> None:
        lower, upper = bootstrap_mean_ci(differences, rng)
        rows.append(
            {
                "effect_type": effect_type,
                "dataset": dataset,
                "model": model,
                "held_level": held_level,
                "contrast": contrast,
                "seed_count": len(differences),
                "mean_rmse_difference": float(differences.mean()),
                "median_rmse_difference": float(np.median(differences)),
                "sd_seed_difference": float(differences.std(ddof=1)),
                "bootstrap_ci95_low": lower,
                "bootstrap_ci95_high": upper,
                "positive_seed_count": int((differences > 0).sum()),
                "zero_seed_count": int((differences == 0).sum()),
                "negative_seed_count": int((differences < 0).sum()),
                "matched_rank_biserial": matched_rank_biserial(differences),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
            }
        )

    for dataset in KILL_DATASETS:
        for model in KILL_MODELS:
            for sensor_set in SENSOR_SETS:
                differences = np.asarray(
                    [
                        indexed.loc[(dataset, model, "linear_uncapped", sensor_set, seed), "rmse"]
                        - indexed.loc[(dataset, model, "piecewise_125", sensor_set, seed), "rmse"]
                        for seed in SEEDS
                    ]
                )
                add_row(
                    "label",
                    dataset,
                    model,
                    sensor_set,
                    "linear_uncapped-minus-piecewise_125",
                    differences,
                )
            for rul_label in RUL_LABELS:
                differences = np.asarray(
                    [
                        indexed.loc[(dataset, model, rul_label, "common_14", seed), "rmse"]
                        - indexed.loc[(dataset, model, rul_label, "all_21", seed), "rmse"]
                        for seed in SEEDS
                    ]
                )
                add_row(
                    "sensor",
                    dataset,
                    model,
                    rul_label,
                    "common_14-minus-all_21",
                    differences,
                )

    frame = pd.DataFrame(rows)
    adjusted, rejected = holm_adjust(frame["exact_sign_flip_p"].to_numpy())
    frame["holm_adjusted_p"] = adjusted
    frame["holm_reject_0_05"] = rejected
    return frame.sort_values(
        ["effect_type", "dataset", "model", "held_level"]
    ).reset_index(drop=True)


def summarize_kill_flip_matrix(reversals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for transition in ("label", "sensor"):
        for model_a, model_b in combinations(KILL_MODELS, 2):
            pair = {model_a, model_b}
            pair_mask = reversals.apply(
                lambda row, expected_pair=pair: {row["model_a"], row["model_b"]}
                == expected_pair,
                axis=1,
            )
            selected = reversals[(reversals["transition"] == transition) & pair_mask]
            rows.append(
                {
                    "transition": transition,
                    "model_a": model_a,
                    "model_b": model_b,
                    "evaluated_contexts": 4,
                    "ranking_reversal_count": len(selected),
                    "ranking_reversal_rate": len(selected) / 4,
                }
            )
    return pd.DataFrame(rows)


def analyze_unified_stability(
    unified: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = unified.copy()
    ranked["rmse_rank"] = ranked.groupby(["subset", "output_mode", "seed"])["rmse"].rank(
        method="average", ascending=True
    )
    stability_rows: list[dict[str, Any]] = []
    kendall_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for subset in UNIFIED_SUBSETS:
        for output_mode in OUTPUT_MODES:
            panel = ranked[
                (ranked["subset"] == subset) & (ranked["output_mode"] == output_mode)
            ]
            rank_matrix = (
                panel.pivot(index="seed", columns="model", values="rmse_rank")
                .loc[list(SEEDS), list(UNIFIED_MODELS)]
                .to_numpy()
            )
            w_value, friedman_q, exact_p = kendalls_w_exact(rank_matrix)
            kendall_rows.append(
                {
                    "subset": subset,
                    "output_mode": output_mode,
                    "seed_count": 5,
                    "model_count": 4,
                    "kendalls_w": w_value,
                    "friedman_q": friedman_q,
                    "exact_permutation_p": exact_p,
                }
            )
            for model in UNIFIED_MODELS:
                model_rows = panel[panel["model"] == model].set_index("seed").loc[list(SEEDS)]
                rmse = model_rows["rmse"].to_numpy()
                ranks = model_rows["rmse_rank"].to_numpy()
                lower, upper = bootstrap_mean_ci(rmse, rng)
                stability_rows.append(
                    {
                        "subset": subset,
                        "output_mode": output_mode,
                        "model": model,
                        "seed_count": 5,
                        "mean_rmse": float(rmse.mean()),
                        "sd_rmse": float(rmse.std(ddof=1)),
                        "median_rmse": float(np.median(rmse)),
                        "min_rmse": float(rmse.min()),
                        "max_rmse": float(rmse.max()),
                        "range_rmse": float(rmse.max() - rmse.min()),
                        "coefficient_of_variation": float(rmse.std(ddof=1) / rmse.mean()),
                        "bootstrap_ci95_low": lower,
                        "bootstrap_ci95_high": upper,
                        "mean_rank": float(ranks.mean()),
                        "sd_rank": float(ranks.std(ddof=1)),
                        "first_place_count": int((ranks == 1).sum()),
                    }
                )
            values = panel.pivot(index="seed", columns="model", values="rmse").loc[
                list(SEEDS), list(UNIFIED_MODELS)
            ]
            for model_a, model_b in combinations(UNIFIED_MODELS, 2):
                differences = values[model_a].to_numpy() - values[model_b].to_numpy()
                signs = np.sign(differences)
                opposite_pairs = sum(
                    left * right < 0 for left, right in combinations(signs, 2)
                )
                pair_rows.append(
                    {
                        "subset": subset,
                        "output_mode": output_mode,
                        "model_a": model_a,
                        "model_b": model_b,
                        "model_a_wins": int((differences < 0).sum()),
                        "model_b_wins": int((differences > 0).sum()),
                        "seed_ties": int((differences == 0).sum()),
                        "opposite_order_seed_pairs": opposite_pairs,
                        "evaluated_seed_pairs": 10,
                        "opposite_order_fraction": opposite_pairs / 10,
                    }
                )

    kendall = pd.DataFrame(kendall_rows)
    adjusted, rejected = holm_adjust(kendall["exact_permutation_p"].to_numpy())
    kendall["holm_adjusted_p"] = adjusted
    kendall["holm_reject_0_05"] = rejected
    return pd.DataFrame(stability_rows), kendall, pd.DataFrame(pair_rows)


def analyze_point_quantile_flips(unified: pd.DataFrame) -> pd.DataFrame:
    values = unified.pivot(
        index=["subset", "seed", "model"], columns="output_mode", values="rmse"
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for subset in UNIFIED_SUBSETS:
        panel = values[values["subset"] == subset]
        for model_a, model_b in combinations(UNIFIED_MODELS, 2):
            pivoted = panel[panel["model"].isin([model_a, model_b])].pivot(
                index="seed", columns="model", values=["point", "quantile"]
            ).loc[list(SEEDS)]
            point_delta = pivoted[("point", model_a)] - pivoted[("point", model_b)]
            quantile_delta = pivoted[("quantile", model_a)] - pivoted[("quantile", model_b)]
            product_delta = point_delta.to_numpy() * quantile_delta.to_numpy()
            tied = (point_delta.to_numpy() == 0) | (quantile_delta.to_numpy() == 0)
            rows.append(
                {
                    "subset": subset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "evaluated_seeds": 5,
                    "strict_ranking_flip_count": int((product_delta < 0).sum()),
                    "tie_affected_seed_count": int(tied.sum()),
                    "strict_ranking_flip_rate": float((product_delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _metric_agreement_row(
    source: str, context: dict[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    ordered = frame.sort_values("model").reset_index(drop=True)
    rmse_ranks = rankdata(ordered["rmse"], method="average")
    nasa_ranks = rankdata(ordered["nasa_score"], method="average")
    spearman = float(spearmanr(rmse_ranks, nasa_ranks).statistic)
    tau = float(kendalltau(rmse_ranks, nasa_ranks, variant="b").statistic)
    rmse_winners = sorted(ordered.loc[rmse_ranks == 1, "model"].tolist())
    nasa_winners = sorted(ordered.loc[nasa_ranks == 1, "model"].tolist())
    discordant = 0
    tied_pairs = 0
    for left, right in combinations(range(len(ordered)), 2):
        rmse_delta = ordered.loc[left, "rmse"] - ordered.loc[right, "rmse"]
        nasa_delta = ordered.loc[left, "nasa_score"] - ordered.loc[right, "nasa_score"]
        if rmse_delta == 0 or nasa_delta == 0:
            tied_pairs += 1
        elif rmse_delta * nasa_delta < 0:
            discordant += 1
    return {
        "source": source,
        **context,
        "model_count": len(ordered),
        "spearman_rho": spearman,
        "kendall_tau_b": tau,
        "rmse_winner": ";".join(rmse_winners),
        "nasa_winner": ";".join(nasa_winners),
        "winner_conflict": rmse_winners != nasa_winners,
        "discordant_model_pairs": discordant,
        "tie_affected_model_pairs": tied_pairs,
    }


def analyze_metric_agreement(
    kill: pd.DataFrame, unified: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for keys, frame in kill.groupby(
        ["dataset", "rul_label", "sensor_set", "seed"], sort=True
    ):
        dataset, rul_label, sensor_set, seed = keys
        rows.append(
            _metric_agreement_row(
                "kill_test",
                {
                    "subset": dataset,
                    "protocol_level_1": rul_label,
                    "protocol_level_2": sensor_set,
                    "seed": seed,
                },
                frame,
            )
        )
    for keys, frame in unified.groupby(["subset", "output_mode", "seed"], sort=True):
        subset, output_mode, seed = keys
        rows.append(
            _metric_agreement_row(
                "unified_grid",
                {
                    "subset": subset,
                    "protocol_level_1": output_mode,
                    "protocol_level_2": "not_applicable",
                    "seed": seed,
                },
                frame,
            )
        )
    agreement = pd.DataFrame(rows)
    conflicts = agreement[agreement["winner_conflict"]].copy()
    return agreement, conflicts


def analyze_point_quantile_descriptive(unified: pd.DataFrame) -> pd.DataFrame:
    paired = unified.pivot(
        index=["subset", "model", "seed"],
        columns="output_mode",
        values=["rmse", "nasa_score"],
    )
    paired.columns = [f"{metric}_{mode}" for metric, mode in paired.columns]
    paired = paired.reset_index()
    paired["rmse_difference"] = paired["rmse_quantile"] - paired["rmse_point"]
    paired["nasa_difference"] = (
        paired["nasa_score_quantile"] - paired["nasa_score_point"]
    )
    rows = []
    for (subset, model), frame in paired.groupby(["subset", "model"], sort=True):
        rmse = frame["rmse_difference"].to_numpy()
        nasa = frame["nasa_difference"].to_numpy()
        rows.append(
            {
                "subset": subset,
                "model": model,
                "paired_seed_count": 5,
                "mean_quantile_minus_point_rmse": float(rmse.mean()),
                "median_quantile_minus_point_rmse": float(np.median(rmse)),
                "sd_quantile_minus_point_rmse": float(rmse.std(ddof=1)),
                "quantile_lower_rmse_seed_count": int((rmse < 0).sum()),
                "mean_quantile_minus_point_nasa_score": float(nasa.mean()),
                "median_quantile_minus_point_nasa_score": float(np.median(nasa)),
                "sd_quantile_minus_point_nasa_score": float(nasa.std(ddof=1)),
                "quantile_lower_nasa_seed_count": int((nasa < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def make_figures(
    kill_contrasts: pd.DataFrame,
    stability: pd.DataFrame,
    agreement: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    effects = kill_contrasts.sort_values("mean_rmse_difference").reset_index(drop=True)
    labels = [
        f"{row.dataset} | {row.model} | {row.held_level}"
        for row in effects.itertuples()
    ]
    colors = ["#006d77" if value == "label" else "#bc6c25" for value in effects["effect_type"]]
    figure, axis = plt.subplots(figsize=(9, 10))
    y = np.arange(len(effects))
    means = effects["mean_rmse_difference"].to_numpy()
    errors = np.vstack(
        [
            means - effects["bootstrap_ci95_low"].to_numpy(),
            effects["bootstrap_ci95_high"].to_numpy() - means,
        ]
    )
    axis.errorbar(means, y, xerr=errors, fmt="none", ecolor="#737373", capsize=2)
    axis.scatter(means, y, c=colors, s=28, zorder=3)
    axis.axvline(0, color="#1f1f1f", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Paired mean RMSE difference (95% seed-bootstrap interval)")
    axis.set_title("Kill Test protocol effects across five seeds")
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    figure.tight_layout()
    kill_path = figures_dir / "FIG_KILL_TEST_EFFECTS.png"
    figure.savefig(kill_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    model_order = list(UNIFIED_MODELS)
    display = {"lstm": "LSTM", "cnn_1d": "1D-CNN", "transformer": "Transformer", "lightgbm": "LightGBM"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, subset in zip(axes.flat, UNIFIED_SUBSETS, strict=True):
        panel = stability[stability["subset"] == subset]
        for offset, mode, color, marker in (
            (-0.08, "point", "#006d77", "o"),
            (0.08, "quantile", "#bc6c25", "s"),
        ):
            selected = panel[panel["output_mode"] == mode].set_index("model").loc[model_order]
            x = np.arange(len(model_order)) + offset
            means = selected["mean_rmse"].to_numpy()
            errors = np.vstack(
                [
                    means - selected["bootstrap_ci95_low"].to_numpy(),
                    selected["bootstrap_ci95_high"].to_numpy() - means,
                ]
            )
            axis.errorbar(
                x,
                means,
                yerr=errors,
                color=color,
                marker=marker,
                linewidth=1.2,
                capsize=3,
                label=mode,
            )
        axis.set_title(subset)
        axis.set_ylabel("Endpoint RMSE")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_xticks(np.arange(len(model_order)), [display[value] for value in model_order], rotation=15)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Unified Grid five-seed RMSE stability", y=1.01)
    figure.tight_layout()
    unified_path = figures_dir / "FIG_UNIFIED_RMSE_STABILITY.png"
    figure.savefig(unified_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    styles = {
        "kill_test": ("#006d77", "Kill Test"),
        "unified_grid": ("#bc6c25", "Unified Grid"),
    }
    for source, (color, label) in styles.items():
        selected = agreement[agreement["source"] == source]
        axis.scatter(
            selected["spearman_rho"],
            selected["kendall_tau_b"],
            c=color,
            marker="o",
            alpha=0.65,
            label=label,
        )
        conflicts = selected[selected["winner_conflict"]]
        axis.scatter(
            conflicts["spearman_rho"],
            conflicts["kendall_tau_b"],
            facecolors="none",
            edgecolors="#111111",
            marker="s",
            s=70,
            linewidth=1.0,
        )
    axis.axhline(0, color="#999999", linewidth=0.7)
    axis.axvline(0, color="#999999", linewidth=0.7)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("Spearman rank agreement: RMSE vs NASA Score")
    axis.set_ylabel("Kendall tau-b: RMSE vs NASA Score")
    axis.set_title("Metric-ranking agreement by protocol context and seed")
    axis.legend(frameon=False)
    axis.grid(color="#e5e5e5", linewidth=0.6)
    figure.tight_layout()
    agreement_path = figures_dir / "FIG_RANK_AGREEMENT.png"
    figure.savefig(agreement_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [kill_path, unified_path, agreement_path]


def write_validation_report(
    path: Path,
    contrasts: pd.DataFrame,
    kendall: pd.DataFrame,
    agreement: pd.DataFrame,
    conflicts: pd.DataFrame,
    point_quantile: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    largest = contrasts.iloc[contrasts["mean_rmse_difference"].abs().argmax()]
    w_min = kendall.loc[kendall["kendalls_w"].idxmin()]
    w_max = kendall.loc[kendall["kendalls_w"].idxmax()]
    report = f"""# Post-grid Statistical Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: `{ANALYSIS_ID}`
- Frozen Plan SHA-256: `{PLAN_SHA256}`

## Validation Report

- **Source:** 120 registered Kill Test cells and 160 validated Unified Grid cells
- **Overall Confidence:** CAUTION
- **Reason:** all registered cells and exact small-sample tests are reproducible, but only five computational seeds are available and this inferential plan was frozen after descriptive outcomes were known.

### Statistical Findings

| Finding | Method | Result | Interpretation boundary |
|---|---|---|---|
| Largest Kill Test contrast | Five paired seeds; exact sign-flip; Holm family of 24 | `{largest.dataset}`, `{largest.model}`, `{largest.held_level}`: mean difference `{largest.mean_rmse_difference:.4f}` RMSE; rank-biserial `{largest.matched_rank_biserial:.3f}`; raw p `{largest.exact_sign_flip_p:.4f}`; Holm p `{largest.holm_adjusted_p:.4f}` | Effect magnitude is descriptive for the frozen design; five-seed exact p-values cannot fall below 0.0625. |
| Kill Test multiplicity | Holm FWER | `{int(contrasts['holm_reject_0_05'].sum())}/24` contrasts rejected at 0.05 | Lack of rejection is not evidence of negligible protocol effects. |
| Unified rank concordance range | Exact Friedman permutation; Holm family of 8 | W `{w_min.kendalls_w:.3f}` (`{w_min.subset}`, `{w_min.output_mode}`) to `{w_max.kendalls_w:.3f}` (`{w_max.subset}`, `{w_max.output_mode}`); `{int(kendall['holm_reject_0_05'].sum())}/8` corrected rejections | W describes stability of model order across the five seeds, not model quality. |
| RMSE/NASA winner conflicts | Complete within-context rankings | `{len(conflicts)}/{len(agreement)}` contexts selected different best models | NASA Score's asymmetric error penalty can change architecture selection. |
| Point/quantile q50 RMSE | Descriptive paired counts only | `{summary['point_quantile']['quantile_lower_rmse_pairs']}/80` quantile cells had lower RMSE | No superiority test: objectives differ (pinball versus MSE). |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Small replicate count | Five seeds yield coarse exact randomization p-values and unstable percentile intervals. | All inferential outputs |
| Analysis timing | The plan predates new inference but follows the registered PASS and descriptive means. | Confirmatory language |
| Objective mismatch | Point and quantile models optimize different losses. | Point/quantile comparisons |
| Nonlinear metric | NASA Score is asymmetric and highly sensitive to late-prediction outliers. | Metric-rank conflicts |
| Scope | Subsets, budgets, seeds, and architectures are fixed rather than sampled from a target population. | External generalization |

### Fallacy Scan

- **Coverage:** 11/11 statistical fallacy types checked

| Fallacy | Severity | Finding and handling |
|---|---|---|
| Simpson's paradox | CAUTION | No pooled subset effect is used; all primary effects and rankings remain subset-specific. |
| Ecological fallacy | NOTE | Run-level endpoint summaries are not used to infer window-level or individual-engine mechanisms. |
| Berkson's paradox | NOTE | C-MAPSS is a benchmark-selected corpus; conclusions are restricted to the frozen benchmark. |
| Collider bias | NOTE | No covariate-adjusted causal regression is fitted. |
| Base-rate neglect | NOTE | No diagnostic sensitivity, specificity, PPV, or NPV claim is made. |
| Regression to the mean | NOTE | No group is selected for analysis because of an extreme observed seed result. |
| Survivorship bias | NOTE | The full 120+160 registered cells completed; failed cells were not excluded. |
| Look-elsewhere effect | CAUTION | All 24 and eight planned test families are reported with Holm correction; descriptive diagnostics are labeled. |
| Garden of forking paths | CAUTION | The post-grid plan and input hashes are frozen and disclosed, but model outcomes were already descriptively known. |
| Correlation does not imply causation | CAUTION | Protocol transitions are controlled computational contrasts, but claims remain design-specific rather than field-wide causal claims. |
| Reverse causality | NOTE | Temporal or observational causal direction is not analyzed. |

### Reproducibility

- **Method:** deterministic regeneration from frozen seed-level Parquet inputs
- **Verdict:** REPRODUCIBLE
- **Input hash checks:** 4/4 passed
- **Registered cell closure:** 120/120 Kill Test and 160/160 Unified Grid
- **Automatic imputation or rerun:** none
"""
    path.write_text(report, encoding="utf-8")


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    kill, unified, reversals = validate_inputs(root)
    rng = np.random.default_rng(MASTER_SEED)
    contrasts = analyze_kill_contrasts(kill, rng)
    flip_matrix = summarize_kill_flip_matrix(reversals)
    stability, kendall, pairwise = analyze_unified_stability(unified, rng)
    mode_flips = analyze_point_quantile_flips(unified)
    agreement, conflicts = analyze_metric_agreement(kill, unified)
    point_quantile = analyze_point_quantile_descriptive(unified)

    results_dir = root / "results" / "postgrid"
    figures_dir = root / "paper" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "KILL_TEST_CONTRAST_INFERENCE.csv": contrasts,
        "KILL_TEST_RANKING_FLIP_MATRIX.csv": flip_matrix,
        "UNIFIED_MODEL_STABILITY.csv": stability,
        "UNIFIED_KENDALL_W.csv": kendall,
        "UNIFIED_PAIRWISE_STABILITY.csv": pairwise,
        "UNIFIED_POINT_QUANTILE_FLIPS.csv": mode_flips,
        "METRIC_RANK_CONSISTENCY.csv": agreement,
        "METRIC_WINNER_CONFLICTS.csv": conflicts,
        "POINT_QUANTILE_DESCRIPTIVE.csv": point_quantile,
    }
    for filename, frame in output_frames.items():
        _save_frame(frame, results_dir / filename)
    figure_paths = make_figures(contrasts, stability, agreement, figures_dir)

    max_effect = contrasts.loc[contrasts["mean_rmse_difference"].abs().idxmax()]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_verified",
        "plan_sha256": PLAN_SHA256,
        "master_seed": MASTER_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha": ALPHA,
        "input_hashes": INPUT_HASHES,
        "cell_closure": {"kill_test": 120, "unified_grid": 160},
        "kill_test": {
            "registered_contrasts": len(contrasts),
            "holm_rejections": int(contrasts["holm_reject_0_05"].sum()),
            "largest_absolute_mean_effect": float(abs(max_effect["mean_rmse_difference"])),
            "largest_effect_context": {
                "effect_type": max_effect["effect_type"],
                "dataset": max_effect["dataset"],
                "model": max_effect["model"],
                "held_level": max_effect["held_level"],
            },
            "registered_ranking_reversals": len(reversals),
        },
        "unified_grid": {
            "kendall_panels": len(kendall),
            "kendall_holm_rejections": int(kendall["holm_reject_0_05"].sum()),
            "minimum_kendalls_w": float(kendall["kendalls_w"].min()),
            "maximum_kendalls_w": float(kendall["kendalls_w"].max()),
            "model_stability_rows": len(stability),
            "pairwise_stability_rows": len(pairwise),
            "point_quantile_strict_ranking_flips": int(mode_flips["strict_ranking_flip_count"].sum()),
        },
        "metric_agreement": {
            "contexts": len(agreement),
            "winner_conflicts": len(conflicts),
            "median_spearman_rho": float(agreement["spearman_rho"].median()),
            "median_kendall_tau_b": float(agreement["kendall_tau_b"].median()),
        },
        "point_quantile": {
            "paired_cells": 80,
            "quantile_lower_rmse_pairs": int(point_quantile["quantile_lower_rmse_seed_count"].sum()),
            "quantile_lower_nasa_pairs": int(point_quantile["quantile_lower_nasa_seed_count"].sum()),
            "inference_permitted": False,
        },
        "fallacy_scan": {"checked": 11, "total": 11},
    }
    report_path = results_dir / "POSTGRID_STATISTICAL_VALIDATION.md"
    write_validation_report(
        report_path,
        contrasts,
        kendall,
        agreement,
        conflicts,
        point_quantile,
        summary,
    )
    artifact_paths = [results_dir / name for name in output_frames]
    artifact_paths.extend(figure_paths)
    artifact_paths.append(report_path)
    summary["output_hashes"] = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in artifact_paths
    }
    summary_path = results_dir / "POSTGRID_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
