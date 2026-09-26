"""Build the literature and experiment supplements from retained evidence."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "rul-eval-audit"
OUT = HERE / "supplements"
ANALYSIS = HERE / "analysis"
LABELS = {"lstm": "LSTM", "cnn_1d": "1D CNN", "lightgbm": "LightGBM"}


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join(
        "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows)


def write(name, text):
    (OUT / name).write_text(text.strip()+"\n", encoding="utf-8")


def literature():
    dest = OUT / "literature"
    dest.mkdir(exist_ok=True)
    names = ["PHASE2_SEARCH_STRATEGY.md", "SEARCH_LOG.md", "candidate_corpus.csv", "PRISMA_COUNTS.csv",
             "FULLTEXT_MANIFEST.csv", "protocol_coding.csv", "PHASE3_ANALYSIS_CODEBOOK.csv",
             "protocol_coding_uq_v2.csv", "SOURCE_VERIFICATION_REPORT.md", "PHASE3_EVIDENCE_MATRIX.csv",
             "PHASE3_UQ_CODEBOOK.csv", "PHASE3_UQ_EVIDENCE_MATRIX.csv", "PHASE3_PROTOCOL_PRACTICE_COUNTS.csv",
             "PHASE3_UQ_PRACTICE_COUNTS.csv"]
    sources = []
    for name in names:
        source = REPO / "papercorpus" / name
        shutil.copy2(source, dest / name)
        sources.append({"source": "rul-eval-audit/papercorpus/"+name, "copy": "supplements/literature/"+name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    original = (REPO / "paper/stage5_submission/SUPPLEMENT_S1.md").read_text(encoding="utf-8")
    original = original.replace("# Supplement S1. Reproducibility Record for the Purposive Literature-Practice Audit",
                                "# Supplement S1. Literature corpus, coding and source trace")
    original = original.replace("`papercorpus/", "`literature/")
    original = original.replace("The final search and acquisition state was frozen on 11 August 2026.",
                                "The original search and acquisition state was frozen on 11 August 2026; that corpus and its archived source-status records are retained unchanged in this revision.")
    corpus = pd.read_csv(dest / "candidate_corpus.csv")
    model_ids = {f"C{i:02}" for i in range(1, 19)} | {"C21"}
    rows = [[r.candidate_id, "Model/protocol" if r.candidate_id in model_ids else "Contextual anchor",
             r.title, f'<https://doi.org/{r.doi}>'] for r in corpus.itertuples()]
    original += "\n\n## S1.6 Complete registered corpus\n\n" + table(["ID", "Role", "Title", "DOI"], rows)
    original += """

## S1.7 Additional context used in the revision

Two evaluation studies were added to the related work. They do not enter the
19-paper coding denominator or change the historical corpus counts.

| Study | Verified identity | Evidence used in this revision |
|---|---|---|
| Kamariotis et al., *A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance* | Reliability Engineering & System Safety 242 (2024), 109723; DOI 10.1016/j.ress.2023.109723 | Metadata and public manuscript; supports the existence of maintenance-cost-oriented prognostic evaluation |
| Bieber et al., *Assessing the Impact of Metrics on the Choice of Prognostic Methodologies* | AIAA Journal 62(2) (2024), 791–801; DOI 10.2514/1.J063365 | Metadata and abstract only; supports the stated study of metric-dependent methodology selection; no claim about unexamined implementation details |

Metadata lookup records are in `../references/`. These access statements describe
the material used by the assisted workflow and do not certify a new human reading
event. No copyright-restricted full text is redistributed. Source coding retains
the original single-author/AI-assistance limitation; no second independent coder
or new inter-rater reliability estimate has been added.
"""
    write("SUPPLEMENT_S1.md", original)
    (OUT / "LITERATURE_COPY_MANIFEST.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")


def experiments():
    summary = json.loads((ANALYSIS / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["completed_grid_runs"] == 96
    q = pd.read_csv(ANALYSIS / "training_quality.csv")
    core = pd.read_csv(ANALYSIS / "core_engine_bootstrap.csv")
    uq = pd.read_csv(ANALYSIS / "uq_stage_engine_bootstrap.csv")
    population_targets = pd.read_csv(ANALYSIS / "target_population_summary.csv")
    metas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((HERE / "runs").glob("*/meta.json"))]
    data_rows, hash_rows, runtime_rows = [], [], []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for seed in (42, 137, 271):
            m = next(x for x in metas if x["subset"] == subset and x["split_seed"] == seed and x["arm"] == "core")
            u, w = m["units"], m["prediction_rows"]
            data_rows.append([subset, seed, f"{u['train']}/{u['val']}/{u['calib']}/{u['test']}",
                              f"{w['train']}/{w['val']}/{w['calib']}/{w['test']}"])
        for filename, value in m["data_sha256"].items(): hash_rows.append([filename, value])
    for model, g in q.groupby("model"):
        params = g.parameters.dropna().astype(int).unique() if "parameters" in g else []
        counts = ", ".join(map(str, sorted(params))) if len(params) else "Tree count below"
        iterations = g.best_iteration.dropna() if model == "lightgbm" else g.best_epoch.dropna()
        runtime_rows.append([LABELS[model], len(g), counts, f"{int(iterations.min())}–{int(iterations.max())}",
                             f"{g.training_seconds.median():.1f} ({g.training_seconds.min():.1f}–{g.training_seconds.max():.1f})"])
    flags = q[q.cap_with_recent_best]
    flag_rows = [[f'`{r.run_id}`', int(r.best_iteration) if r.model == "lightgbm" else int(r.best_epoch),
                  int(r.iterations_evaluated) if r.model == "lightgbm" else int(r.epochs_completed)] for r in flags.itertuples()]
    primary = core[(core.truth == "capped125") & (core.population == "endpoint")]
    core_rows = [[r.subset, r.split_seed, LABELS[r.model], f"{r.rmse:.3f}",
                  f"{r.rmse_ci_lower:.3f}–{r.rmse_ci_upper:.3f}",
                  f"{r.initialization_rmse_min:.3f}–{r.initialization_rmse_max:.3f}", f"{r.nasa_mean:.3f}"] for r in primary.itertuples()]
    uq_rows = [[r.subset, LABELS[r.model], r.stage, r.engines, r.windows,
                f"{100*r.coverage:.1f} [{100*r.coverage_ci_lower:.1f}, {100*r.coverage_ci_upper:.1f}]",
                f"{r.width:.2f}", f"{r.interval_score:.2f}"] for r in uq[uq.split_seed == 42].itertuples()]
    historic = pd.read_csv(REPO / "results/postgrid/METRIC_WINNER_CONFLICTS.csv")
    historic.to_csv(ANALYSIS / "historical_metric_winner_conflicts.csv", index=False)
    hist_rows = [[r.source, r.subset, r.protocol_level_1, r.protocol_level_2, r.seed,
                  LABELS.get(r.rmse_winner, r.rmse_winner), LABELS.get(r.nasa_winner, r.nasa_winner)] for r in historic.itertuples()]
    diag_rows = []
    for file in sorted((HERE / "diagnostics").glob("*/meta.json")):
        d = json.loads(file.read_text(encoding="utf-8"))
        diag_rows.append([d["target_scale"], d["training"]["best_epoch"], f"{d['training']['best_val_rmse']:.6f}",
                          f"{d['endpoint_rmse_native']:.6f}", f"{d['prediction_range']:.6f}"])
    files = [
        ("all_run_scores.csv", "576 rows: 96 fits × two scoring truths × three prediction populations"),
        ("core_engine_bootstrap.csv", "216 rows: core summaries, paired engine intervals and initialization ranges"),
        ("paired_factor_contrasts.csv", "96 rows: all target/input contrasts, three splits, two common truths"),
        ("metric_winner_contexts.csv", "144 rows: 24 contexts × two truths × three populations; tied winning sets retained"),
        ("training_quality.csv", "All 96 fits, selected epochs/iterations, runtime, parameter count and cap flags"),
        ("repaired_vs_historical_baselines.csv", "Matched split-42 neural and tree baseline comparisons; conditional paired engine intervals"),
        ("uq_stage_engine_bootstrap.csv", "108 stage summaries with coverage/width/interval score and contributing engine counts"),
        ("uq_overall_engine_bootstrap.csv", "36 overall equal-engine interval summaries reconstructed from all stages"),
        ("uq_per_engine_stage.parquet", "Initialization-specific per-engine stage summaries and calibration half-widths"),
        ("historical_common_truth_all_runs.csv", "All 120 archived prediction bundles rescored on both common truths; original fitting limitations retained"),
        ("historical_common_truth_descriptive_contrasts.csv", "Complete historical target contrasts; descriptive archive only"),
        ("historical_metric_winner_conflicts.csv", "The original seven metric-winner conflicts, shown separately from corrective fits"),
        ("target_contrast_decomposition.csv", "Exact squared-loss attribution to endpoints above and below the cap"),
        ("target_population_contrasts.csv", "144 descriptive target contrasts: 24 fitted pairs × two truths × three populations"),
        ("target_population_summary.csv", "Direction counts and RMSE-difference ranges for all six truth/population choices"),
        ("endpoint_loss_concentration.csv", "All 60 core fits: largest 5% loss share and signed-error contributions"),
        ("core_endpoint_loss_contributions.parquet", "Per-endpoint squared and NASA losses for all core fits"),
        ("population_stage_composition.csv", "Life-stage weights for each prediction population; four subsets"),
    ]
    for filename, _ in files: assert (ANALYSIS / filename).exists(), filename
    text = f"""# Supplement S2. Corrective experiments and complete numerical results

## S2.1 Study status and experiment units

The analysis plan (`../ANALYSIS_SPEC.md`) was recorded before inspection of the new
corrective fits, after the historical benchmark results and rejection were known.
The work is exploratory. All 96 specified grid fits and both diagnostic fits
completed. Core fits comprise four subsets × three engine partitions × two LSTM
initializations, two CNN initializations and one deterministic LightGBM record.
The 36 factor fits complete the raw/capped target × 14/21-sensor design for LSTM
initialization 11 and LightGBM on FD001 and FD004. Tree results are stored once.

Independent test resampling uses engines, not windows or arbitrary seed labels.
Engine partitions reuse the available development population, and every partition
is evaluated on the same official test engines. Partition ranges, initialization
variation and conditional test-engine intervals describe different sources of
variation and are not combined into a pooled effective sample size.

## S2.2 Data, partition and retained-window counts

The following counts are calculated from the actual split and prediction records.
Within each cell, numbers are fitting / validation / calibration / official test.
Development trajectories are complete; official test trajectories are truncated.
The original NASA archive SHA-256 is
`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
The actual per-file identities used by the new runs appear below. Engine counts
are taken from unit identifiers, including FD004's 249 development/248 test
engines. Operating settings are not supplied as features, and no operating-regime
normalization, smoothing or target-aware sensor selection is fitted.

{table(['Subset','Partition seed','Engine counts','Window counts'], data_rows)}

{table(['Source file','SHA-256'], hash_rows)}

Windows have 30 cycles and stride one. The last observation is the prediction
time. Engines shorter than 30 observations contribute one endpoint, padded on the
left by their first observation. Other engines contribute all windows from cycle
30 onwards; the first 29 cycles do not receive separate predictions. Min–max
scalers use only fitting-engine observations and are applied without clipping.
Raw RUL and capped RUL are kept as separate columns, and all scores are in cycles.

## S2.3 Fitting and selection

The LSTM is one 64-unit layer followed by a scalar linear head. The 1D CNN uses
two convolutions with 64 channels (kernel sizes 5 and 3), ReLU activations,
global average pooling and a scalar linear head. Neural targets are divided by
125 for both raw and capped fits; predictions are converted back to cycles.
No prediction is clipped. Adam uses learning rate 0.001, weight decay 0.00001,
batch size 256 and deterministic shuffling. Validation uses every retained
validation window, pooled equally, and the fitting target of the run. Thus
validation selection in a training-target contrast is part of the trained bundle.
Calibration data do not select checkpoints or hyperparameters.

The corrective neural recipe halves the learning rate after five unimproved
validation epochs (PyTorch ReduceLROnPlateau with its default relative threshold),
down to 0.00001. The saved checkpoint has the lowest validation RMSE, using an
absolute improvement tolerance of 1e-9 cycles. Early stopping requires 15
unimproved epochs and at least 20 elapsed epochs. The initial cap is 150 epochs.
If the best epoch is in the last 15 epochs and the best RMSE improved by at least
0.1 cycles during those epochs, the same optimizer/run continues to a cap of 300.

LightGBM receives the flattened 30-cycle window, uses squared loss, 31 leaves,
learning rate 0.05, full row/feature sampling, zero L2 penalty, deterministic
single-thread CPU fitting, at most 1,500 trees and validation stopping patience 50.
The first metric is RMSE. Predictions use the selected best iteration. Further
library defaults are reproducible from the recorded environment and source.

{table(['Model','Fits','Trainable neural parameters','Selected epoch/iteration range','Seconds: median (range)'], runtime_rows)}

Runtime includes fitting, validation, checkpoint handling and final split
prediction in each model routine. It excludes dataset loading and window
construction and is not an isolated throughput benchmark. Neural runs used an
NVIDIA GeForce RTX 4060 Laptop GPU and tree runs one CPU thread; the two jobs ran
concurrently. These measurements do not establish a hardware-independent speed
ranking. Software versions are supplied in `../ENVIRONMENT_SNAPSHOT.json`.

There were {int(q.extended.fillna(False).sum())} neural budget extensions and
{len(flags)} grid fits flagged for a recent best at the final cap. For neural fits
the flag means the best epoch is in the final 15; for trees it means 1,500
iterations were evaluated and the best is above 1,450. A flagged fit is not
declared converged. Absence of the flag is also not proof of a global optimum.

{table(['Flagged run','Selected epoch/iteration','Evaluated epoch/iteration'], flag_rows) if flag_rows else 'No grid fit triggered the defined cap flag.'}

These counts and curves describe the preserved primary grid. Supplement S4
reports a separate threefold-budget extension for all nine flagged fits. All
nine extended fits stopped by validation patience before their new caps. The
primary records and this table are retained so the original stopping decisions
remain inspectable; the extended models and sensitivity results are supplied
under `budget_sensitivity/` in Supporting Information S3.

## S2.4 Original-recipe diagnostic

The two FD002 fits hold architecture, 14-sensor inputs, split 42, initialization
11 and the original constant-learning-rate, 50-epoch, patience-eight recipe
fixed. Numerical target scale is the changed setting. The original-scale test
predictions exactly reproduce the corresponding archived prediction array.
Scaling also changes relative regularization and optimizer numerics, so it is a
recipe diagnostic rather than causal identification of one internal mechanism.

{table(['Target divisor','Selected epoch','Validation RMSE','Endpoint test RMSE','All-test prediction range'], diag_rows)}

The initial diagnostic metadata mislabeled window counts as `units`. These counts
were moved unchanged to `prediction_rows`, and distinct engine counts now occupy
`units`. Original metadata bytes and hashes are retained alongside the corrected
records; `../DIAGNOSTIC_METADATA_CORRECTION.json` documents this correction.
No checkpoint, curve, prediction or error was changed by that metadata repair.

For the archived LSTM runs, the final recurrent state lies componentwise in
[-1,1], so a scalar linear head with weights w and bias b has upper enclosure
b+sum(abs(w)) in real arithmetic. A small distance below this enclosure implies
that the head-weighted recurrent representation lies near its maximizing bounded
direction. The complete 20-run prediction-dispersion table is supplied in
`../historical_evidence/lstm_prediction_degeneracy.csv`; 16 runs span less than
0.004 cycles over calibration and test records. This locates the observed
degeneracy in the fitted representation/output geometry, but does not identify
which optimization dynamics caused it.

## S2.5 Common-truth scores and resampling

For each fitted model, both capped-125 and raw truth are scored over (i) one
official endpoint per engine, (ii) equal engine weights after averaging losses
within each engine's windows and (iii) equal weights for all retained windows.
Squared losses are averaged before taking the square root. Mean NASA loss is
exp(-d/13)-1 for underprediction and exp(d/10)-1 otherwise, with d = prediction
minus scoring truth. Its endpoint sum equals its mean times the engine count.

The 5,000 bootstrap replicates use generator seed 20260926 and resample complete
test engines with replacement. Paired contrasts reuse the same sampled engine
indices. Core tables average loss contributions over the two neural
initializations within engine, before resampling or taking a square root. They
are not prediction ensembles. LightGBM contributes one fitting record. The 2.5th
and 97.5th percentiles are conditional, marginal intervals; no familywise
significance test is applied to the many comparisons.

All three primary partitions and initialization ranges are shown here. The
machine-readable file also includes raw-truth and window-population scores.

{table(['Subset','Split','Model','RMSE','95% engine interval','Initialization RMSE range','Mean NASA loss'], core_rows)}

## S2.6 Stage-specific residual intervals

For each core fit, absolute residuals on all calibration windows and capped truth
are pooled. The half-width is order ceil((n+1)×0.9), capped at n, from the n
ordered scores. The interval is the point prediction plus/minus this half-width;
endpoints are not clipped. Calibration windows overlap and do not supply n
independent calibration units. This procedure is used as a standard empirical
reference, without an exchangeability or stage-conditional guarantee.

Coverage, width and interval score are averaged over the windows of each engine
within a raw-RUL stage, then over contributing engines. For neural summaries,
initialization-specific engine summaries are averaged before resampling engines.
The interval score is width plus 20 times the shortfall below the lower endpoint
or excess above the upper endpoint. The reference-split results follow; all
splits, conditional intervals and calibration counts are in the CSV files.
Because the symmetric half-width is constant within each fitted model, mean
width has no test-engine resampling variation when calibration is fixed. This
does not imply that width is known without calibration-sample uncertainty.
If every contributing engine has empirical coverage one, the nonparametric
bootstrap also returns a degenerate [1,1] interval. This boundary behavior cannot
rule out unobserved failures or establish perfect population coverage.

{table(['Subset','Model','Raw-RUL stage','Engines','Windows','Coverage %, 95% interval','Width','Interval score'], uq_rows)}

## S2.7 Explanatory decompositions

These descriptive decompositions were added during writing after some corrective
results were available. They use all relevant prespecified cells and introduce no
additional fitting or hypothesis test. Let a and b be predictions from raw- and
capped-trained models, and let r and c=min(r,125) be the two scoring truths. Then

$$[(a-r)^2-(b-r)^2]-[(a-c)^2-(b-c)^2]=-2(r-c)(a-b).$$

Consequently, only endpoints above 125 can change the squared-loss contrast when
switching scoring truth. The decomposition table gives the contributions on the
full-engine denominator, separately for endpoints at/below and above the cap.
The identity is independently checked against direct squared-error differences
at every paired endpoint. It is elementary loss accounting, not a new theorem or
an identification of how either predictor was learned.

The full rescoring matrix also separates training-target preference by scoring
population. Positive differences below favor capped training. Each row contains
the same 24 fitted pairs (two subsets × three partitions × two models × two
sensor sets). These are descriptive point-estimate ranges across fitted settings,
not confidence intervals or additional independent test populations.

{table(['Scoring truth','Population','Raw minus capped RMSE range','Raw training lower','Capped training lower'], [[r.truth,r.population,f'{r.difference_min:+.2f} to {r.difference_max:+.2f}',r.raw_trained_lower_rmse,r.capped_trained_lower_rmse] for r in population_targets.itertuples()])}

On raw truth, the endpoint preference for capped training reverses for all 24
pairs under both window populations. On FD004, raw training improves the
above-125 component of endpoint squared error for every fitted pair, while capped
training improves the at/below-125 component. Only 67 of the 248 official
endpoints are above the cap. The component table shows how the latter gains
outweigh the former losses in the endpoint population. Window populations place
more weight on early life and answer a different prediction question.

Loss-concentration tables report the share of each core model's NASA loss from
its largest ceil(0.05×N) endpoint losses, the largest single loss and the share
from positive errors. Every core fit is included. The per-engine loss table
permits exact reproduction of a metric-winner comparison. Stage-composition
tables show the weighted proportions of raw RUL 0–30, 31–80 and >80 for each
prediction population, using one reference prediction record per subset; input
identities and stage membership do not depend on the fitted model.

## S2.8 Historical results retained for traceability

The archived models use the original recipes and include numerically degenerate
LSTM configurations. Their complete common-truth results remain accessible.
Deterministic tree copies and seed-labelled historical contrasts are not used as
independent inferential repetitions in this revision. Historical quantile and
point-model intervals compare different trained pipelines; they do not isolate
a calibration-method effect. The original arbitrary experiment PASS/FAIL
labels and seed-level significance claims do not support the revised conclusions.

The original seven RMSE/NASA winner conflicts are listed below. This table belongs
to the earlier 80-context analysis, including its original target, inputs and
objectives, and must not be pooled with the 24 corrective primary contexts.
Original machine labels are retained here solely to identify archived records.

{table(['Historical arm','Subset','Target/objective','Sensor setting','Seed','RMSE winner','NASA winner'], hist_rows)}

## S2.9 File guide and reproduction

All paths in the following table are relative to `../analysis/`.

{table(['File','Contents'], [(f'`{filename}`', description) for filename,description in files])}

Each `../runs/<run_id>/` and `../diagnostics/<run_id>/` directory contains the
metadata, partition, scaler, validation curve, selected checkpoint and validation,
calibration and test predictions. Histories and SHA-256 values make numerical
claims traceable. The original frozen specification and source snapshot are
retained. Historical archives are identified by DOIs 10.5281/zenodo.21915989 and
10.5281/zenodo.21915990; this revision does not claim the new artifacts are already
present in those public versions.

Acquire the official NASA C-MAPSS files separately and place them in
`rul-eval-audit/data/interim/cmapss/`, verifying the hashes above. Preserve the
package layout, with the revision scripts in
`post_rejection_20260926/transfer_revision/`. From that directory run:

```text
python -X utf8 train_repair.py --arm diagnostic
python -X utf8 train_repair.py --arm grid --model neural
python -X utf8 train_repair.py --arm grid --model lightgbm
python -X utf8 analyze_repair.py
python -X utf8 explain_results.py
python -X utf8 make_figures.py
python -X utf8 build_supplements.py
```

The two grid commands may run as separate GPU/CPU jobs. Completed records are
reused only if their identities and specification hash match; an incomplete
record requires inspection before retry. To reproduce existing tables and figures,
the provided predictions suffice and training need not be repeated. The full
matched historical-baseline comparison additionally uses the original archived
split-42 prediction files included in the reproducibility package. Environment
differences can change floating-point training results even with fixed seeds.
"""
    write("SUPPLEMENT_S2.md", text)


def environment():
    versions = {}
    for package in ("numpy", "pandas", "torch", "lightgbm", "scikit-learn", "scipy", "pyarrow", "matplotlib", "python-docx", "pillow"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    snap = {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "torch_cuda_build": "13.0", "neural_device": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "tree_threads": 1, "recorded_for": "2026-09-26 corrective revision"}
    (HERE / "ENVIRONMENT_SNAPSHOT.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    literature()
    experiments()
    environment()
    print("Built S1/S2, copied the literature coding register, and recorded the environment.")


if __name__ == "__main__":
    main()
