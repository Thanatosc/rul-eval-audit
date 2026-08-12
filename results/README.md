# Results

Generated artifacts belong under:

- `runs/`: immutable per-run metrics and provenance
- `tables/`: derived tabular summaries
- `figures/`: generated publication figures

Every run artifact must record the configuration hash, code commit, dataset
checksums, seed, environment versions, start/end time, and failure status.

The shared Topic 6/Topic 7 priority register is `UNIFIED_GRID_STATUS.csv`. All
160 cells defined by `configs/unified_grid.yaml` are `completed`, with zero
failed cells; all 160 run directories and 480 prediction tables pass the final
artifact and endpoint-metric audit. The registered `kill_v1` matrix is also
complete: all 120 rows in
`KILL_TEST_STATUS.csv` are `completed`, all run directories pass the artifact
validator, and the frozen decision is `PASS`.

Unified-grid summaries are `UNIFIED_GRID_SUMMARY.json`,
`UNIFIED_GRID_RUN_METRICS.parquet`, `UNIFIED_GRID_MEAN_METRICS.csv`, and
`UNIFIED_GRID_POINT_QUANTILE_PAIRS.csv`. Reproduce the final validation with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_unified_grid.py --root .
```

Kill Test summaries are `KILL_TEST_DECISION.json`,
`KILL_TEST_RMSE_EFFECTS.csv`, `KILL_TEST_RANKING_REVERSALS.csv`,
`KILL_TEST_MEAN_RMSE.csv`, and `KILL_TEST_RUN_METRICS.parquet`. Reproduce the
validation and mechanical decision with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_kill_test.py --root . --write
```

Generated `runs/synthetic_smoke/` artifacts exercise code and schemas only.
Their data class is synthetic, their metrics are not research findings, and
they cannot be used to change a real-cell status.

`runs/synthetic_unified_smoke/` analogously covers all four backbones and both
output modes and remains excluded from research findings.
