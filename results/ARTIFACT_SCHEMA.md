# Run Artifact Schema

The following contract is mandatory for every completed model run.

## `meta.json`

Required keys: `run_id`, `dataset`, `subset`, `model`, `output_mode`, `seed`,
`protocol_version`, `protocol_sha256`, `split_file`, `split_sha256`,
`code_revision`, `started_at`, `finished_at`, `training_seconds`,
`status`, `metrics`.

The machine-readable schema is `results/schemas/run_meta.schema.json`.
Completed-run validation requires `status=completed`, SHA-256 digests with 64
hexadecimal characters, ISO-8601 timestamps, and a non-negative training time.

## Checkpoint

`checkpoint/` must exist and contain at least one file. The implementation may
choose its native checkpoint filename and format, but an empty directory does
not satisfy the contract.

## Prediction tables

`preds_val.parquet`, `preds_calib.parquet`, and `preds_test.parquet` contain
one row per evaluated sliding window with:

| Column | Type | Required |
|---|---|---|
| `unit_id` | integer/string | yes |
| `cycle` | integer | yes |
| `true_rul` | float | yes |
| `pred_rul` | float | yes |
| `pred_q10` | float | quantile runs |
| `pred_q50` | float | quantile runs |
| `pred_q90` | float | quantile runs |

Rows must retain engine identity and cycle position; aggregated RMSE/Score-only
files are insufficient for Topic 6 reuse.

The Arrow-level contract is `results/schemas/prediction_table.schema.yaml`.
Prediction tables must be non-empty and null-free in required columns. Quantile
runs additionally require `pred_q10 <= pred_q50 <= pred_q90` per row and use
`pred_q50` as `pred_rul`.

## Validation

Validate the grid register from the repository root:

```powershell
.\.venv\Scripts\python.exe -m rul_audit.protocols.assets --root .
```

Once split files and completed runs exist, include them explicitly:

```powershell
.\.venv\Scripts\python.exe -m rul_audit.protocols.assets --root . `
  --split configs/splits/FD001_seed42.json `
  --run-dir results/runs/<run_id>
```

The validator checks the 160-cell cross product, deterministic run IDs, engine-
unit split disjointness, calibration isolation, metadata, checkpoint presence,
Parquet columns/dtypes, nulls, and quantile ordering.
