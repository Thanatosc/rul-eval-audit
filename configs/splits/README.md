# Unit-Level Split Register

Split files were generated after the NASA archive was verified and extracted.
Each JSON file contains deterministic engine-unit IDs for `train`, `val`, and
`calib`; windows are derived only after this split.

Required shape:

```json
{
  "dataset": "FD001",
  "seed": 42,
  "unit_split": {
    "train": ["registered unit IDs"],
    "val": ["registered unit IDs"],
    "calib": ["registered unit IDs"]
  },
  "fractions": {"train": 0.7, "val": 0.15, "calib": 0.15},
  "allocation_unit": "engine_unit",
  "calib_isolation": "never_used_for_training_or_tuning",
  "status": "ready"
}
```

The split seed and exact unit lists are part of the run provenance. The frozen
algorithm sorts unit IDs, applies `numpy.random.default_rng(42).permutation`,
rounds the 70% train count, then divides the remainder as evenly as possible
between validation and calibration. No random window-level split is permitted.

Validate each populated split before window generation:

```powershell
.\.venv\Scripts\python.exe -m rul_audit.protocols.assets --root . `
  --split configs/splits/FD001_seed42.json
```

The validator rejects within-group duplicates, pairwise overlap among
`train`/`val`/`calib`, non-engine allocation, altered fractions, empty ready
groups, or a missing calibration-isolation declaration.
