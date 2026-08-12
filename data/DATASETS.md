# Dataset Provenance Register

## C-MAPSS

| Field | Recorded value |
|---|---|
| Dataset | Turbofan Engine Degradation Simulation Data Set (C-MAPSS) |
| Official distribution | `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` |
| NASA Open Data landing page | `https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data-xk2n5` |
| Accessed | 2026-08-11 |
| Preserved archive | `data/raw/C-MAPSS_Turbofan.zip` |
| Archive bytes | 12,429,152 |
| Archive SHA-256 | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| Extraction | Outer ZIP -> official `CMAPSSData.zip` -> `data/interim/cmapss/`; source text is unmodified |
| Terms | NASA public distribution; no explicit license file is present in the archive |
| Project policy | Cite the source and do not redistribute the archive by default; verify destination-specific terms before public release |
| Status | identity, archive hash, 26-column shape, RUL joins, row counts, and unit counts verified |

File-verified counts:

| Subset | Train rows | Train units | Test rows | Test units | RUL rows |
|---|---:|---:|---:|---:|---:|
| FD001 | 20,631 | 100 | 13,096 | 100 | 100 |
| FD002 | 53,759 | 260 | 33,991 | 259 | 259 |
| FD003 | 24,720 | 100 | 16,596 | 100 | 100 |
| FD004 | 61,249 | 249 | 41,214 | 248 | 248 |

The distributed `readme.txt` reverses the FD004 train/test trajectory counts.
The table above is derived from distinct unit IDs and agrees with each official
RUL vector; the discrepancy is preserved rather than silently normalized.

## Required Experimental Boundaries

1. The original archive remains unchanged under ignored `data/raw/`.
2. Split the training fleet by engine unit before creating windows.
3. Fit the primary scaler on registered training units only.
4. Calibration units never participate in training, tuning, or scaler fitting.
5. Official test data never participate in preprocessing or model selection.
6. Preserve `val`, `calib`, and `test` per-window predictions for Topic 6.

## N-CMAPSS

N-CMAPSS remains a later migration milestone. No N-CMAPSS archive has been
downloaded, and it does not block the registered C-MAPSS Kill Test.
