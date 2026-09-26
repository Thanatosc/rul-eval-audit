# Supplement S5. Sensor perturbations and inference cost

## S5.1 Scope and fixed controls

These checks address the remaining request for sensor-noise sensitivity and
computational cost. Six already fitted predictors are assessed: LSTM, 1D CNN
and LightGBM on FD001 and FD004, partition seed 42, initialization seed 11,
capped training targets and 14 sensors. The FD004 LSTM uses its budget-extended
checkpoint from S4; the other five use their primary-grid checkpoints. Selection
was by these fixed reference settings, not by their perturbed test errors.

The specification was frozen in `review_followups/REVIEW_CHECK_SPEC.json` before
the additional results. No models were trained. The archive therefore still has
107 fits (96 primary, two diagnostics, nine budget checks), plus 72 new perturbed
test-prediction tables. The clean test predictions were reproduced exactly before
applying perturbations. This remains an exploratory, simulated-benchmark check.

## S5.2 Perturbation construction and analysis

For Gaussian noise, each unique test sensor observation receives an independent
zero-mean draw with standard deviation 1%, 3% or 5% of that sensor's training-only
range. Equivalently, standard deviations are 0.01, 0.03 or 0.05 after the original
training-only MinMax transform. A separate bias condition shifts each sensor by
plus or minus 3% of its training range, with the sign sampled independently for
each engine and sensor and held fixed through that trajectory. All selected
training sensor ranges were positive. Transformed values are not clipped.

Corruption seeds are 101, 211 and 307. Within each subset and seed, all three
models receive exactly the same perturbed observations. Gaussian amplitudes
share one standard-normal draw. Perturbations are applied before windowing,
so overlapping windows reuse the same noisy observation. For short engines,
left padding repeats the corrupted first observation. Weights, training and
calibration data, engine split, labels, window length/stride and scaler are fixed.

For each configuration, squared-error and NASA-loss contributions are averaged
over the three corruption realizations within engine before aggregation. The
5,000 paired bootstrap replicates then resample complete test engines, using
seed 20260926. Corruption draws are not extra engine samples or training
replications. Intervals condition on the selected fits and these three draws;
they do not quantify general uncertainty about real sensor-noise distributions.
All six scoring truth/population combinations are retained in the data tables.

The amplitudes are controlled stress levels, not estimates of field instrument
accuracy. Global training ranges can have different physical meanings and
within-condition signal-to-noise ratios across sensors and subsets. A small
percentage of range is not necessarily a small perturbation relative to a
degradation signal. Correlated noise, transient faults, time-varying drift,
noise-aware training and field-calibrated corruption models were not studied.

## S5.3 Endpoint errors and winning models

![](review_followups/figures/Figure_S5_sensor_noise.png)

**Figure S5.** Gaussian sensor perturbations under the fixed predictors. Scores
use common capped-RUL truth at official endpoints. Error bars are conditional
test-engine 95% percentile intervals after averaging losses over three
corruption realizations. The x-axis is a fraction of training sensor range,
not an observed physical noise level. The table below also includes the bias
condition. All models and prescribed amplitudes are shown.

On FD001, LSTM remains the lowest-RMSE model in every condition. On FD004, the
clean-data winner is LightGBM (17.706 cycles). At 1% Gaussian noise, its RMSE is
47.063, compared with 21.262 for LSTM and 25.423 for CNN. At 3% and 5% Gaussian
noise, CNN has the lowest RMSE; at 3% engine bias, LSTM has the lowest RMSE.
The RMSE winner changes in 4/8 perturbed subset/condition contexts and the
NASA-loss winner in 6/8. These are eight dependent descriptive contexts, not
eight independent trials or a general ranking of architectures.

**Table S5.1. Common capped-truth endpoint errors.** Changes are perturbed minus
clean RMSE; intervals use paired engines. The clean LSTM entry on FD004 is the
extended-checkpoint value and differs from the original primary-grid entry.

| Subset | Model | Condition | RMSE | Change [95% interval] |
| --- | --- | --- | --- | --- |
| FD001 | LSTM | clean | 13.426 | +0.000 [+0.000, +0.000] |
| FD001 | LSTM | 1% Gaussian SD | 13.484 | +0.058 [-0.073, +0.185] |
| FD001 | LSTM | 3% Gaussian SD | 13.886 | +0.460 [+0.069, +0.845] |
| FD001 | LSTM | 5% Gaussian SD | 14.611 | +1.186 [+0.536, +1.855] |
| FD001 | LSTM | 3% engine bias | 14.120 | +0.694 [+0.198, +1.200] |
| FD001 | 1D CNN | clean | 15.771 | +0.000 [+0.000, +0.000] |
| FD001 | 1D CNN | 1% Gaussian SD | 15.832 | +0.062 [-0.060, +0.169] |
| FD001 | 1D CNN | 3% Gaussian SD | 16.090 | +0.319 [-0.034, +0.628] |
| FD001 | 1D CNN | 5% Gaussian SD | 16.554 | +0.783 [+0.210, +1.297] |
| FD001 | 1D CNN | 3% engine bias | 16.858 | +1.087 [+0.471, +1.681] |
| FD001 | LightGBM | clean | 13.767 | +0.000 [+0.000, +0.000] |
| FD001 | LightGBM | 1% Gaussian SD | 14.170 | +0.402 [-0.014, +0.803] |
| FD001 | LightGBM | 3% Gaussian SD | 14.426 | +0.659 [-0.303, +1.644] |
| FD001 | LightGBM | 5% Gaussian SD | 16.263 | +2.496 [+0.769, +4.313] |
| FD001 | LightGBM | 3% engine bias | 16.008 | +2.240 [+0.958, +3.616] |
| FD004 | LSTM | clean | 19.089 | +0.000 [+0.000, +0.000] |
| FD004 | LSTM | 1% Gaussian SD | 21.262 | +2.172 [+1.156, +3.274] |
| FD004 | LSTM | 3% Gaussian SD | 34.604 | +15.514 [+13.052, +18.062] |
| FD004 | LSTM | 5% Gaussian SD | 47.543 | +28.454 [+25.089, +31.897] |
| FD004 | LSTM | 3% engine bias | 38.237 | +19.147 [+16.828, +21.513] |
| FD004 | 1D CNN | clean | 25.138 | +0.000 [+0.000, +0.000] |
| FD004 | 1D CNN | 1% Gaussian SD | 25.423 | +0.285 [+0.065, +0.510] |
| FD004 | 1D CNN | 3% Gaussian SD | 26.939 | +1.801 [+1.076, +2.529] |
| FD004 | 1D CNN | 5% Gaussian SD | 29.330 | +4.192 [+2.960, +5.403] |
| FD004 | 1D CNN | 3% engine bias | 45.018 | +19.880 [+17.368, +22.294] |
| FD004 | LightGBM | clean | 17.706 | +0.000 [+0.000, +0.000] |
| FD004 | LightGBM | 1% Gaussian SD | 47.063 | +29.357 [+25.850, +32.911] |
| FD004 | LightGBM | 3% Gaussian SD | 52.423 | +34.717 [+31.026, +38.483] |
| FD004 | LightGBM | 5% Gaussian SD | 56.131 | +38.425 [+34.612, +42.379] |
| FD004 | LightGBM | 3% engine bias | 59.311 | +41.605 [+38.350, +45.002] |

## S5.4 Prediction-displacement accounting

After inspecting the completed checks, we examined the large error changes using
an exact endpoint identity. If e is clean prediction error and d is the change
in prediction after corruption, then the MSE change is
$2\,\mathrm{mean}(ed)+\mathrm{mean}(d^2)$.
The averages retain the same engine and corruption-realization weights as S5.3.

Under 1% Gaussian noise on FD004, the tree's mean prediction displacement is
−32.898 cycles and its root mean squared displacement is 44.968 cycles. The
squared-displacement contribution is 2,022.116 cycle-squared units, partly offset
by a −120.657 cross term, giving a 1,901.459 increase in MSE. Thus the large RMSE
change reflects substantial downward prediction displacement; it is not caused
by a change in scoring labels or engine weights. LSTM and CNN displacements are
smaller in this condition.

This is post hoc accounting of the fitted response. It does not isolate tree
split mechanisms, hidden representations, physical failure mechanisms or the
cause of any architecture's sensitivity.

**Table S5.2. Prediction displacement at 1% Gaussian noise.** MSE terms have
cycle-squared units; displacement summaries are in cycles. Full amplitudes and
realizations are supplied as CSV files.

| Subset | Model | Mean displacement | RMS displacement | Squared-displacement term | Cross term | MSE increase |
| --- | --- | --- | --- | --- | --- | --- |
| FD001 | 1D CNN | -0.088 | 0.948 | 0.900 | +1.044 | +1.944 |
| FD001 | LightGBM | -0.073 | 2.827 | 7.995 | +3.243 | +11.238 |
| FD001 | LSTM | -0.029 | 1.174 | 1.379 | +0.185 | +1.564 |
| FD004 | 1D CNN | -0.000 | 2.996 | 8.973 | +5.425 | +14.398 |
| FD004 | LightGBM | -32.898 | 44.968 | 2022.116 | -120.657 | +1901.459 |
| FD004 | LSTM | -0.640 | 10.903 | 118.867 | -31.209 | +87.658 |

## S5.5 Complexity and measured inference cost

For a window of length L, sensor dimension D and hidden/channel size H or C,
the principal multiply-accumulate terms scale as $O(LH(D+H))$ for the single-layer
LSTM and $O(LDCk_1+LC^2k_2)$ for the two convolutional layers.
Here L=30, D=14, H=C=64, k1=5 and k2=3. The neural models
contain 20,545 and 16,961 trainable parameters, respectively. A tree ensemble
requires O(T h) threshold decisions per prediction for T trees and typical
traversed depth h; the two selected ensembles contain 948 and 831 trees. These
are implementation counts and order-of-growth descriptions, not measured FLOPs.

Latency measures warm forward calls using the first 1 or 64 retained test windows
as fixed batches, after ten warmups,
with 50 measured calls per setting (600 measured calls in total). Neural models
use the RTX 4060 Laptop GPU and tree models one CPU thread. Inputs are already on
the selected device. CUDA is synchronized around each timed call. Loading,
preprocessing, data transfer and final CPU conversion are excluded. The p95 is
an empirical percentile of repeated calls on the same batch, not a percentile
across engine workloads or a model-performance confidence interval.

CPU and GPU implementations have different execution paths. These observations
do not isolate architectural efficiency, establish real-time scheduling bounds,
or measure end-to-end deployed latency. Checkpoint byte sizes and all 600 raw
timing observations are included. Training costs remain in Supplement S2.

| Subset | Model | Device | Size | Batch | Median ms/batch | p95 ms/batch | Predictions/s at median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | LSTM | GPU | 20,545 parameters | 1 | 0.2946 | 1.0487 | 3,394 |
| FD001 | LSTM | GPU | 20,545 parameters | 64 | 0.7876 | 1.0942 | 81,254 |
| FD001 | 1D CNN | GPU | 16,961 parameters | 1 | 0.2329 | 0.4939 | 4,294 |
| FD001 | 1D CNN | GPU | 16,961 parameters | 64 | 0.2422 | 0.5680 | 264,190 |
| FD001 | LightGBM | CPU, 1 thread | 948 trees | 1 | 0.1074 | 0.2554 | 9,307 |
| FD001 | LightGBM | CPU, 1 thread | 948 trees | 64 | 4.5805 | 5.4158 | 13,972 |
| FD004 | LSTM | GPU | 20,545 parameters | 1 | 0.2411 | 0.4182 | 4,147 |
| FD004 | LSTM | GPU | 20,545 parameters | 64 | 0.2994 | 0.4793 | 213,725 |
| FD004 | 1D CNN | GPU | 16,961 parameters | 1 | 0.2326 | 0.3521 | 4,298 |
| FD004 | 1D CNN | GPU | 16,961 parameters | 64 | 0.2574 | 0.5609 | 248,640 |
| FD004 | LightGBM | CPU, 1 thread | 831 trees | 1 | 0.0605 | 0.0760 | 16,515 |
| FD004 | LightGBM | CPU, 1 thread | 831 trees | 64 | 2.8603 | 3.4767 | 22,375 |

## S5.6 Verification and reproduction

Independent replay reconstructed every corruption and reloaded all six models.
All 72 perturbed prediction tables (1,594,368 rows)
were reproduced with maximum absolute difference 0.0.
The check independently recomputed 468 realization scores, 180 summary scores
and 288 paired intervals, and verified overlap consistency and positive training
sensor ranges. `REVIEW_CHECK_VERIFICATION.json` supplies the numerical report.

A metadata-write correction and excluded incomplete timing series are documented
in `EXECUTION_CORRECTION.json`. The frozen design was unchanged, and the first
partial prediction is byte-identical to its completed counterpart.

From the shared revision directory, these commands use supplied predictions:

    python -X utf8 review_followups/analyze_review_checks.py
    python -X utf8 review_followups/explain_sensor_sensitivity.py
    python -X utf8 review_followups/build_review_supplement.py

The following additionally require official NASA inputs and the recorded model
environment; the first replays all corrupted predictions, while the second
reuses completed checks or executes them when absent:

    python -X utf8 review_followups/verify_review_checks.py
    python -X utf8 review_followups/run_review_checks.py

The analysis directory contains `all_realization_scores.csv`,
`paired_sensitivity.csv`, `model_winners.csv`, `model_complexity.csv`,
`latency_samples.csv`, `latency_summary.csv`, two endpoint-displacement CSVs and
the summary/identity JSON records. `model_checks/` supplies the 72 prediction
tables, raw timings and provenance; model weights are referenced from the
existing primary/budget run folders rather than copied as new fits.
