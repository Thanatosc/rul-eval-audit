# Source Verification Report

Status: `phase_2_fulltext_complete`; audit date: 2026-08-11.

This is a Phase 2 source gate. It grades source fitness and records verification
limits; it does not synthesize the evidence or make model-performance claims.

## Overall Assessment

**Sources reviewed:** 25 full texts  
**Local original PDFs verified:** 24  
**Exact remote institutional PDFs assessed:** 1  
**Preflight PASS:** 24  
**Metadata/DOI identity verified:** 25  
**Flagged for protocol caveats:** 19 model/protocol papers  
**Rejected from the verified corpus:** 0  
**Predatory-journal alerts:** none detected in the inspected publisher records  
**COI alerts:** none identified in the inspected declarations; funding and
data-availability statements were recorded where present.

The evidence levels below use the technology-field adjustment: controlled
computational comparisons are Level III, descriptive dataset/EDA papers are
Level VI, and review-level historical sources are Level V. The overall grade is
fitness for the protocol-audit claim, not a claim that all sources are equally
strong empirical evidence.

## Source Quality Matrix

| Source | Evidence level | Venue | Method | Currency | COI | Overall | Verification |
|---|---|---|---|---|---|---|---|
| C01 Asif et al. | III | pass | warn, validation/fit-scope/test-selection ambiguity | pass | government grant disclosed; no competing issue located | C | IEEE PDF + DOI; 16-page preflight |
| C02 Mo et al. | III | pass | warn, subset-wide preprocessing and missing sequence policy | pass | funding disclosed; no issue located | C | exact PDF + DOI; 10-page preflight |
| C03 Li et al. | III | pass | warn, random validation unit and fit scope unclear | pass | no competing interests declared | C | exact PDF + DOI; 16-page preflight |
| C04 Wang et al. | III | pass | warn, best-of-10 and undefined validation/window phrase | pass | no competing interests declared | C | exact PDF + DOI; 12-page preflight |
| C05 Zhuang et al. | III | pass | warn, fit scope/cross-validation allocation/dispersion absent | pass | no issue located | C | author PDF + DOI; 11-page preflight |
| C06 Li et al. | III | pass | warn, test monitored each epoch and used for dropout comparison | pass | no competing interests declared | C | exact PDF + DOI; 12-page preflight |
| C07 Zhang et al. | III | pass | warn, all-window test denominator and no validation | pass | no competing interests declared | C | exact PDF + DOI; 13-page preflight |
| C08 Jin et al. | III | pass | warn, fold allocation and fit scope unclear | pass | no formal declaration located | C | author PDF + DOI; 10-page preflight |
| C09 Xu et al. | III | pass | warn, inconsistent sample counts and test-tuned reduction ratio | pass | no competing interests declared | C | exact PDF + DOI; 10-page preflight |
| C10 İnce et al. | III | pass | warn, best-of-300/50 without validation boundary | pass | HAVELSAN support acknowledged | C | entitled IEEE PDF + DOI; 4-page preflight |
| C11 Ghoneim et al. | III | pass | warn, table/scope/supplement conflicts | pass | no competing interests declared | C | PeerJ PDF + DOI; 31-page preflight |
| C12 Freitas & Lopes | III | pass | pass, objective changed in reproduction | pass | no issue located | B | SCITEPRESS PDF + DOI |
| C13 Özüpak | III | pass | warn, seed/scope/test-aggregation ambiguity | pass | no conflicts declared | C | exact Dicle bitstream + DOI; remote only |
| C14 Shamim et al. | III | warn, preprint | pass, chunk split risk | pass | no issue located | C | arXiv PDF + DOI |
| C15 Jeon et al. | III | pass | pass, seeds unnamed | pass | no issue located | B | MDPI PDF + DOI |
| C16 Xu et al. | III | pass | warn, window and split ambiguity | pass | no issue located | C | author PDF + DOI |
| C17 Xiang et al. | III | pass | warn, fit scope/validation/repetition/aggregation absent | pass | no competing interests declared | C | exact PDF + DOI; 14-page preflight |
| C18 Nejjar et al. | III | pass | warn, transductive/no test split | pass | no issue located | C | arXiv PDF + journal DOI |
| C21 Berghout et al. | III | pass | warn, target isolation unclear | pass | no issue located | C | MDPI PDF + DOI |
| C19 Chatterjee & Keprate | VI | pass | pass for EDA, not model evaluation | conditional | no issue located | B | repository PDF + DOI |
| C20 Arias Chao et al. | VI | pass | pass for dataset definition | conditional | no issue located | A | MDPI PDF + DOI |
| C22 Ramasso & Saxena | V | pass | pass for historical review | foundational | no issue located | B | PHM PDF + DOI |
| C23 Ramasso & Saxena | V | pass | pass for historical review | foundational | no issue located | B | PHM PDF + DOI |
| C24 Saxena et al. | VI | pass | pass for data-generation origin | foundational | no issue located | A | NASA NTRS PDF + DOI |
| C25 Liu & Paparrizos | III | pass | pass, adjacent domain only | pass | no issue located | A | NeurIPS PDF + DOI |

Grade distribution: A=3, B=5, C=17, D=0, F=0.

## Flagged Sources

### C01: validation and model-selection boundary is not reproducible

The paper reports all four C-MAPSS subsets and derives separate piecewise-linear
RUL plateaus of 78, 103, 79, and 87 from the training trajectories. It also
documents correlation thresholds and retained-feature counts. However, the
window value 12 is a non-overlapping label-knee detection parameter, not a
documented LSTM sample-window length. The temporal input packaging and stride
therefore remain unreported.

A validation set is named only as a symbol in Algorithm 2; no unit allocation,
fraction, or construction rule is supplied. The text states that hundreds of
MATLAB runs were executed and then selects the architecture, label window, and
threshold with the lowest reported RMSE and Score values. Because those values
are presented as test-set results and no separate validation procedure is
shown, reuse of the official test set for hyperparameter selection cannot be
ruled out. Z-score normalization is defined using each sensor output's mean and
standard deviation, but its fit scope is not stated and the framework diagram
places preprocessing before the train/test branch. No random seed, independent
repetition count, uncertainty interval, or implementation repository is given.

### C02: subset-wide preprocessing and missing input sequence

The paper defines monotonicity/correlation selection and min-max scaling using
statistics from each whole subset, without a train-only fit statement. This
cannot be promoted to a leakage-free preprocessing protocol. It selects 14
sensors and caps piecewise-linear RUL at 125, but does not report the model
input sequence length or stride. The printed kernel size 3 is a gated
convolutional-unit kernel, not a defensible substitute for the missing window.
Validation construction, seeds, repetitions, and variance are absent.

### C03: random validation allocation unit is not identified

The paper explicitly caps RUL at 125, uses all 21 sensors in nine
knowledge-derived maps, sets window length 40, and randomly assigns 5% of
training samples to validation. It does not state whether that random sample is
drawn by engine or by overlapping window, so same-engine train/validation
contamination remains possible. Normalization fit scope and seeds are absent.
Unlike the earlier interim coding, the full text does report ten runs with mean
and standard deviation; that correction is now reflected in the protocol table.

### C04: best-run result and ambiguous window statement

The official FD001 comparison runs the model ten times but visually emphasizes
the best RMSE/Score (13.22/232.24), not the means (13.77/269.88), and supplies no
SD/CI. A validation dataset is mentioned during tuning but never constructed.
The phrase "sliding time window step is set to 30" cannot establish whether 30
is the window length or stride. A separate complete-trajectory unit 1-80/81-100
maintenance experiment and a PHM08 experiment are distinct evaluation designs
and are not merged into the official-test protocol cell.

### C05: probabilistic passes are not independent replications

The paper states 20 simulations and 1,000 stochastic forward passes, but the
forward passes characterize predictive uncertainty rather than independently
trained model replications. It reports no seed list or run-level distribution
for the 20 simulations. Normalization fit scope, stride, and the allocation of
the cross-validation used for loss-weight tuning remain unreported. The official
test comparison and the separate complete-unit maintenance validation are kept
separate.

### C06: official test data are monitored during training

This paper is unusually clear that sensor min-max bounds are fitted on training
data and applied to test data. However, it evaluates test datasets at every
training epoch to plot UCS/RMSE and compares dropout settings 0.5, 0.7, and 0.9
with test metrics. The final result therefore carries direct test-monitoring and
model-selection risk. Validation construction, cap, stride, seed, and independent
training repetition count are not reported.

### C07: all-window test denominator differs from the standard endpoint policy

The paper lists 10,096/26,511/12,207/29,436 testing samples for FD001-FD004,
which are the overlapping windows generated from all test trajectories, not one
endpoint per engine. RMSE and Score therefore use a materially different
denominator from standard official-test comparisons. Stride 1 is exactly
recoverable from the sample counts, but the text itself only names a symbol
`l`. Validation, grid-search selection data, fit scope, seeds, and run-level
dispersion are not reported; only ten-trial means are shown.

### C08: five-fold training scheme is under-specified

The paper states that training data are divided into five folds and four folds
are used at each epoch. It does not explain fold rotation, the held-out fold's
role, or whether allocation is by engine or window. Normalization fit scope,
stride, seed, independent repetition count, and result dispersion are absent.
The author GitHub repository is recorded as code availability but is not treated
as proof that the paper's evaluation split is reproducible.

### C09: sample-count conflict and test-tuned attention ratio

The stated window lengths 31/30/60/50 are incompatible with the paper's own
training-sample counts and the standard C-MAPSS trajectory totals. In addition,
the reduction ratio is varied and `r=2` is selected from test RMSE/Score, while
no validation split is reported. This is an explicit test-set tuning pathway.
The normalization formula refers to extrema of all sensor data points without a
train-only fit boundary; seeds, repetitions, and dispersion are absent.

### C10: large best-of-search reporting without a validation boundary

The first genetic-algorithm stage trains 300 candidates and reports the best
performance. Subsequent stages carry forward 50 high-performing models, compare
two Gaussian-noise levels and pruning, and again report the best. The paper does
not define a validation or fitness-data split distinct from the official FD001
test data. The selected window is also not disclosed. These omissions create a
large best-of-search and possible test-reuse bias; the noise variance values are
regularization settings, not result-variance estimates. The local IEEE file is
user-entitled and institution-restricted and must not be redistributed.

### C11: internally inconsistent tables and unusable supplement

Tables 5 and 7 report an FD001 Score of 212.18, whereas Table 6 reports
217.18. The ablation prose claims comparisons among LSTM-only,
Attention-LSTM, and the full model, but its table contains only the final FD001
and FD003 dataset rows. The limitations paragraph says the study exclusively
uses FD001 even though the results repeatedly report FD003. Seed, repetition,
validation, normalization-fit, and test-window aggregation policies are absent.

The official Python supplement does not repair these omissions: feature and
sequence functions contain placeholders, the configured random state is never
applied, FD003 is absent, and test windows are passed to `model.fit` as
validation data. It also specifies 100 epochs with MSE loss, conflicting with
the paper's 120 epochs and Score loss. The official Markdown guide uses
`example.com` download placeholders and reports expected settings/outputs that
do not match either the paper or Python file. The code is therefore available
as an audit artifact, not as a reproducible implementation.

### C12: reproduction is not objective-preserving

The paper describes an AGATT reproduction but changes the original RMSE loss to
the asymmetric NASA Score loss. It remains useful for repeated-run and variance
reporting, but it cannot be treated as a pure reproduction of the original
training objective.

### C13: reproducibility wording conflicts

The paper reports an engine-wise 80/20 FD001 train/validation split, an unseen
official test set, train-only normalization fitting, and three-run mean/SD
results. These are useful protocol disclosures. However, the normalization
transform and final test-window/engine aggregation are not stated. The
experimental text says all experiments were seeded with 42, while the
robustness section says the three runs used different random seeds. A setup
passage also says all experiments used FD001, whereas the abstract and Table 2
report additional FD002-FD004 experiments. No implementation repository is
provided.

The exact Dicle institutional bitstream was remotely read and matched to the
article metadata, but browser, Dicle, and Wiley direct downloads returned HTTP
403. It is therefore coded as remote full text with no local hash or PDF
preflight, not as a locally acquired source.

### C14: preprint and chunk-level split

The paper is an arXiv preprint. Its leakage comparison is relevant, but the
reported chunk-level 70/15/15 split is not an engine-unit split. It is therefore
evidence about split sensitivity, not a compliant reference protocol.

### C16: window and N-CMAPSS split ambiguity

The setup section reports a grid optimum of 60, while the results section reports
FD002=70 and FD004=90. The N-CMAPSS 70/30 train/test split does not state whether
allocation occurred by engine unit. Both issues are retained as audit flags.

### C17: uncertainty intervals do not replace protocol replication

The BGT paper evaluates four C-MAPSS and nine N-CMAPSS subsets with explicit
windows 75/30, stride 1, caps 125/65, and prediction-interval metrics. It does
not report validation construction, scaler fit scope, seeds, independently
trained repetitions, the number of MC-dropout samples `T`, or the exact CMAPSS
endpoint aggregation. Its uncertainty intervals characterize predictive
uncertainty and cannot be coded as across-run variance. DS08d is excluded
because the authors could not open it; this exclusion is retained explicitly.

### C18: transductive target-data protocol

The paper explicitly states that all source and target domain data are used in
training and that no test-training split is used. Target labels are withheld,
but this remains a transductive design and is not evidence for an independent
test protocol.

### C19: dev/test combination is EDA-only

The paper combines development and test files for some exploratory analyses.
This is not coded as model leakage because the paper is a dataset EDA anchor,
not a trained-model evaluation. It must not be used to justify combining data
in the unified model grid.

### C21: target isolation is not documented

The transfer-learning paper fine-tunes on the N-CMAPSS target dataset, but does
not document an independent target test isolation or preprocessing fit scope.

## DOI And Identity Checks

- DOI or authoritative official metadata is verified for all 25 candidates.
- Twenty-four local PDF first pages were checked against title and author
  identity. Each accepted local file has a SHA-256 manifest row and a
  `pdf_read_preflight/1` sidecar with `verdict=PASS`.
- C13 was checked against the exact Dicle repository item and 20-page bitstream
  text. Its remote-only manifest row intentionally leaves hash/bytes blank and
  does not claim a local preflight.
- C03, C06, C07, C16, C17, C18, C21, and C22 have publication-year
  reconciliation notes where online-first, issue, or DOI-deposit years differ.
  The formal citation year is retained in the bibliography and protocol table.
- C10 is an exact, user-entitled IEEE Xplore PDF with an institutional license
  notice. It is valid for local audit extraction but not for redistribution.
- Europe PMC and repository false positives for C18/C19/C13, plus the
  Cloudflare HTML returned for the C16 Oulu bitstream, were rejected and logged.
  None is treated as evidence about the target paper.

## Predatory Journal And COI Checks

No inspected source showed the red flags available in the Phase 2 checklist:
missing publisher identity, implausible venue, or unverifiable DOI. The
verification is not a substitute for a live Scopus/WoS or COPE audit. Funding
and conflict statements were checked where the paper exposes them. A matrix
entry saying “no issue located” does not mean a formal declaration exists.

C03, C04, C06, C07, C09, C11, C13, and C17 explicitly declare no competing
interests. C02/C03/C04/C06/C07/C09/C17 disclose public funding or
acknowledgments; C10 acknowledges HAVELSAN support; C13 identifies a
TÜBİTAK-Wiley open-access agreement. C08 and several older anchors do not expose
a formal declaration in the inspected text.

## Claim-Level Verification Boundary

The current audit verifies source existence, identity, acquisition route, and
protocol-field extraction. It does not yet make cross-source claims about which
protocol is superior. Such claims require Phase 3 synthesis after the Phase 2
gate is explicitly closed and the user confirms the transition.

## Verification Limitations

- Semantic Scholar matching and body-snippet search were unavailable in the
  current paper-search configuration.
- Journal metrics were unavailable; venue labels are not converted into impact
  factor or quartile claims.
- C13 is the only remote-only full text; local acquisition remains blocked, so
  it has no local hash or preflight claim.
- C10 is locally auditable under the user's IEEE entitlement but cannot be
  distributed with the project.
- Quality grades are Phase 2 source-fitness judgments, not model rankings and
  not a substitute for the later cross-source risk-of-bias synthesis.
