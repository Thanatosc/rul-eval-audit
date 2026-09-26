# Supplement S1. Literature corpus, coding and source trace

## S1.1 Scope and status

This supplement records how the bounded literature-practice audit was searched,
screened, acquired, and coded. The audit was designed to identify protocol
variation and reporting gaps that could motivate controlled re-benchmarking. It
was not designed to estimate field-wide prevalence, and it is not presented as a
PRISMA 2020-compliant systematic review. The analytical denominator is fixed at
19 contemporary model/protocol papers; six dataset, historical, review, or
adjacent-domain sources are retained as labelled anchors outside that
denominator.

The original search and acquisition state was frozen on 11 August 2026; that corpus and its archived source-status records are retained unchanged in this revision. All 25
candidate records had an accepted exact full text at closure: 24 local PDFs with
identity verification and `pdf_read_preflight/1` PASS, plus one exact remote
institutional PDF (C13) without a local file hash or page-preflight claim.

## S1.2 Sources, dates, and search concepts

Metadata discovery used Crossref and OpenAlex. DOI-level open-access status was
checked through Unpaywall. Full-text acquisition and identity checks used arXiv,
NASA NTRS, publisher pages and static assets, author or institutional
repositories, exact-title web search, and, when necessary, lawful copies
provided or accessed by the author. CORE and OpenAIRE were used only when an
exact identity could be established. Generic repository fallback was disabled
after documented false-positive files, and Sci-Hub was excluded. Restricted
publisher files were used only for local audit and are not redistributable.

The final concept query was:

```text
(C-MAPSS OR CMAPSS OR N-CMAPSS)
AND (remaining useful life OR RUL OR prognostics)
AND (evaluation OR benchmark OR protocol OR leakage OR window
     OR normalization OR sensor selection OR score)
```

Acquisition follow-up combined each exact title or DOI with the first author and
terms such as `PDF`, `repository`, and `arXiv`. The route-level log in
`literature/SEARCH_LOG.md` records provider, date, exact query or identifier,
filters, retained artifact, and screening decision for every search action. The
machine-readable candidate register retains each record's source query in
`literature/candidate_corpus.csv`.

## S1.3 Eligibility and screening decisions

A contemporary model paper was eligible when it:

1. evaluated RUL on C-MAPSS or N-CMAPSS;
2. supplied enough full text to code at least one evaluation-protocol field;
3. had title, author, and DOI identity verified against authoritative metadata;
   and
4. was available through an official, open-access, author, institutional, or
   otherwise lawful full-text route.

A source was retained as a labelled anchor when it defined the dataset or
metric, reviewed historical C-MAPSS benchmarking, documented relevant data
quality, or supplied a directly relevant benchmark-audit precedent. Candidate
records would have been excluded or deferred for a mismatched full-text
identity, abstract-only access, an unrelated research question, or absence of
an acceptable full-text route. Access failure alone was never coded as absence
of a protocol detail. No record in the fixed 25-record candidate register met an
exclusion condition at closure, so the observed exclusion count was zero.

| Screening stage | Count | Disposition |
|---|---:|---|
| Records identified | 25 | Fixed candidate register |
| Duplicates removed | 0 | DOI and normalized-title checks complete |
| Title/abstract records screened | 25 | Complete |
| Full texts sought | 25 | Complete |
| Full texts not retrieved | 0 | Acquisition gap closed |
| Full texts assessed and coded | 25 | 24 local preflight PASS; one exact remote full text |
| Model/protocol papers | 19 | Main analytical denominator: C01-C18 and C21 |
| Labelled anchors | 6 | C19, C20, and C22-C25; outside the main denominator |

These counts are an acquisition-stage screening analogue only. They must not be
read as a comprehensive search yield or a prevalence denominator for all RUL
research.

## S1.4 Coding rules

The primary protocol codebook records dataset coverage, target form and cap,
sensor set and selection rule, window length and stride, normalization and fit
population, test-prediction denominator, validation boundary, metrics, seeds,
independent repetitions, run-level dispersion, code availability, and
verification status. The analysis codebook converts these extractions into the
fixed categorical variables used in the manuscript tables. The UQ extension
preserves the same 19-paper denominator and adds whether predictive UQ was
reported, its method, and whether an independent empirical coverage or
calibration check satisfied the frozen rule.

`NR`, `unclear`, and `not confirmed` are publication-record classifications.
They do not assert that an unreported implementation choice was absent or
invalid. One human author performed screening and final coding with AI-assisted
extraction. Deterministic checks tested allowed values and internal count
consistency, but no second coder was used and no inter-rater reliability is
claimed.

## S1.5 Reproducibility files

| Function | Project-relative artifact | Contents |
|---|---|---|
| Search design | `literature/PHASE2_SEARCH_STRATEGY.md` | Sampling frame, routes, concepts, eligibility, closure |
| Search execution | `literature/SEARCH_LOG.md` | Exact provider queries/identifiers, dates, filters, outcomes |
| Screening register | `literature/candidate_corpus.csv` | C01-C25 identity, source query, relevance, decision state |
| Screening counts | `literature/PRISMA_COUNTS.csv` | Machine-readable 25 to 19+6 flow |
| Full-text provenance | `literature/FULLTEXT_MANIFEST.csv` | Acquisition route, access basis, local path, hashes, read status |
| Primary extraction | `literature/protocol_coding.csv` | Paper-level protocol fields and verification notes |
| Analysis codebook | `literature/PHASE3_ANALYSIS_CODEBOOK.csv` | Frozen categorical variables for the 19-paper analysis |
| UQ extraction | `literature/protocol_coding_uq_v2.csv` | UQ reporting, method, validation, and evidence anchors |
| Source-quality gate | `literature/SOURCE_VERIFICATION_REPORT.md` | Identity checks, source fitness, caveats, acquisition limits |

The software/data release excludes licensed literature PDFs and excludes the
NASA archive by default because local possession does not establish
redistribution rights. The registers, coding tables, provenance records, and
scripts required to inspect the audit decisions are the intended public
artifacts.


## S1.6 Complete registered corpus

| ID | Role | Title | DOI |
|---|---|---|---|
| C01 | Model/protocol | A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset | <https://doi.org/10.1109/access.2022.3203406> |
| C02 | Model/protocol | Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit | <https://doi.org/10.1007/s10845-021-01750-x> |
| C03 | Model/protocol | Remaining useful life prediction of aero-engine enabled by fusing knowledge and deep learning models | <https://doi.org/10.1016/j.ress.2022.108869> |
| C04 | Model/protocol | Dynamic predictive maintenance strategy for system remaining useful life prediction via deep learning ensemble method | <https://doi.org/10.1016/j.ress.2024.110012> |
| C05 | Model/protocol | A prognostic driven predictive maintenance framework based on Bayesian deep learning | <https://doi.org/10.1016/j.ress.2023.109181> |
| C06 | Model/protocol | Sensor-aware CapsNet: Towards trustworthy multisensory fusion for remaining useful life prediction | <https://doi.org/10.1016/j.jmsy.2023.11.009> |
| C07 | Model/protocol | An attention-based temporal convolutional network method for predicting remaining useful life of aero-engine | <https://doi.org/10.1016/j.engappai.2023.107241> |
| C08 | Model/protocol | Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction | <https://doi.org/10.1109/tim.2022.3167778> |
| C09 | Model/protocol | Global attention mechanism based deep learning for remaining useful life prediction of aero-engine | <https://doi.org/10.1016/j.measurement.2023.113098> |
| C10 | Model/protocol | Remaining Useful Life Prediction on C-MAPSS Dataset via Joint Autoencoder-Regression Architecture | <https://doi.org/10.1109/siu55565.2022.9864796> |
| C11 | Model/protocol | Remaining useful life prediction for turbofan engines using an attention-based data-driven deep-learning approach | <https://doi.org/10.7717/peerj-cs.3438> |
| C12 | Model/protocol | A Statistical Analysis of the State-of-the-Art Transformer-Based Remaining Useful Life Models on C-MAPSS FD001 | <https://doi.org/10.5220/0015076000004091> |
| C13 | Model/protocol | A Hybrid CNN-GRU Approach for Robust and Explainable Remaining Useful Life Prediction of Turbofan Engines | <https://doi.org/10.1155/er/7022062> |
| C14 | Model/protocol | Leakage-Robust Evaluation and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning for Joint Fault Diagnosis and Remaining Useful Life Estimation | <https://doi.org/10.48550/arxiv.2607.16493> |
| C15 | Model/protocol | Effects of Window and Batch Size on Autoencoder-LSTM Models for Remaining Useful Life Prediction | <https://doi.org/10.3390/machines14020135> |
| C16 | Model/protocol | Multi-Resolution LSTM-Based Prediction Model for Remaining Useful Life of Aero-Engine | <https://doi.org/10.1109/tvt.2023.3319377> |
| C17 | Model/protocol | Bayesian gated-transformer model for risk-aware prediction of aero-engine remaining useful life | <https://doi.org/10.1016/j.eswa.2023.121859> |
| C18 | Model/protocol | Domain adaptation via alignment of operation profile for Remaining Useful Lifetime prediction | <https://doi.org/10.1016/j.ress.2023.109718> |
| C19 | Contextual anchor | Exploratory Data Analysis of the N-CMAPSS Dataset for Prognostics | <https://doi.org/10.1109/ieem50564.2021.9673064> |
| C20 | Contextual anchor | Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics | <https://doi.org/10.3390/data6010005> |
| C21 | Model/protocol | ProgNet: A Transferable Deep Network for Aircraft Engine Damage Propagation Prognosis under Real Flight Conditions | <https://doi.org/10.3390/aerospace10010010> |
| C22 | Contextual anchor | Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets | <https://doi.org/10.36001/ijphm.2014.v5i2.2236> |
| C23 | Contextual anchor | Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset | <https://doi.org/10.36001/phmconf.2014.v6i1.2512> |
| C24 | Contextual anchor | Damage propagation modeling for aircraft engine run-to-failure simulation | <https://doi.org/10.1109/phm.2008.4711414> |
| C25 | Contextual anchor | The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark | <https://doi.org/10.52202/079017-3437> |

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
