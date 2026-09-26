# Literature Search Log

All searches must be appended with the exact query, source, date, filters, raw
result artifact, screening decision, and deduplication rule.

| Search ID | Date | Source | Query | Filters | Raw artifact | Notes |
|---|---|---|---|---|---|---|
| P0-001 | 2026-08-11 | Crossref | C-MAPSS evaluation practices audit benchmark inconsistency remaining useful life | max 10 | terminal result; see `NOVELTY_SCREEN.md` | broad novelty search |
| P0-002 | 2026-08-11 | OpenAlex | C-MAPSS evaluation practices audit benchmark inconsistency remaining useful life | max 10 | terminal result; see `NOVELTY_SCREEN.md` | broad novelty search |
| P0-003 | 2026-08-11 | Crossref/OpenAlex | C-MAPSS benchmark evaluation protocol | max 5-10 | terminal result; adjacent 2025-2026 records | duplicate-risk search |
| P0-004 | 2026-08-11 | Crossref | C-MAPSS sliding window leakage evaluation | max 10 | terminal result; full-text follow-up required | leakage/protocol search |
| P0-005 | 2026-08-11 | Crossref | C-MAPSS sensor selection normalization protocol | max 10 | terminal result; full-text follow-up required | preprocessing search |
| P0-006 | 2026-08-11 | NASA NTRS API | citation `20150007677` | exact identifier | verified JSON response; see `PHASE0_ANCHORS.md` | official anchor check |

## Phase 2 Candidate Corpus

The initial candidate set contains 25 records. It is a screening queue, not the
final included bibliography. See `candidate_corpus.csv` for per-record status;
full-text coding may exclude records that do not actually use C-MAPSS/N-CMAPSS or
do not expose the protocol fields required by the RQ.

## Phase 0 Anchor Checks

| Anchor | Identifier | Verification status | Evidence artifact |
|---|---|---|---|
| Saxena et al., PHM 2008 C-MAPSS paper | 10.1109/phm.2008.4711414 | verified | `PHASE0_ANCHORS.md` |
| NASA NTRS 20150007677 | NTRS 20150007677; 10.36001/phmconf.2014.v6i1.2512 | verified | `PHASE0_ANCHORS.md` |
| Arias Chao et al., N-CMAPSS dataset paper | 10.3390/data6010005 | verified | `PHASE0_ANCHORS.md` |
| Liu and Paparrizos, TSB-AD precedent | 10.52202/079017-3437 | verified | `PHASE0_ANCHORS.md` |

## Phase 2 Full-Text Discovery

All Phase 2 download attempts set `useSciHub=false` or used an official public
endpoint directly. Repository matches were accepted only when DOI or the full
title-and-author identity matched the candidate record.

| Search ID | Date | Source | Exact query or identifier | Filters | Raw artifact | Screening decision |
|---|---|---|---|---|---|---|
| P2-001 | 2026-08-11 | paper-search `get_paper_by_doi` | `10.5220/0015076000004091` | all configured metadata/OA sources | terminal JSON | C12 exact Crossref/OpenAlex/Unpaywall identity; CORE returned DOI `10.5220/0006086303470352` and was rejected |
| P2-002 | 2026-08-11 | OpenAIRE via paper-search | `10.5220/0015076000004091` | max 5 | terminal error | HTTP 409; no evidence used |
| P2-003 | 2026-08-11 | official-domain web search | `"A Statistical Analysis of the State-of-the-Art Transformer-Based Remaining Useful Life Models on C-MAPSS FD001" PDF` | domain `scitepress.org`, max 5 | terminal result | exact official PDF found and accepted |
| P2-004 | 2026-08-11 | paper-search `get_paper_by_doi` | `10.1109/phm.2008.4711414` | all configured metadata/OA sources | terminal JSON | C24 exact Crossref/OpenAlex identity; PMC/Europe PMC/CORE mismatches rejected |
| P2-005 | 2026-08-11 | OpenAIRE via paper-search | `10.1109/phm.2008.4711414` | max 5 | terminal error | HTTP 409; no evidence used |
| P2-006 | 2026-08-11 | official-domain web search | `"Damage propagation modeling for aircraft engine run-to-failure simulation" PDF` | domain `ntrs.nasa.gov`, max 5 | terminal result | exact NASA NTRS record `20090029214` PDF found and accepted |
| P2-007 | 2026-08-11 | paper-search `get_paper_by_doi` | `10.52202/079017-3437` | all configured metadata/OA sources | terminal JSON | C25 exact Crossref/OpenAlex/Unpaywall identity; CORE returned DOI `10.52202/079017-2302` and was rejected |
| P2-008 | 2026-08-11 | OpenAIRE via paper-search | `10.52202/079017-3437` | max 5 | terminal error | HTTP 409; no evidence used |
| P2-009 | 2026-08-11 | official-domain web search | `site:proceedings.neurips.cc/paper_files/paper/2024/hash Qinghua Liu John Paparrizos elephant room` | max 10 | terminal result | exact NeurIPS abstract page and official PDF link found and accepted |
| P2-010 | 2026-08-11 | Semantic Scholar via paper-search | `The Elephant in the Room Towards A Reliable Time-Series Anomaly Detection Benchmark` | max 10 | terminal error | HTTP 429; omitted rather than treated as an unmatched record |
| P2-011 | 2026-08-11 | arXiv via paper-search | `The Elephant in the Room Towards A Reliable Time-Series Anomaly Detection Benchmark` | max 10 | terminal JSON | zero results; official NeurIPS copy retained |

## Acquisition and Identity Outcomes

| Candidate | Discovery/acquisition route | Outcome | Evidence retained |
|---|---|---|---|
| C01 | Unpaywall/IEEE Xplore; initial endpoint HTTP 418, exact arnumber PDF URL then succeeded | verified, preflight PASS (16 pages) | manifest + sidecar + protocol coding; official CC-BY PDF |
| C11 | PeerJ official HTML -> official PDF -> exact supplements | verified, preflight PASS (31 pages); HTML and supplement identity matched | manifest + sidecar + protocol coding; supplements retained as reproducibility caveats |
| C13 | Dicle University exact item/bitstream plus Wiley endpoints | exact 20-page institutional PDF text remotely assessed; direct download HTTP 403 | protocol coding with explicit no-local-PDF/no-preflight limitation |
| C14 | arXiv native PDF | verified, preflight PASS (52 pages) | manifest + sidecar + protocol coding |
| C15 | MDPI official static PDF | verified, preflight PASS (21 pages) | manifest + sidecar + protocol coding |
| C20 | initial paper-search repository fallback | Europe PMC returned unrelated DOI `10.1038/s41598-026-40514-6`; rejected and removed | negative manifest row retained to expose the failure |
| C20 | MDPI official static PDF | verified, preflight PASS (14 pages) | manifest + sidecar + dataset-anchor coding |
| C21 | MDPI official static PDF | verified, preflight PASS (12 pages) | manifest + sidecar + protocol coding |
| C22 | PHM Society official PDF | verified, preflight PASS (15 pages) | manifest + sidecar + benchmark-anchor coding |
| C23 | PHM Society official PDF | verified, preflight PASS (11 pages) | manifest + sidecar + review-anchor coding |
| C12 | SCITEPRESS official proceedings PDF | verified, preflight PASS (8 pages) | manifest + sidecar + AGATT reproduction coding |
| C24 | NASA NTRS official PDF | verified, preflight PASS (9 pages) | manifest + sidecar + origin-anchor coding |
| C25 | NeurIPS official proceedings PDF | verified, preflight PASS (31 pages) | manifest + sidecar + adjacent-domain audit coding |
| C18 | arXiv author-posted PDF `2302.01704` | verified, preflight PASS (19 pages) | manifest + sidecar + transductive N-CMAPSS domain-adaptation coding |
| C19 | institutional repository copy (NVA/Sikt) | verified, preflight PASS (8 pages) | manifest + sidecar + N-CMAPSS dataset-support coding |
| C16 | MOSAIC author-laboratory PDF | verified, preflight PASS (12 pages) | manifest + sidecar + C-MAPSS/N-CMAPSS protocol coding |

The C20 false-positive incident established the current fail-closed rule: a
repository fallback is never accepted on title similarity alone. DOI must match
exactly when present; otherwise the first-page title and authors must match the
verified candidate metadata before any method claim is coded.

### Remaining-candidate OA sweep

| Search ID | Date | Source | Exact query or identifier | Filters | Raw artifact | Screening decision |
|---|---|---|---|---|---|---|
| P2-012 | 2026-08-11 | Unpaywall via paper-search | exact DOI, individually: `10.1007/s10845-021-01750-x`; `10.1016/j.ress.2022.108869`; `10.1016/j.ress.2024.110012`; `10.1016/j.ress.2023.109181`; `10.1016/j.jmsy.2023.11.009`; `10.1016/j.engappai.2023.107241`; `10.1109/tim.2022.3167778`; `10.1016/j.measurement.2023.113098`; `10.1109/siu55565.2022.9864796`; `10.1109/tvt.2023.3319377`; `10.1016/j.eswa.2023.121859`; `10.1016/j.ress.2023.109718`; `10.1109/ieem50564.2021.9673064` | one result per DOI | terminal JSON | C18 returned CC-BY hybrid publisher landing page; C19 returned green repository landing page; other 11 records reported closed |
| P2-013 | 2026-08-11 | paper-search `download_with_fallback` | DOI `10.1016/j.ress.2023.109718`, exact title, `useSciHub=false` | official/repository/OA only | downloaded candidate + first-page check | Europe PMC returned unrelated 2026 SiC MOSFET review; rejected and removed |
| P2-014 | 2026-08-11 | paper-search `download_with_fallback` | DOI `10.1109/ieem50564.2021.9673064`, exact title, `useSciHub=false` | official/repository/OA only | downloaded candidate + first-page check | Europe PMC returned unrelated PeerJ article DOI `10.7717/peerj-cs.1712`; rejected and removed |
| P2-015 | 2026-08-11 | arXiv official PDF | `2302.01704`; DOI `10.1016/j.ress.2023.109718` | exact title/author identity; no Sci-Hub | `papercorpus/pdfs/C18_operation_profile_domain_adaptation.pdf` + preflight sidecar | accepted author-posted full text; DOI/title/authors match; 19/19 pages readable |
| P2-016 | 2026-08-11 | Norwegian national research information repository | DOI `10.1109/ieem50564.2021.9673064`; exact title | repository record and first-page DOI/title identity | `papercorpus/pdfs/C19_n_cmapss_eda.pdf` + preflight sidecar | accepted institutional copy; 8/8 pages readable; dataset-support anchor |
| P2-017 | 2026-08-11 | arXiv via paper-search | exact titles for C02-C10, C16, and C17 | max 5 per title | terminal JSON | no exact matches; C03 returned five unrelated records and all were rejected by title identity |
| P2-018 | 2026-08-11 | exact-title web search | C02-C10, C16, and C17 title + PDF | official, author, or institutional copies only; Sci-Hub excluded | terminal search results | exact Oulu repository and author-lab copies found for C16; C08 only IEEE/ResearchGate/reproduction-code leads; C10 repository lead had a non-matching title; no other acceptable exact full text found |
| P2-019 | 2026-08-11 | Oulu repository bitstream | exact C16 title | first-byte PDF check | 4.5 KB Cloudflare HTML | rejected and removed; not entered as a PDF manifest row |
| P2-020 | 2026-08-11 | MOSAIC author laboratory | exact C16 title and authors | public author PDF; no Sci-Hub | `papercorpus/pdfs/C16_multi_resolution_lstm.pdf` + preflight sidecar | accepted; title/authors match, DOI triangulated with OpenAlex, 12/12 pages readable |
| P2-021 | 2026-08-11 | PeerJ official article page | DOI `10.7717/peerj-cs.3438`; exact title | publisher HTML full text and citation metadata | `https://peerj.com/articles/cs-3438/` | C11 title, four authors, DOI, full article text, and official PDF URL matched; HTML accepted for method cross-check |
| P2-022 | 2026-08-11 | PeerJ official PDF | `https://peerj.com/articles/cs-3438.pdf` | browser request HTTP 200; CC-BY 4.0; no Sci-Hub | `papercorpus/pdfs/C11_attention_dab_lstm_peerj.pdf` + preflight sidecar | accepted; first page title/authors/DOI match; 31/31/31 page-count signals agree; SHA-256 recorded in manifest |
| P2-023 | 2026-08-11 | PeerJ official supplements | DOI suffixes `supp-1` and `supp-2` | exact article-linked Markdown and Python files | official CloudFront supplement URLs on PeerJ page | accepted as official supplements but not as executable reproduction code: Python contains placeholder features/sequences, uses test as validation, and conflicts with paper hyperparameters; Markdown uses `example.com` placeholders and inconsistent expected outputs |
| P2-024 | 2026-08-11 | Dicle University institutional repository | C13 exact title; DOI `10.1155/er/7022062`; handle `11468/33267` | item metadata plus exact 2.42 MB bitstream; no Sci-Hub | item `5e96754d-5a2e-4030-ae48-b10d7c47349f`; bitstream `2609b5bf-4d16-4f23-ba2c-131e3118c1b8` | C13 item title, author, publisher, OA status, DOI route, and 20-page PDF identity match; managed remote extraction returned complete text; browser/direct download remains HTTP 403, so no local hash or preflight is claimed |
| P2-025 | 2026-08-11 | Wiley official endpoints | DOI `10.1155/er/7022062`; `epdf`, `pdfdirect`, and `pdf` routes | publisher routes only | HTTP 403 responses | no local PDF acquired; exact Dicle remote full text retained with degraded verification status |
| P2-026 | 2026-08-11 | paper-search `download_with_fallback` | C13 DOI and exact title; `useSciHub=false` | official/repository/OA only | downloaded Europe PMC candidate | returned PMID `41471631`, a 67-page *Sensors* review titled *Artificial Intelligence of Things for Next-Generation Predictive Maintenance*; title, author, and DOI mismatch; rejected and removed; no manifest hash fabricated |
| P2-027 | 2026-08-11 | IEEE Xplore official PDF | DOI `10.1109/ACCESS.2022.3203406`; arnumber `9874872` | exact publisher PDF, CC-BY 4.0; no Sci-Hub | `papercorpus/pdfs/C01_deep_learning_model_ieee_access.pdf` + preflight sidecar | accepted; title, six authors, DOI, and license match; 16/16/16 page-count signals agree; SHA-256 recorded in manifest |
| P2-028 | 2026-08-11 | user-provided lawful copy | C02 DOI `10.1007/s10845-021-01750-x` | exact first-page title/author/journal/DOI plus PDF preflight | `papercorpus/pdfs/s10845-021-01750-x.pdf` + preflight sidecar | accepted; 10/10/10 pages; SHA-256 recorded; redistribution status not inferred |
| P2-029 | 2026-08-11 | user-provided lawful copy | C03 DOI `10.1016/j.ress.2022.108869` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C03_knowledge_deep_learning.json` | accepted; 16/16/16 pages; SHA-256 recorded |
| P2-030 | 2026-08-11 | user-provided lawful copy | C04 DOI `10.1016/j.ress.2024.110012` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C04_dynamic_pdm_ensemble.json` | accepted; 12/12/12 pages; SHA-256 recorded |
| P2-031 | 2026-08-11 | user-provided lawful copy | C06 DOI `10.1016/j.jmsy.2023.11.009` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C06_sensor_aware_capsnet.json` | accepted; 12/12/12 pages; SHA-256 recorded |
| P2-032 | 2026-08-11 | user-provided lawful copy | C07 DOI `10.1016/j.engappai.2023.107241` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C07_attention_tcn.json` | accepted; 13/13/13 pages; SHA-256 recorded |
| P2-033 | 2026-08-11 | author personal website | C08 DOI `10.1109/TIM.2022.3167778`; exact title | author-hosted manuscript and first-page identity | `papercorpus/pdfs/C08_bilstm_two_stream.pdf` + preflight sidecar | accepted; 10/10/10 pages; author code repository retained |
| P2-034 | 2026-08-11 | user-provided lawful copy | C09 DOI `10.1016/j.measurement.2023.113098` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C09_global_attention.json` | accepted; 10/10/10 pages; SHA-256 recorded |
| P2-035 | 2026-08-11 | user-entitled IEEE Xplore copy | C10 DOI `10.1109/SIU55565.2022.9864796` | exact title/author/conference/DOI; institutional license notice | local long-title PDF + `C10_joint_autoencoder_regression.json` | accepted for local audit; 4/4/4 pages; must not be redistributed |
| P2-036 | 2026-08-11 | user-provided lawful copy | C17 DOI `10.1016/j.eswa.2023.121859` | exact first-page title/author/journal/DOI plus PDF preflight | local long-title PDF + `C17_bayesian_gated_transformer.json` | accepted; 14/14/14 pages; SHA-256 recorded |

After P2-013/P2-014, generic repository fallback is disabled for the remaining
engineering candidates. Further acquisition uses only an exact DOI-bearing
repository record, an official publisher/proceedings URL, or an author/institutional
copy whose first-page identity matches the verified metadata.

## Phase 2 Full-Text Closure

All 25 candidate records now have accepted exact full text: 24 local PDFs with
`pdf_read_preflight/1` PASS and one exact remote institutional PDF (C13) without
a local preflight. The included set contains 19 model/protocol papers and six
labelled anchors. `full_texts_not_retrieved=0`; no acquisition request queue
remains open.
