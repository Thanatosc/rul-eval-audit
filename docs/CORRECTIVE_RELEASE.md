# Corrective release and provenance

The manuscript primary grid remains 96 fits; all original curves, checkpoints,
predictions and specification are preserved. Nine additional fits address all
recorded final-cap flags. They restart from the same initialization with a
threefold budget, reproduce the original trajectory/prediction prefix, and keep
all validation selection and early-stopping rules. All nine stop before the new
cap. The substituted sensitivity grid is a view of 96 cells, not 96 new fits.

The maximum checked original prediction-prefix difference and the saved-model
reload difference are both zero. Independent verification covers 408,732 new
prediction rows, 54 scores and 36 paired endpoint intervals. Rankings and target
preferences are unchanged; the finite-budget result does not prove equivalence
or global convergence.

Primary numerical tables/figures are retained. Budget results are in Supplement
S4 and `budget_sensitivity/analysis/`; sensor/latency results are in S5 and
`review_followups/analysis/`. Sensor perturbations change the FD004 reference
winner and are not field-calibrated noise models. Test-engine uncertainty
conditions on fitted models; partitions, initialization contexts and tree
references do not supply additional independent physical systems.

The raw sensor inputs and copyrighted literature full texts are intentionally
obtained separately. Dataset v1.1.0 includes 20 historical test-prediction inputs
needed by the corrective analysis. Consult dataset v1.0.2 for the entire original
120/160/280-cell study. Its old model rankings and native-truth contrasts must
be interpreted with the corrective evidence described above.
