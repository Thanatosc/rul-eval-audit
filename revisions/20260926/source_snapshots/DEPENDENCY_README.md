# Reproduction dependency snapshot

This directory supplies the original model/data-loading source, selected split
manifests and historical prediction/table inputs needed by the corrective study,
"Training adequacy and evaluation populations in remaining useful life benchmarks."
It is a selected dependency snapshot, not a full copy of the historical repository.

Keep this directory beside `post_rejection_20260926/transfer_revision/` at the
same extraction root. The latter contains 96 primary grid fits, two diagnostics
and nine additional budget checks, with analysis code and Supplements S1, S2 and
S4. Follow those supplements for exact environment versions and reproduction.

NASA C-MAPSS archives, source sensor tables, literature full texts and private
project notes are not included. To retrain, obtain the official NASA files,
place them in `data/interim/cmapss/`, and verify the S2 hashes first.

The complete historical software v0.1.2 is preserved at
<https://doi.org/10.5281/zenodo.21915989> and the historical dataset v1.0.2 at
<https://doi.org/10.5281/zenodo.21915990>. Those historical records do not contain
the corrective or additional budget fits. Their native-target contrasts and weak
baseline rankings must be interpreted in light of the corrective study.
