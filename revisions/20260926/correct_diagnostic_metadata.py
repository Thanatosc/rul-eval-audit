"""Correct a metadata field name while retaining original bytes and a correction log."""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
records = []
for directory in sorted((HERE / "diagnostics").iterdir()):
    path = directory / "meta.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    if "prediction_rows" in old:
        continue
    original = path.read_bytes()
    backup = directory / "meta_original_v1.json"
    assert not backup.exists()
    backup.write_bytes(original)
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    corrected = dict(old)
    corrected["prediction_rows"] = old["units"]
    corrected["units"] = {k: len(v) for k, v in split.items()}
    corrected["units"]["test"] = pd.read_parquet(directory / "preds_test.parquet").unit_id.nunique()
    corrected["metadata_correction"] = "2026-09-26: v1 units field counted windows. Values moved unchanged to prediction_rows; units now counts distinct engines. No predictions, checkpoints, curves or metrics modified."
    path.write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    records.append({"run_id": old["run_id"], "original_sha256": hashlib.sha256(original).hexdigest(),
                    "corrected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reason": corrected["metadata_correction"], "corrected_units": corrected["units"]})
if records:
    (HERE / "DIAGNOSTIC_METADATA_CORRECTION.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps({"metadata_records_corrected": len(records)}, indent=2))
