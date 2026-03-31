"""Fix HuggingFace Datasets parquet metadata for LeRobot datasets.

Some LeRobot-generated parquet files embed HuggingFace Datasets feature metadata that uses
`{"_type": "List", ...}` for fixed-length vectors. Newer `datasets` versions may not
be able to parse this and will crash while loading the parquet.

This script rewrites the parquet footer metadata by converting feature `_type` values
from "List" to "Sequence" inside the embedded `huggingface` JSON metadata.

Usage:
  uv run scripts/fix_lerobot_parquet_features.py --dataset-root /path/to/dataset

Then rerun:
  uv run scripts/compute_norm_stats.py --config-name <your-config>
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tyro


def _rewrite_list_to_sequence(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_obj = {k: _rewrite_list_to_sequence(v) for k, v in obj.items()}
        if new_obj.get("_type") == "List":
            new_obj["_type"] = "Sequence"
        return new_obj
    if isinstance(obj, list):
        return [_rewrite_list_to_sequence(v) for v in obj]
    return obj


def _fix_one_parquet(path: Path) -> bool:
    schema = pq.read_schema(path)
    schema_kv = dict(schema.metadata or {})

    hf = schema_kv.get(b"huggingface")
    if hf is None:
        return False

    hf_json = json.loads(hf.decode("utf-8"))
    fixed = _rewrite_list_to_sequence(hf_json)
    if fixed == hf_json:
        return False

    schema_kv[b"huggingface"] = json.dumps(fixed, separators=(",", ":")).encode("utf-8")
    schema = schema.with_metadata(schema_kv)

    parquet_file = pq.ParquetFile(path)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    writer = pq.ParquetWriter(tmp_path, schema)
    try:
        for batch in parquet_file.iter_batches(batch_size=8192):
            batch = batch.replace_schema_metadata(schema.metadata)
            writer.write_batch(batch)
    finally:
        writer.close()

    os.replace(tmp_path, path)
    return True


def main(dataset_root: Path) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {dataset_root}/data")

    changed = 0
    for p in parquet_files:
        if _fix_one_parquet(p):
            changed += 1

    print(f"Scanned {len(parquet_files)} parquet files; updated {changed} files.")


if __name__ == "__main__":
    tyro.cli(main)
