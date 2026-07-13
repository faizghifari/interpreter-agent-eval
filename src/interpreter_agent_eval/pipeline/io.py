"""JSONL I/O and resume helpers shared by all pipeline stages."""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def record_id(sample: Dict[str, Any]) -> str:
    """Stable unique key for a record — survives file re-ordering or growth.

    Order of preference:
      1. ``{segment_id}_{direction}`` (OpenSubtitles-sourced data) — matches the
         legacy ``run_eval._record_id``.
      2. an upstream ``record_id``.
      3. a content hash, for datasets carrying none of the above (e.g. the MAPS
         proverb data). Without this, every such record keyed to the empty
         string — silently breaking resume and collapsing batch result-mapping.
    """
    seg = sample.get("segment_id", "")
    direction = sample.get("direction", "")
    if seg or direction:
        return f"{seg}_{direction}"
    rid = sample.get("record_id")
    if rid:
        return rid
    src = sample.get("source_language_code", "") or ""
    tgt = sample.get("target_language_code", "") or ""
    basis = "|".join(
        str(sample.get(k, ""))
        for k in (
            "source_language_code",
            "target_language_code",
            "seed_file",
            "seed_row_id",
            "source_text",
        )
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"{src}_{tgt}_{digest}"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping blank/malformed lines. Missing file -> []."""
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_done_ids(path: str, id_field: str = "record_id") -> Set[str]:
    """Collect the set of ``id_field`` values already written to ``path``."""
    done: Set[str] = set()
    for entry in read_jsonl(path):
        rid = entry.get(id_field)
        if rid:
            done.add(rid)
    return done


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python for JSON.

    No-op (and no import cost beyond the first call) when numpy isn't involved.
    """
    try:
        import numpy as np
    except ImportError:
        return obj

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class JsonlWriter:
    """Append-only JSONL writer, safe for concurrent appends from a thread pool."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(convert_numpy_types(record), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    """Write ``records`` to ``path`` (overwriting). Returns count written."""
    with JsonlWriter(path) as writer:
        # JsonlWriter opens in append mode; truncate first for a clean rewrite.
        writer._fh.seek(0)
        writer._fh.truncate()
        n = 0
        for rec in records:
            writer.append(rec)
            n += 1
    return n
