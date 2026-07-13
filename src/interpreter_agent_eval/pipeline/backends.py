"""Execution backends for driving a per-record operation over many records.

A backend's only job is to apply ``fn(record) -> result`` across records and
hand each result to ``on_result`` for persistence. ``SyncBackend`` runs inline
(optionally across a thread pool, since the cloud/LM-Studio calls are I/O bound).

The async batch-API backend (OpenAI / Gemini ``/batches``) will implement the
same ``map`` contract in a later phase by splitting into submit + collect, so
stages need no changes to adopt it.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable


class SyncBackend:
    """Synchronous backend. ``concurrency > 1`` uses a thread pool.

    ``fn`` is expected not to raise (pipeline operations capture their own
    errors); if it does, the exception propagates out of ``map``.
    """

    def __init__(self, concurrency: int = 1):
        self.concurrency = max(1, int(concurrency))

    def map(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        records: Iterable[Dict[str, Any]],
        on_result: Callable[[Dict[str, Any]], None],
    ) -> None:
        records = list(records)
        if self.concurrency == 1:
            for rec in records:
                on_result(fn(rec))
            return

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = [ex.submit(fn, rec) for rec in records]
            for fut in as_completed(futures):
                on_result(fut.result())
