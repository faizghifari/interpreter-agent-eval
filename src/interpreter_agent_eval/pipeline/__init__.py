"""Staged, resumable evaluation pipeline.

The monolithic per-sample loop (translate -> respond -> verify -> judge) is
decomposed into independent stages. Each stage reads a JSONL of work units and
writes a JSONL, accreting fields keyed by a stable ``record_id``. Stages are
idempotent and resumable: re-running skips units already present in the output.

Stage order::

    prepare    -> 00_units.jsonl
    translate  -> 01_translated.jsonl   (cloud LLM, batchable)
    respond    -> 02_responded.jsonl    (local user-sim, concurrent)
    verify     -> 03_verified.jsonl     (GlotLID, local/CPU)
    judge      -> 04_judged.jsonl       (cloud LLM, batchable)
    consolidate-> results.jsonl         (legacy run_eval schema)

Phase 1 ships a synchronous/threaded backend. The async batch-API backend
(OpenAI + Gemini) plugs in at the translate/judge stages without changing the
stage contracts.
"""
