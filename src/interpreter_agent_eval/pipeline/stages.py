"""Stage runners: resumable drivers that wire an operation to a backend + I/O.

Each ``run_*`` reads an input JSONL, applies its operation only to records not
already present in the output (resume), and appends results via a thread-safe
writer. Stage filenames in a run directory are fixed so the chain is obvious.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from .backends import SyncBackend
from .batch import (
    BatchRequest,
    BatchClient,
    build_batch_client,
    FAILED,
    TERMINAL,
)
from .io import JsonlWriter, load_done_ids, read_jsonl
from . import operations as ops
from . import registry

# Canonical artifact names within a run directory.
UNITS = "00_units.jsonl"
TRANSLATED = "01_translated.jsonl"
RESPONDED = "02_responded.jsonl"
VERIFIED = "03_verified.jsonl"
JUDGED = "04_judged.jsonl"
RESULTS = "results.jsonl"


# Per-stage "success" predicate: a record is written to a stage's output ONLY
# when the stage actually produced its output. Failures and waiting-for-upstream
# skips are not written, so they aren't counted as done and reprocess on the next
# run once their inputs exist — giving correct resume + cascade with no strip
# logic and no duplicate lines. Records that can't produce go to a .pending
# sidecar for inspection.
def _produced_translate(r: Dict[str, Any]) -> bool:
    return bool(r.get("translated_text"))


def _produced_respond(r: Dict[str, Any]) -> bool:
    return bool(r.get("user_b_response"))


def _produced_verify(r: Dict[str, Any]) -> bool:
    # verify is meaningful only with a translation to check.
    return bool(r.get("translated_text"))


def _produced_judge(r: Dict[str, Any]) -> bool:
    return r.get("evaluation") is not None


def _drive(
    label: str,
    records: List[Dict[str, Any]],
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    output_path: str,
    concurrency: int = 1,
    resume: bool = True,
    id_field: str = "record_id",
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    """Apply ``fn`` to every not-yet-done record; write only successes to output.

    ``produced`` decides success; records that fail it are routed to a
    ``<output>.pending.jsonl`` sidecar instead of the main output (so they remain
    not-done and reprocess on the next run).
    """
    # resume=False stages (prepare, consolidate) must rewrite, not append —
    # otherwise re-running duplicates every record into the output file.
    if not resume and os.path.exists(output_path):
        os.remove(output_path)
    done = load_done_ids(output_path, id_field) if resume else set()
    todo = [r for r in records if r.get(id_field) not in done]
    print(
        f"[{label}] {len(records)} total | {len(done)} already done | {len(todo)} to process "
        f"(concurrency={concurrency}) -> {output_path}"
    )
    if not todo:
        return output_path

    pending_path = output_path + ".pending.jsonl"
    if os.path.exists(pending_path):
        os.remove(pending_path)  # rebuilt fresh each run from this run's failures

    pending = {"n": 0}
    backend = SyncBackend(concurrency)
    with JsonlWriter(output_path) as writer, JsonlWriter(pending_path) as pwriter:

        def on_result(out: Dict[str, Any]) -> None:
            if produced is None or produced(out):
                writer.append(out)
            else:
                pwriter.append(out)
                pending["n"] += 1

        backend.map(fn, todo, on_result)
    if pending["n"]:
        print(
            f"[{label}] {pending['n']} not produced (failed/awaiting upstream) -> {pending_path}"
        )
    else:
        os.remove(pending_path)
    return output_path


# ---------------------------------------------------------------------------
# Async batch driver (translate / judge only)
#
# A batch run is submit -> poll -> collect. The job id is persisted to a sidecar
# (``<output>.batch.json``) so you can submit now and collect later — even from
# a fresh process. Re-invoking the stage with backend="batch" resumes: if the
# sidecar exists it polls/collects that job instead of submitting a new one.
# ---------------------------------------------------------------------------
def _sidecar_path(output_path: str) -> str:
    return output_path + ".batch.json"


def _drive_batch(
    label: str,
    records: List[Dict[str, Any]],
    build_request_fn: Callable[[Dict[str, Any]], Optional[BatchRequest]],
    apply_fn: Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]],
    client: BatchClient,
    provider_type: str,
    model: str,
    output_path: str,
    resume: bool = True,
    id_field: str = "record_id",
    wait: bool = True,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    sidecar = _sidecar_path(output_path)
    job: Optional[Dict[str, Any]] = None
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            job = json.load(f)
        print(f"[{label}] resuming batch job {job['job_id']} ({job['provider']})")

    if job is None:
        done = load_done_ids(output_path, id_field) if resume else set()
        todo = [r for r in records if r.get(id_field) not in done]
        print(
            f"[{label}] {len(records)} total | {len(done)} done | {len(todo)} to submit"
        )
        # Records that yield no prompt (e.g. awaiting an upstream translation)
        # can't batch — they aren't written to the main output, so they stay
        # not-done and reprocess once their input exists.
        requests = [
            req for req in (build_request_fn(r) for r in todo) if req is not None
        ]
        if not requests:
            print(f"[{label}] nothing batchable -> {output_path}")
            return output_path
        job_id = client.submit(requests, model)
        job = {
            "job_id": job_id,
            "provider": provider_type,
            "model": model,
            "order": [req.custom_id for req in requests],
        }
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(job, f)
        print(f"[{label}] submitted {len(requests)} reqs as {job_id} -> {sidecar}")

    if not wait:
        print(f"[{label}] submitted; not waiting. Collect later (batch-collect).")
        return output_path

    started = time.time()
    state = client.poll(job["job_id"])
    while state not in TERMINAL:
        if timeout is not None and time.time() - started > timeout:
            print(f"[{label}] poll timeout; job still running. Collect later.")
            return output_path
        print(
            f"[{label}] job {job['job_id']} state={state}; waiting {poll_interval:.0f}s"
        )
        time.sleep(poll_interval)
        state = client.poll(job["job_id"])

    if state == FAILED:
        print(f"[{label}] batch FAILED: {job['job_id']} (sidecar kept for inspection)")
        return output_path

    return _collect_batch(
        label, records, job, apply_fn, client, output_path, resume, id_field, produced
    )


def _collect_batch(
    label: str,
    records: List[Dict[str, Any]],
    job: Dict[str, Any],
    apply_fn: Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]],
    client: BatchClient,
    output_path: str,
    resume: bool,
    id_field: str,
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    order = job.get("order", [])
    req_stubs = [BatchRequest(custom_id=cid, prompt="") for cid in order]
    results = client.collect(job["job_id"], req_stubs)
    in_job = set(order)
    done = load_done_ids(output_path, id_field) if resume else set()
    pending_path = output_path + ".pending.jsonl"
    written = 0
    pend = 0
    with JsonlWriter(output_path) as w, JsonlWriter(pending_path) as pw:
        for r in records:
            rid = r.get(id_field)
            if rid in done or rid not in in_job:
                continue
            out = apply_fn(r, results.get(rid))
            if produced is None or produced(out):
                w.append(out)
                written += 1
            else:
                pw.append(out)
                pend += 1
    if not pend:
        try:
            os.remove(pending_path)
        except OSError:
            pass
    try:
        os.remove(_sidecar_path(output_path))
    except OSError:
        pass
    print(
        f"[{label}] collected {written} record(s)"
        + (f", {pend} failed -> {pending_path}" if pend else "")
        + f" -> {output_path}"
    )
    return output_path


def _translate_request(
    record: Dict[str, Any], thinking_level: str
) -> Optional[BatchRequest]:
    system, user = ops.build_translate_request(record)
    if user is None:
        return None
    return BatchRequest(
        custom_id=record["record_id"],
        prompt=user,
        system=system,
        config={"thinking_level": thinking_level},
    )


def _judge_request(
    record: Dict[str, Any], thinking_level: str
) -> Optional[BatchRequest]:
    prompt = ops.build_judge_prompt(record)
    if prompt is None:
        return None
    return BatchRequest(
        custom_id=record["record_id"],
        prompt=prompt,
        config={"json": True, "thinking_level": thinking_level},
    )


# ---------------------------------------------------------------------------
# Stage 0: prepare
# ---------------------------------------------------------------------------
def run_prepare(
    data_files: List[str],
    output_path: str,
    num_samples: Optional[int] = None,
    filter_target_lang: Optional[str] = None,
) -> str:
    units: List[Dict[str, Any]] = []
    for path in data_files:
        recs = read_jsonl(path)
        if filter_target_lang:
            recs = [
                r for r in recs if r.get("target_language_code") == filter_target_lang
            ]
        recs = recs[:num_samples] if num_samples is not None else recs
        base = os.path.basename(path)
        for i, sample in enumerate(recs):
            units.append(ops.to_work_unit(sample, sample_index=i + 1, source_file=base))

    # Dedup by record_id (keep first). Datasets can overlap — e.g. a record split
    # across two hardest-N files — and a duplicate id would otherwise be processed
    # twice and duplicated downstream.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for u in units:
        rid = u.get("record_id")
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(u)
    if len(deduped) != len(units):
        print(f"[prepare] dropped {len(units) - len(deduped)} duplicate record_id(s)")

    # prepare always rewrites (it's the source of the run); resume=False.
    return _drive(
        "prepare", deduped, lambda r: r, output_path, concurrency=1, resume=False
    )


# ---------------------------------------------------------------------------
# Stage 1: translate
# ---------------------------------------------------------------------------
def run_translate(
    input_path: str,
    output_path: str,
    provider_type: str = registry.DEFAULT_INTERPRETER_PROVIDER,
    model_name: str = registry.DEFAULT_INTERPRETER_MODEL,
    thinking_level: str = "minimal",
    concurrency: int = 4,
    backend: str = "sync",
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    batch_client: Optional[BatchClient] = None,
    interpreter_provider: Any = None,
    interpreter_label: Optional[str] = None,
) -> str:
    records = read_jsonl(input_path)
    label_str = interpreter_label or registry.label_for(provider_type, model_name)

    # Local GPU NMT models (NLLB / SeamlessM4T): direct translation, no prompt.
    # Forced to concurrency 1 — a single GPU model isn't thread-safe.
    if provider_type in ("nllb", "seamless"):
        from .local_nmt import build_local_translator

        # only treat --model as a checkpoint id when it looks like one (has "/")
        ckpt = model_name if model_name and "/" in model_name else None
        translator = build_local_translator(provider_type, ckpt)
        label_str = translator.label()

        def fn(r):
            return ops.translate_record_local(r, translator, label_str)

        return _drive(
            "translate",
            records,
            fn,
            output_path,
            concurrency=1,
            produced=_produced_translate,
        )

    # Local GPU chat LLM (Tiny Aya Global): instruction-tuned, so it uses the
    # normal translation-brief path (translate_record), but a single GPU model
    # isn't thread-safe -> force concurrency 1.
    if provider_type == "aya":
        from .local_llm import build_local_llm

        ckpt = model_name if model_name and "/" in model_name else None
        provider = build_local_llm(provider_type, ckpt)
        label_str = provider.label()

        def fn(r):
            return ops.translate_record(r, provider, label_str)

        return _drive(
            "translate",
            records,
            fn,
            output_path,
            concurrency=1,
            produced=_produced_translate,
        )

    # Free web-endpoint translators (Google / Papago): direct translation, no
    # prompt, no API key. Papago's PentaGo backend isn't thread-safe (per-call
    # event loop + shared signed engine) so force concurrency 1 there.
    if provider_type in ("google", "papago"):
        from .web_mt import build_web_translator

        translator = build_web_translator(provider_type)
        label_str = translator.label()

        def fn(r):
            return ops.translate_record_local(r, translator, label_str)

        return _drive(
            "translate",
            records,
            fn,
            output_path,
            concurrency=1 if provider_type == "papago" else concurrency,
            produced=_produced_translate,
        )

    if backend == "batch":
        if batch_client is None:
            batch_client = build_batch_client(provider_type)
        return _drive_batch(
            "translate",
            records,
            lambda r: _translate_request(r, thinking_level),
            lambda r, text: ops.apply_translate_response(r, text, label_str),
            batch_client,
            provider_type,
            model_name,
            output_path,
            wait=batch_wait,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            produced=_produced_translate,
        )

    if interpreter_provider is None:
        interpreter_provider = registry.build_interpreter_provider(
            provider_type, model_name, thinking_level
        )
    interpreter_label = label_str

    def fn(r):
        return ops.translate_record(r, interpreter_provider, interpreter_label)

    return _drive(
        "translate",
        records,
        fn,
        output_path,
        concurrency=concurrency,
        produced=_produced_translate,
    )


# ---------------------------------------------------------------------------
# Stage 2: respond
# ---------------------------------------------------------------------------
def run_respond(
    input_path: str,
    output_path: str,
    concurrency: int = 1,
    user_sim_factory: Optional[Callable] = None,
) -> str:
    if user_sim_factory is None:
        user_sim_factory = registry.make_user_sim_factory()
    records = read_jsonl(input_path)

    def fn(r):
        return ops.respond_record(r, user_sim_factory)

    return _drive(
        "respond",
        records,
        fn,
        output_path,
        concurrency=concurrency,
        produced=_produced_respond,
    )


# ---------------------------------------------------------------------------
# Stage 3: verify
# ---------------------------------------------------------------------------
def run_verify(
    input_path: str,
    output_path: str,
    min_confidence: float = 0.8,
    glotlid_model: Any = None,
    concurrency: int = 1,
) -> str:
    if glotlid_model is None:
        from interpreter_agent_eval.utils.language_verification import (
            load_glotlid_model,
        )

        glotlid_model = load_glotlid_model()
    records = read_jsonl(input_path)

    def fn(r):
        return ops.verify_record(r, glotlid_model, min_confidence)

    # fasttext predict holds the GIL; serial is simplest and fast enough.
    return _drive(
        "verify",
        records,
        fn,
        output_path,
        concurrency=concurrency,
        produced=_produced_verify,
    )


# ---------------------------------------------------------------------------
# Stage 4: judge
# ---------------------------------------------------------------------------
def run_judge(
    input_path: str,
    output_path: str,
    provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    model_name: str = registry.DEFAULT_JUDGE_MODEL,
    thinking_level: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    concurrency: int = 4,
    backend: str = "sync",
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    batch_client: Optional[BatchClient] = None,
    judge_provider: Any = None,
    judge_label: Optional[str] = None,
) -> str:
    records = read_jsonl(input_path)
    label_str = judge_label or registry.label_for(provider_type, model_name)

    if backend == "batch":
        if batch_client is None:
            batch_client = build_batch_client(provider_type)
        return _drive_batch(
            "judge",
            records,
            lambda r: _judge_request(r, thinking_level),
            lambda r, text: ops.apply_judge_response(r, text, label_str),
            batch_client,
            provider_type,
            model_name,
            output_path,
            wait=batch_wait,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            produced=_produced_judge,
        )

    if judge_provider is None:
        judge_provider = registry.build_judge_provider(
            provider_type, model_name, thinking_level
        )
    judge_label = label_str

    def fn(r):
        return ops.judge_record(r, judge_provider, judge_label)

    return _drive(
        "judge",
        records,
        fn,
        output_path,
        concurrency=concurrency,
        produced=_produced_judge,
    )


# ---------------------------------------------------------------------------
# Stage 5: consolidate
# ---------------------------------------------------------------------------
def run_consolidate(input_path: str, output_path: str) -> str:
    records = read_jsonl(input_path)
    # Rewrite from scratch so the final file always reflects the latest judge output.
    return _drive(
        "consolidate", records, ops.consolidate_record, output_path, resume=False
    )


# ---------------------------------------------------------------------------
# Orchestrator: run every stage sequentially in a run directory
# ---------------------------------------------------------------------------
def run_all(
    data_files: List[str],
    run_dir: str,
    num_samples: Optional[int] = None,
    filter_target_lang: Optional[str] = None,
    interpreter_provider_type: str = registry.DEFAULT_INTERPRETER_PROVIDER,
    interpreter_model: str = registry.DEFAULT_INTERPRETER_MODEL,
    interpreter_thinking: str = "minimal",
    judge_provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    judge_model: str = registry.DEFAULT_JUDGE_MODEL,
    judge_thinking: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    translate_concurrency: int = 4,
    respond_concurrency: int = 1,
    judge_concurrency: int = 4,
    translate_backend: str = "sync",
    judge_backend: str = "sync",
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    min_confidence: float = 0.8,
    glotlid_model: Any = None,
    final_output: Optional[str] = None,
) -> str:
    """Run prepare -> translate -> respond -> verify -> judge -> consolidate.

    Returns the path to the final consolidated results file. Idempotent: re-run
    with the same ``run_dir`` to resume after an interruption.

    ``translate_backend`` / ``judge_backend`` may be "batch" to route those cloud
    stages through the OpenAI/Gemini Batch API. With ``batch_wait=False`` the run
    submits the batch and returns; re-run later to collect and finish.
    """
    os.makedirs(run_dir, exist_ok=True)

    def p(name):
        return os.path.join(run_dir, name)

    results_path = final_output or p(RESULTS)

    run_prepare(data_files, p(UNITS), num_samples, filter_target_lang)
    run_translate(
        p(UNITS),
        p(TRANSLATED),
        interpreter_provider_type,
        interpreter_model,
        interpreter_thinking,
        concurrency=translate_concurrency,
        backend=translate_backend,
        batch_wait=batch_wait,
        poll_interval=poll_interval,
        batch_timeout=batch_timeout,
    )
    run_respond(p(TRANSLATED), p(RESPONDED), concurrency=respond_concurrency)
    run_verify(
        p(RESPONDED),
        p(VERIFIED),
        min_confidence=min_confidence,
        glotlid_model=glotlid_model,
    )
    run_judge(
        p(VERIFIED),
        p(JUDGED),
        judge_provider_type,
        judge_model,
        judge_thinking,
        concurrency=judge_concurrency,
        backend=judge_backend,
        batch_wait=batch_wait,
        poll_interval=poll_interval,
        batch_timeout=batch_timeout,
    )
    run_consolidate(p(JUDGED), results_path)
    print(f"\n[done] Final results: {results_path}")
    return results_path
