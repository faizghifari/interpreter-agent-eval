"""Stage runners for the multi-turn pipeline.

Mirrors ``pipeline/stages.py``'s shape (resumable drivers wiring an operation
to I/O) but with its own driver logic — the resume semantics genuinely differ
(wave loop for scripted translate, per-conversation loop for dynamic converse;
see plan D3) so nothing here reuses ``pipeline/stages.py``'s private
``_drive`` helpers, only the shared ``io.py`` / ``backends.py`` primitives.
"""

import concurrent.futures
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..backends import SyncBackend
from ..batch import FAILED, TERMINAL, BatchClient, BatchRequest, build_batch_client
from ..io import JsonlWriter, load_done_ids, read_jsonl, write_jsonl
from ...models import JudgeEvaluation
from .. import registry
from . import checklist_gen as cg
from . import operations as ops

# Canonical artifact names within a run directory (plan's Stage flow table).
UNITS = "00_units.jsonl"  # scripted only
TRANSLATED = "01_translated.jsonl"  # scripted
RESPONDED = "02_responded.jsonl"  # scripted, optional
CONVERSED = "01_conversed.jsonl"  # dynamic only
VERIFIED = "03_verified.jsonl"  # shared tail
TURN_JUDGED = "04_turn_judged.jsonl"  # shared tail
CONV_JUDGED = "05_conv_judged.jsonl"  # shared tail


def run_mt_prepare(
    data_files: List[str],
    output_path: str,
    num_conversations: Optional[int] = None,
) -> str:
    """Flatten scripted scenario file(s) into per-turn work units (00_units.jsonl).

    Always rewrites ``output_path`` (prepare is the source of truth for a run,
    same convention as the single-turn ``run_prepare``). No LLM calls.
    """
    conversations: List[Any] = []
    for path in data_files:
        conversations.extend(read_jsonl(path))

    if num_conversations is not None:
        conversations = conversations[:num_conversations]

    units: List[Any] = []
    skipped = 0
    for conversation in conversations:
        if conversation.get("mode") != "scripted":
            skipped += 1
            continue
        units.extend(ops.conversation_to_turn_units(conversation))
    if skipped:
        print(f"[mt-prepare] skipped {skipped} non-scripted record(s)")

    # Dedup by record_id (keep first) — mirrors run_prepare's defensive dedup;
    # a duplicate would otherwise be processed twice and duplicated downstream.
    seen = set()
    deduped = []
    for u in units:
        rid = u.get("record_id")
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(u)
    if len(deduped) != len(units):
        print(f"[mt-prepare] dropped {len(units) - len(deduped)} duplicate record_id(s)")

    n = write_jsonl(output_path, deduped)
    print(
        f"[mt-prepare] {len(conversations)} conversation(s) -> {n} turn unit(s) -> {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Stage: mt-translate — scripted, wave loop (plan D3)
#
# All conversations' turn-0 units run in parallel, then turn-1, etc. A unit is
# runnable in wave k only once every lower-index turn of its OWN conversation
# is already in the output; an earlier failure silently withholds that
# conversation's later turns for the rest of this run (they are neither
# written nor pending — re-running the whole stage retries the failed turn,
# and once it succeeds its successors become runnable, in this run or the
# next). Batch API (plan Step 8): one job PER WAVE — a wave can't submit until
# the previous wave's job has been collected, since it needs those translations
# for its own transcript-so-far block. Sidecar: ``{output_path}.batch.t{NN}.json``.
# ---------------------------------------------------------------------------
def _wave_batch_sidecar(output_path: str, wave_index: int) -> str:
    return f"{output_path}.batch.t{wave_index:02d}.json"


def _turn_translate_batch_request(
    unit: Dict[str, Any],
    prior_translations: Dict[Tuple[str, int], str],
    context_mode: str,
    thinking_level: str,
) -> Optional[BatchRequest]:
    system, user = ops.build_turn_translate_request(unit, prior_translations, context_mode)
    if user is None:
        return None
    return BatchRequest(
        custom_id=unit["record_id"], prompt=user, system=system, config={"thinking_level": thinking_level}
    )


def run_mt_translate(
    input_path: str,
    output_path: str,
    provider_type: str = registry.DEFAULT_INTERPRETER_PROVIDER,
    model_name: str = registry.DEFAULT_INTERPRETER_MODEL,
    thinking_level: str = "minimal",
    concurrency: int = 4,
    context_mode: str = "transcript",
    backend: str = "sync",
    interpreter_provider: Any = None,
    interpreter_label: Optional[str] = None,
    batch_client: Optional[BatchClient] = None,
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    api_key: Optional[str] = None,
) -> str:
    if backend not in ("sync", "batch"):
        raise ValueError(f"Unknown backend '{backend}'; use 'sync' or 'batch'")
    if backend == "batch" and provider_type in ("nllb", "seamless", "aya", "google", "papago"):
        raise NotImplementedError(
            "mt-translate's batch backend doesn't apply to local models/translators"
        )

    units = read_jsonl(input_path)
    label_str = interpreter_label or registry.label_for(provider_type, model_name)

    # Unified signature (unit, prior_translations) -> record regardless of
    # branch, even though the NMT path ignores prior_translations (context-free
    # by nature) — mypy requires conditional function variants to match, and a
    # single call-site shape simplifies the sync wave loop below. Only ever
    # referenced when backend == "sync" (assigned by one of the two branches
    # below before that point).
    if backend == "sync":
        if provider_type in ("nllb", "seamless"):
            from ..local_nmt import build_local_translator

            # only treat --model as a checkpoint id when it looks like one (has "/")
            ckpt = model_name if model_name and "/" in model_name else None
            translator = build_local_translator(provider_type, ckpt)
            label_str = translator.label()
            concurrency = 1  # single GPU model isn't thread-safe

            def call_fn(unit: Dict[str, Any], _prior: Dict[Tuple[str, int], str]) -> Dict[str, Any]:
                return ops.translate_turn_record_local(unit, translator, label_str, _prior)

        elif provider_type == "aya":
            if interpreter_provider is None:
                from ..local_llm import build_local_llm

                ckpt = model_name if model_name and "/" in model_name else None
                interpreter_provider = build_local_llm(provider_type, ckpt)
            label_str = interpreter_label or interpreter_provider.label()
            concurrency = 1  # one resident CUDA model; concurrent generate() is unsafe

            def call_fn(unit: Dict[str, Any], _prior: Dict[Tuple[str, int], str]) -> Dict[str, Any]:
                return ops.translate_turn_record(
                    unit, interpreter_provider, label_str, model_name, _prior, context_mode
                )

        elif provider_type in ("google", "papago"):
            from ..web_mt import build_web_translator

            translator = build_web_translator(provider_type)
            label_str = translator.label()
            concurrency = 1 if provider_type == "papago" else concurrency

            def call_fn(unit: Dict[str, Any], _prior: Dict[Tuple[str, int], str]) -> Dict[str, Any]:
                return ops.translate_turn_record_local(unit, translator, label_str, _prior)

        else:
            if interpreter_provider is None:
                interpreter_provider = registry.build_interpreter_provider(
                    provider_type, model_name, thinking_level
                )

            def call_fn(unit: Dict[str, Any], _prior: Dict[Tuple[str, int], str]) -> Dict[str, Any]:
                return ops.translate_turn_record(
                    unit, interpreter_provider, label_str, model_name, _prior, context_mode
                )
    elif batch_client is None:
        batch_client = build_batch_client(provider_type, api_key=api_key)

    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for u in units:
        by_conv.setdefault(u["conversation_id"], []).append(u)
    for conv_units in by_conv.values():
        conv_units.sort(key=lambda u: u["turn_index"])

    max_turn = max((u["turn_index"] for u in units), default=-1)

    translated: Dict[Tuple[str, int], str] = {}
    done_record_ids: set = set()
    for r in read_jsonl(output_path):
        done_record_ids.add(r.get("record_id"))
        if r.get("translated_text"):
            translated[(r["conversation_id"], r["turn_index"])] = r["translated_text"]

    pending_path = output_path + ".pending.jsonl"
    if os.path.exists(pending_path):
        os.remove(pending_path)  # rebuilt fresh each run from this run's failures

    backend_runner = SyncBackend(concurrency)
    written = 0
    pend = 0

    with JsonlWriter(output_path) as writer, JsonlWriter(pending_path) as pwriter:
        for k in range(max_turn + 1):
            wave: List[Dict[str, Any]] = []
            for conv_id, conv_units in by_conv.items():
                unit = next((u for u in conv_units if u["turn_index"] == k), None)
                if unit is None or unit["record_id"] in done_record_ids:
                    continue
                prior_indices = [u["turn_index"] for u in conv_units if u["turn_index"] < k]
                needs_prior_translations = context_mode == "transcript"
                if needs_prior_translations and not all(
                    (conv_id, ti) in translated for ti in prior_indices
                ):
                    continue  # blocked: an earlier turn of this conversation isn't done
                wave.append(unit)

            if not wave:
                continue

            prior_snapshot = dict(translated)

            def on_result(out: Dict[str, Any]) -> None:
                nonlocal written, pend
                if out.get("translated_text"):
                    writer.append(out)
                    translated[(out["conversation_id"], out["turn_index"])] = out["translated_text"]
                    done_record_ids.add(out["record_id"])
                    written += 1
                else:
                    pwriter.append(out)
                    pend += 1

            if backend == "sync":
                print(f"[mt-translate] wave {k}: {len(wave)} turn(s)")

                if provider_type == "aya" and hasattr(interpreter_provider, "generate_batch"):
                    requests = [
                        ops.build_turn_translate_request(u, prior_snapshot, context_mode)
                        for u in wave
                    ]
                    try:
                        texts = interpreter_provider.generate_batch(
                            [(user, system) for system, user in requests]
                        )
                        if len(texts) != len(wave):
                            raise RuntimeError(
                                f"Aya returned {len(texts)} outputs for {len(wave)} inputs"
                            )
                        for unit, text, request in zip(wave, texts, requests):
                            out = ops.apply_turn_translate_response(
                                unit,
                                text,
                                label_str,
                                model_name,
                                prior_snapshot,
                                context_mode,
                            )
                            # Aya occasionally returns a fluent answer in the
                            # source language (or English) for non-Latin target
                            # scripts.  _assign_translation rejects that output,
                            # but treating only an empty string as retryable makes
                            # a deterministic rerun reproduce the same failure.
                            # Retry the rejected response with an explicit target-
                            # language constraint while keeping the model and
                            # decoding configuration unchanged.
                            if not out.get("translated_text"):
                                system, user = request
                                target_code = unit.get("target_lang") or "the target language"
                                target_name = ops._lang_name(target_code)
                                retry_prompt = (
                                    user
                                    + "\nThe previous response was empty or not written in "
                                    + f"{target_name} ({target_code}). Return only the translation "
                                    + f"in {target_name}; do not explain it and do not use the source language."
                                )
                                retry_text = interpreter_provider.generate(
                                    retry_prompt,
                                    system_prompt=system,
                                )
                                out = ops.apply_turn_translate_response(
                                    unit,
                                    retry_text,
                                    label_str,
                                    model_name,
                                    prior_snapshot,
                                    context_mode,
                                )
                            if not out.get("translated_text"):
                                # Some small local checkpoints keep copying the
                                # source language when the long production brief
                                # is retained.  A final recovery request keeps the
                                # same dialogue evidence but gives the output-
                                # language constraint a short dedicated system
                                # message and an in-script generation cue.
                                target_code = unit.get("target_lang") or ""
                                target_name = ops._lang_name(target_code)
                                source_name = ops._lang_name(unit.get("source_lang") or "")
                                script_cue = {
                                    "arb": "الترجمة العربية:",
                                    "ben": "বাংলা অনুবাদ:",
                                    "kor": "한국어 번역:",
                                }.get(target_code, f"{target_name} translation:")
                                strict_system = (
                                    f"You are a {target_name} translation engine. "
                                    f"Respond only in {target_name}; never repeat {source_name} or use English."
                                )
                                strict_prompt = (
                                    user
                                    + f"\nSTRICT REQUIREMENT: Write the answer entirely in {target_name} "
                                    + f"({target_code}) script. Begin directly after this cue:\n{script_cue}"
                                )
                                retry_text = interpreter_provider.generate(
                                    strict_prompt,
                                    system_prompt=strict_system,
                                )
                                out = ops.apply_turn_translate_response(
                                    unit,
                                    retry_text,
                                    label_str,
                                    model_name,
                                    prior_snapshot,
                                    context_mode,
                                )
                            on_result(out)
                    except Exception as e:  # noqa: BLE001
                        print(f"[mt-translate] Aya batch failed in wave {k}: {e}")
                        for unit in wave:
                            out = ops.apply_turn_translate_response(
                                unit,
                                None,
                                label_str,
                                model_name,
                                prior_snapshot,
                                context_mode,
                            )
                            out["translate_error"] = str(e)
                            on_result(out)
                    continue

                def fn(u: Dict[str, Any], _prior: Dict[Tuple[str, int], str] = prior_snapshot) -> Dict[str, Any]:
                    return call_fn(u, _prior)

                backend_runner.map(fn, wave, on_result)
                continue

            # backend == "batch": one job for this wave; a wave can't proceed
            # until its job is collected (later waves need these translations).
            sidecar = _wave_batch_sidecar(output_path, k)
            job: Optional[Dict[str, Any]] = None
            if os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8") as f:
                    job = json.load(f)
                print(f"[mt-translate] wave {k}: resuming batch job {job['job_id']}")

            if job is None:
                requests = [
                    req
                    for req in (
                        _turn_translate_batch_request(u, prior_snapshot, context_mode, thinking_level)
                        for u in wave
                    )
                    if req is not None
                ]
                if not requests:
                    print(f"[mt-translate] wave {k}: nothing batchable")
                    continue
                job_id = batch_client.submit(requests, model_name)  # type: ignore[union-attr]
                job = {"job_id": job_id, "order": [r.custom_id for r in requests]}
                with open(sidecar, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                print(f"[mt-translate] wave {k}: submitted {len(requests)} req(s) as {job_id} -> {sidecar}")

            if not batch_wait:
                print(f"[mt-translate] wave {k}: submitted; not waiting. Re-run to collect and continue.")
                return output_path

            started = time.time()
            state = batch_client.poll(job["job_id"])  # type: ignore[union-attr]
            while state not in TERMINAL:
                if batch_timeout is not None and time.time() - started > batch_timeout:
                    print(f"[mt-translate] wave {k}: poll timeout; job still running. Re-run to collect.")
                    return output_path
                detail = batch_client.progress(job["job_id"])  # type: ignore[union-attr]
                detail_str = f" ({detail})" if detail else ""
                print(
                    f"[mt-translate] wave {k}: job {job['job_id']} state={state}{detail_str}; "
                    f"waiting {poll_interval:.0f}s"
                )
                time.sleep(poll_interval)
                state = batch_client.poll(job["job_id"])  # type: ignore[union-attr]

            if state == FAILED:
                print(f"[mt-translate] wave {k}: batch FAILED: {job['job_id']} (sidecar kept for inspection)")
                return output_path

            order = job.get("order", [])
            req_stubs = [BatchRequest(custom_id=cid, prompt="") for cid in order]
            results = batch_client.collect(job["job_id"], req_stubs)  # type: ignore[union-attr]

            unit_by_id = {u["record_id"]: u for u in wave}
            for cid in order:
                unit = unit_by_id.get(cid)
                if unit is None:
                    continue
                out = ops.apply_turn_translate_response(
                    unit, results.get(cid), label_str, model_name, prior_snapshot, context_mode
                )
                on_result(out)

            try:
                os.remove(sidecar)
            except OSError:
                pass

    if pend:
        print(f"[mt-translate] {pend} turn(s) not produced (failed/blocked) -> {pending_path}")
    else:
        try:
            os.remove(pending_path)
        except OSError:
            pass

    print(f"[mt-translate] {written} translated -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Flat-parallel driver — no cross-turn ordering dependency (plan D3/D5: respond
# / verify / judge-turns / judge-conv are all flat-parallel in both modes).
# Mirrors pipeline/stages.py's private ``_drive`` (not imported — that helper
# belongs to the protected single-turn module).
# ---------------------------------------------------------------------------
def _drive_flat(
    label: str,
    records: List[Dict[str, Any]],
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    output_path: str,
    concurrency: int = 1,
    resume: bool = True,
    id_field: str = "record_id",
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
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

    pending_n = [0]
    backend = SyncBackend(concurrency)
    with JsonlWriter(output_path) as writer, JsonlWriter(pending_path) as pwriter:

        def on_result(out: Dict[str, Any]) -> None:
            if produced is None or produced(out):
                writer.append(out)
            else:
                pwriter.append(out)
                pending_n[0] += 1

        backend.map(fn, todo, on_result)
    if pending_n[0]:
        print(f"[{label}] {pending_n[0]} not produced (failed/awaiting upstream) -> {pending_path}")
    else:
        os.remove(pending_path)
    return output_path


# ---------------------------------------------------------------------------
# Single-job batch driver — for stages with no cross-record dependency (judge
# stages; plan Step 8 says "single job for both judge stages"). Mirrors
# pipeline/stages.py's private ``_drive_batch``/``_collect_batch`` (not
# imported — those belong to the protected single-turn module).
# ---------------------------------------------------------------------------
def _batch_sidecar_path(output_path: str) -> str:
    return output_path + ".batch.json"


def _collect_single_job_batch(
    label: str,
    records: List[Dict[str, Any]],
    job: Dict[str, Any],
    apply_fn: Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]],
    client: BatchClient,
    output_path: str,
    id_field: str = "record_id",
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    order = job.get("order", [])
    req_stubs = [BatchRequest(custom_id=cid, prompt="") for cid in order]
    results = client.collect(job["job_id"], req_stubs)
    in_job = set(order)
    done = load_done_ids(output_path, id_field)
    pending_path = output_path + ".pending.jsonl"
    written = 0
    pend = 0
    with JsonlWriter(output_path) as w, JsonlWriter(pending_path) as pw:
        for r in records:
            rid = r.get(id_field, "")
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
        os.remove(_batch_sidecar_path(output_path))
    except OSError:
        pass
    print(
        f"[{label}] collected {written} record(s)"
        + (f", {pend} failed -> {pending_path}" if pend else "")
        + f" -> {output_path}"
    )
    return output_path


def _drive_single_job_batch(
    label: str,
    records: List[Dict[str, Any]],
    build_request_fn: Callable[[Dict[str, Any]], Optional[BatchRequest]],
    apply_fn: Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]],
    client: BatchClient,
    model: str,
    output_path: str,
    id_field: str = "record_id",
    wait: bool = True,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    produced: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    sidecar = _batch_sidecar_path(output_path)
    job: Optional[Dict[str, Any]] = None
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            job = json.load(f)
        print(f"[{label}] resuming batch job {job['job_id']}")

    if job is None:
        done = load_done_ids(output_path, id_field)
        todo = [r for r in records if r.get(id_field) not in done]
        print(f"[{label}] {len(records)} total | {len(done)} done | {len(todo)} to submit")
        requests = [req for req in (build_request_fn(r) for r in todo) if req is not None]
        if not requests:
            print(f"[{label}] nothing batchable -> {output_path}")
            return output_path
        job_id = client.submit(requests, model)
        job = {"job_id": job_id, "order": [req.custom_id for req in requests]}
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(job, f)
        print(f"[{label}] submitted {len(requests)} req(s) as {job_id} -> {sidecar}")

    if not wait:
        print(f"[{label}] submitted; not waiting. Collect later (batch-collect).")
        return output_path

    started = time.time()
    state = client.poll(job["job_id"])
    while state not in TERMINAL:
        if timeout is not None and time.time() - started > timeout:
            print(f"[{label}] poll timeout; job still running. Collect later.")
            return output_path
        detail = client.progress(job["job_id"])
        detail_str = f" ({detail})" if detail else ""
        print(f"[{label}] job {job['job_id']} state={state}{detail_str}; waiting {poll_interval:.0f}s")
        time.sleep(poll_interval)
        state = client.poll(job["job_id"])

    if state == FAILED:
        print(f"[{label}] batch FAILED: {job['job_id']} (sidecar kept for inspection)")
        return output_path

    return _collect_single_job_batch(label, records, job, apply_fn, client, output_path, id_field, produced)


# ---------------------------------------------------------------------------
# Stage: mt-respond — scripted only, optional comprehension probe (plan D2)
# ---------------------------------------------------------------------------
def _produced_respond(r: Dict[str, Any]) -> bool:
    return bool(r.get("listener_response"))


def run_mt_respond(
    input_path: str,
    output_path: str,
    concurrency: int = 1,
    user_sim_factory: Optional[Callable[[str], Tuple[Any, str, str]]] = None,
) -> str:
    if user_sim_factory is None:
        user_sim_factory = registry.make_user_sim_factory()
    units = read_jsonl(input_path)

    def fn(u: Dict[str, Any]) -> Dict[str, Any]:
        return ops.respond_turn_record(u, user_sim_factory)

    return _drive_flat(
        "mt-respond", units, fn, output_path, concurrency=concurrency, produced=_produced_respond
    )


# ---------------------------------------------------------------------------
# Stage: mt-verify — GlotLID on translated_text (+ listener_response)
# ---------------------------------------------------------------------------
def _produced_verify(r: Dict[str, Any]) -> bool:
    return bool(r.get("translated_text"))


def run_mt_verify(
    input_path: str,
    output_path: str,
    min_confidence: float = 0.8,
    glotlid_model: Any = None,
    concurrency: int = 1,
) -> str:
    if glotlid_model is None:
        from interpreter_agent_eval.utils.language_verification import load_glotlid_model

        glotlid_model = load_glotlid_model()
    units = read_jsonl(input_path)

    def fn(u: Dict[str, Any]) -> Dict[str, Any]:
        return ops.verify_turn_record(u, glotlid_model, min_confidence)

    # fasttext predict holds the GIL; serial is simplest and fast enough.
    return _drive_flat(
        "mt-verify", units, fn, output_path, concurrency=concurrency, produced=_produced_verify
    )


# ---------------------------------------------------------------------------
# Stage: mt-judge-turns — flat-parallel by default; sync wave loop only when
# --judge-history is enabled (plan D5: prior-turn verdicts feeding forward is
# experimental, default off, off in every funded tier, sync-only when on).
# ---------------------------------------------------------------------------
def _produced_turn_judge(r: Dict[str, Any]) -> bool:
    return r.get("evaluation") is not None


def _turn_judge_batch_request(record: Dict[str, Any], thinking_level: str) -> Optional[BatchRequest]:
    prompt = ops.build_turn_judge_prompt(record)  # no prior_judgments — batch is judge_history-incompatible
    if prompt is None:
        return None
    return BatchRequest(
        custom_id=record["record_id"],
        prompt=prompt,
        config={
            "json": True,
            "response_schema": JudgeEvaluation,
            "thinking_level": thinking_level,
        },
    )


def run_mt_judge_turns(
    input_path: str,
    output_path: str,
    provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    model_name: str = registry.DEFAULT_JUDGE_MODEL,
    thinking_level: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    concurrency: int = 4,
    judge_provider: Any = None,
    judge_label: Optional[str] = None,
    judge_history: bool = False,
    backend: str = "sync",
    batch_client: Optional[BatchClient] = None,
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    api_key: Optional[str] = None,
) -> str:
    if backend not in ("sync", "batch"):
        raise ValueError(f"Unknown backend '{backend}'; use 'sync' or 'batch'")
    if backend == "batch" and judge_history:
        raise NotImplementedError(
            "--judge-history is sync-only (plan D5): prior verdicts chain judge calls together, "
            "which the batch API can't express. Use backend='sync' when --judge-history is set."
        )

    records = read_jsonl(input_path)
    label_str = judge_label or registry.label_for(provider_type, model_name)

    if backend == "batch":
        if batch_client is None:
            batch_client = build_batch_client(provider_type, api_key=api_key)
        return _drive_single_job_batch(
            "mt-judge-turns",
            records,
            lambda r: _turn_judge_batch_request(r, thinking_level),
            lambda r, text: ops.apply_turn_judge_response(r, text, label_str),
            batch_client,
            model_name,
            output_path,
            wait=batch_wait,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            produced=_produced_turn_judge,
        )

    if judge_provider is None:
        judge_provider = registry.build_judge_provider(provider_type, model_name, thinking_level)

    if not judge_history:

        def fn(r: Dict[str, Any]) -> Dict[str, Any]:
            return ops.judge_turn_record(r, judge_provider, label_str)

        return _drive_flat(
            "mt-judge-turns",
            records,
            fn,
            output_path,
            concurrency=concurrency,
            produced=_produced_turn_judge,
        )

    # --judge-history: turn k's prompt needs turn k-1's JUDGE output, so this
    # becomes a wave loop exactly like mt-translate's, keyed on judge results
    # instead of translations. Forced sync (no thread pool): every call
    # depends on the previous, so concurrency would buy nothing.
    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_conv.setdefault(r["conversation_id"], []).append(r)
    for conv_recs in by_conv.values():
        conv_recs.sort(key=lambda r: r["turn_index"])

    max_turn = max((r["turn_index"] for r in records), default=-1)

    judged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    done_ids: set = set()
    for r in read_jsonl(output_path):
        done_ids.add(r.get("record_id"))
        if r.get("evaluation") is not None:
            judged[(r["conversation_id"], r["turn_index"])] = r

    pending_path = output_path + ".pending.jsonl"
    if os.path.exists(pending_path):
        os.remove(pending_path)

    written = 0
    pend = 0
    with JsonlWriter(output_path) as writer, JsonlWriter(pending_path) as pwriter:
        for k in range(max_turn + 1):
            wave: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            for conv_id, conv_recs in by_conv.items():
                rec = next((r for r in conv_recs if r["turn_index"] == k), None)
                if rec is None or rec["record_id"] in done_ids:
                    continue
                prior_indices = [r["turn_index"] for r in conv_recs if r["turn_index"] < k]
                if not all((conv_id, ti) in judged for ti in prior_indices):
                    continue  # blocked: an earlier turn of this conversation isn't judged yet
                wave.append((rec, [judged[(conv_id, ti)] for ti in prior_indices]))

            for rec, priors in wave:
                out = ops.judge_turn_record(rec, judge_provider, label_str, prior_judgments=priors)
                if out.get("evaluation") is not None:
                    writer.append(out)
                    judged[(out["conversation_id"], out["turn_index"])] = out
                    done_ids.add(out["record_id"])
                    written += 1
                else:
                    pwriter.append(out)
                    pend += 1

    if pend:
        print(f"[mt-judge-turns] {pend} turn(s) not produced (failed/blocked) -> {pending_path}")
    else:
        try:
            os.remove(pending_path)
        except OSError:
            pass
    print(f"[mt-judge-turns] {written} judged (--judge-history) -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Stage: mt-judge-conv — one call per conversation over the full transcript,
# free-flow conversations get a post-hoc checklist_gen call first (plan D4/D5)
# ---------------------------------------------------------------------------
def run_mt_judge_conv(
    conversation_data_files: List[str],
    turn_judged_path: str,
    output_path: str,
    judge_provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    judge_model: str = registry.DEFAULT_JUDGE_MODEL,
    judge_thinking: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    checklist_provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    checklist_model: str = registry.DEFAULT_JUDGE_MODEL,
    checklist_thinking: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    concurrency: int = 4,
    judge_provider: Any = None,
    judge_label: Optional[str] = None,
    checklist_provider: Any = None,
    no_function_grounding: bool = False,
    backend: str = "sync",
    batch_client: Optional[BatchClient] = None,
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    consistency_runs: int = 1,
    filter_by_annotation: bool = False,
    meaningful_threshold: float = cg.DEFAULT_MEANINGFUL_THRESHOLD,
) -> str:
    if backend not in ("sync", "batch"):
        raise ValueError(f"Unknown backend '{backend}'; use 'sync' or 'batch'")

    conversations: List[Dict[str, Any]] = []
    for path in conversation_data_files:
        conversations.extend(read_jsonl(path))
    conv_units = {c["conversation_id"]: ops.conversation_level_unit(c) for c in conversations}

    turn_records = read_jsonl(turn_judged_path)
    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for r in turn_records:
        by_conv.setdefault(r["conversation_id"], []).append(r)

    label_str = judge_label or registry.label_for(judge_provider_type, judge_model)
    if checklist_provider is None:
        checklist_provider = registry.build_judge_provider(
            checklist_provider_type, checklist_model, checklist_thinking
        )

    done = load_done_ids(output_path)
    todo_units = [
        unit
        for cid, unit in conv_units.items()
        if unit["record_id"] not in done and by_conv.get(cid)
    ]
    print(
        f"[mt-judge-conv] {len(conv_units)} conversation(s) | {len(done)} already done | "
        f"{len(todo_units)} to process (concurrency={concurrency}) -> {output_path}"
    )
    if not todo_units:
        return output_path

    # Sync prepass: fill in the free-flow post-hoc checklist first (D4) — much
    # smaller volume than the judge calls themselves, so it stays a plain
    # sequential pass even under backend="batch" (only the judge call batches).
    filled_units = [
        ops.ensure_conversation_checklist(
            unit, by_conv[unit["conversation_id"]], checklist_provider, unit.get("lang_b"),
            use_grounding=not no_function_grounding,
            consistency_runs=consistency_runs,
            filter_by_annotation=filter_by_annotation,
            meaningful_threshold=meaningful_threshold,
        )
        for unit in todo_units
    ]

    if backend == "batch":
        if batch_client is None:
            batch_client = build_batch_client(judge_provider_type, api_key=api_key)

        def build_req(unit: Dict[str, Any]) -> Optional[BatchRequest]:
            prompt = ops.build_conversation_judge_prompt(unit, by_conv[unit["conversation_id"]])
            if prompt is None:
                return None
            return BatchRequest(
                custom_id=unit["record_id"], prompt=prompt, config={"json": True, "thinking_level": judge_thinking}
            )

        return _drive_single_job_batch(
            "mt-judge-conv",
            filled_units,
            build_req,
            lambda u, text: ops.apply_conversation_judge_response(u, text, label_str),
            batch_client,
            judge_model,
            output_path,
            wait=batch_wait,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            produced=lambda r: r.get("evaluation") is not None,
        )

    if judge_provider is None:
        judge_provider = registry.build_judge_provider(judge_provider_type, judge_model, judge_thinking)

    def fn(unit: Dict[str, Any]) -> Dict[str, Any]:
        return ops.judge_conversation_record(unit, by_conv[unit["conversation_id"]], judge_provider, label_str)

    pending_path = output_path + ".pending.jsonl"
    if os.path.exists(pending_path):
        os.remove(pending_path)

    pending_n = 0
    sync_backend = SyncBackend(concurrency)
    with JsonlWriter(output_path) as writer, JsonlWriter(pending_path) as pwriter:

        def on_result(out: Dict[str, Any]) -> None:
            nonlocal pending_n
            if out.get("evaluation") is not None:
                writer.append(out)
            else:
                pwriter.append(out)
                pending_n += 1

        sync_backend.map(fn, filled_units, on_result)

    if pending_n:
        print(f"[mt-judge-conv] {pending_n} not produced -> {pending_path}")
    else:
        os.remove(pending_path)
    print(f"[mt-judge-conv] -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Stage: mt-converse — dynamic mode, both variants (plan D1/D3/D4)
#
# Conversations run in parallel (thread pool); WITHIN one conversation, turns
# are strictly sequential — user-sim utterance -> checklist -> translate ->
# append immediately (per-turn checkpointed). A mid-conversation failure
# writes a pending record and abandons that conversation for this run; a
# fresh invocation resumes each conversation from its first missing turn.
# ---------------------------------------------------------------------------
def run_mt_converse(
    seed_data_files: List[str],
    output_path: str,
    interpreter_provider_type: str = registry.DEFAULT_INTERPRETER_PROVIDER,
    interpreter_model: str = registry.DEFAULT_INTERPRETER_MODEL,
    interpreter_thinking: str = "minimal",
    checklist_provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    checklist_model: str = registry.DEFAULT_JUDGE_MODEL,
    checklist_thinking: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    context_mode: str = "transcript",
    concurrency: int = 4,
    user_sim_factory: Optional[Callable[[str], Tuple[Any, str, str]]] = None,
    interpreter_provider: Any = None,
    interpreter_label: Optional[str] = None,
    checklist_provider: Any = None,
    no_function_grounding: bool = False,
    defer_checklist: bool = False,
) -> str:
    """``defer_checklist`` (plan Step 8 optimization): skip inline checklist-gen
    entirely and leave ``checklist_items``/``verification_prompt`` empty on
    every turn — run ``run_mt_checklist_batch`` afterward to fill them in as
    ONE batch job instead of N sync calls woven into this live loop.
    """
    seeds: List[Dict[str, Any]] = []
    for path in seed_data_files:
        seeds.extend(read_jsonl(path))
    seeds = [s for s in seeds if s.get("mode") == "dynamic"]

    if user_sim_factory is None:
        user_sim_factory = registry.make_user_sim_factory()
    # Warm the (non-thread-safe-on-first-use) provider cache single-threaded
    # before conversations run in parallel, so no two threads race to populate
    # the same cache entry the first time a language is needed.
    for lang in {s["lang_a"] for s in seeds} | {s["lang_b"] for s in seeds}:
        try:
            user_sim_factory(lang)
        except Exception:  # noqa: BLE001 — surfaced per-turn if genuinely unconfigured
            pass

    label_str = interpreter_label or registry.label_for(interpreter_provider_type, interpreter_model)
    if interpreter_provider is None:
        interpreter_provider = registry.build_interpreter_provider(
            interpreter_provider_type, interpreter_model, interpreter_thinking
        )
    if checklist_provider is None and not defer_checklist:
        checklist_provider = registry.build_judge_provider(
            checklist_provider_type, checklist_model, checklist_thinking
        )

    existing = read_jsonl(output_path)
    by_conv_existing: Dict[str, List[Dict[str, Any]]] = {}
    for r in existing:
        by_conv_existing.setdefault(r["conversation_id"], []).append(r)

    writer = JsonlWriter(output_path)
    pending_path = output_path + ".pending.jsonl"
    if os.path.exists(pending_path):
        os.remove(pending_path)  # rebuilt fresh each run from this run's abandoned conversations
    pending_writer = JsonlWriter(pending_path)
    pending_count = [0]
    pending_lock = threading.Lock()

    def process_conversation(seed: Dict[str, Any]) -> None:
        conv_id = seed["conversation_id"]
        num_turns = seed["num_turns"]
        prior_turns = sorted(by_conv_existing.get(conv_id, []), key=lambda r: r["turn_index"])
        if len(prior_turns) >= num_turns:
            return  # already fully converged in a prior run

        history_entries: List[Dict[str, Any]] = [
            {
                "turn_index": r["turn_index"],
                "speaker": r["speaker"],
                "source_text": r.get("source_text"),
                "translated_text": r.get("translated_text"),
            }
            for r in prior_turns
        ]
        prior_translations: Dict[Tuple[str, int], str] = {
            (conv_id, r["turn_index"]): r["translated_text"] for r in prior_turns if r.get("translated_text")
        }
        start_turn = len(prior_turns)  # per-turn checkpointed; resume from first missing turn

        for turn_index in range(start_turn, num_turns):
            turn_record = ops.converse_next_turn(
                seed,
                turn_index,
                history_entries,
                user_sim_factory,
                checklist_provider,
                interpreter_provider,
                label_str,
                interpreter_model,
                prior_translations,
                context_mode,
                use_grounding=not no_function_grounding,
                skip_checklist=defer_checklist,
            )
            if not turn_record.get("translated_text"):
                pending_writer.append(turn_record)
                with pending_lock:
                    pending_count[0] += 1
                return  # abandon this conversation for this run (plan D3)

            writer.append(turn_record)
            history_entries.append(
                {
                    "turn_index": turn_index,
                    "speaker": turn_record["speaker"],
                    "source_text": turn_record.get("source_text"),
                    "translated_text": turn_record.get("translated_text"),
                }
            )
            prior_translations[(conv_id, turn_index)] = turn_record["translated_text"]

    print(f"[mt-converse] {len(seeds)} conversation(s) (concurrency={concurrency}) -> {output_path}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(process_conversation, seed) for seed in seeds]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    writer.close()
    pending_writer.close()
    if pending_count[0]:
        print(f"[mt-converse] {pending_count[0]} conversation(s) abandoned this run -> {pending_path}")
    else:
        try:
            os.remove(pending_path)
        except OSError:
            pass
    print(f"[mt-converse] -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Stage: mt-checklist-batch — post-hoc per-turn checklist gen for turns
# produced with mt-converse's defer_checklist=True (plan Step 8 optimization).
# Flat-parallel like judge stages: every turn's checklist is independent, so
# it's exactly as batchable as scripted's pre-generated checklists — the only
# difference from D4's original design is *when* it runs, not what it produces.
# ---------------------------------------------------------------------------
def _checklist_batch_request_for_turn(
    record: Dict[str, Any], use_grounding: bool, thinking_level: str
) -> Optional[BatchRequest]:
    target_lang = record.get("target_lang")
    source_text = record.get("source_text")
    if not target_lang or not source_text:
        return None
    taxonomy = None
    if use_grounding and cg.taxonomy_available(target_lang):
        taxonomy = cg.load_function_taxonomy(target_lang)
    history_text = ops.render_bilingual_history(record.get("history"))
    cultural_context = cg.get_cultural_context(record.get("lang_a") or "", record.get("lang_b") or "")
    return cg.build_turn_checklist_batch_request(
        record["record_id"],
        target_lang,
        record.get("conversation_context") or "",
        record.get("speaker") or "",
        source_text,
        history_text=history_text,
        taxonomy=taxonomy,
        thinking_level=thinking_level,
        cultural_context=cultural_context,
    )


def _apply_checklist_batch_response(
    record: Dict[str, Any],
    text: Optional[str],
    filter_by_annotation: bool = False,
    meaningful_threshold: float = cg.DEFAULT_MEANINGFUL_THRESHOLD,
) -> Dict[str, Any]:
    out = dict(record)
    items = cg.parse_checklist_batch_response(
        text,
        cg.TURN_HARD_CEILING,
        filter_by_annotation=filter_by_annotation,
        target_lang=record.get("target_lang"),
        meaningful_threshold=meaningful_threshold,
    )
    errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
    if errs:
        out["checklist_items"] = []
        out["verification_prompt"] = None
        out["checklist_error"] = "; ".join(errs)
        return out
    out["checklist_items"] = [item.model_dump() for item in items]
    out["verification_prompt"] = cg.compose_verification_prompt(items)
    note = cg.checklist_count_note(items, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
    if note:
        out["checklist_count_note"] = note
    return out


def _produced_checklist_batch(r: Dict[str, Any]) -> bool:
    return bool(r.get("checklist_items"))


def run_mt_checklist_batch(
    input_path: str,
    output_path: str,
    checklist_provider_type: str = registry.DEFAULT_JUDGE_PROVIDER,
    checklist_model: str = registry.DEFAULT_JUDGE_MODEL,
    checklist_thinking: str = registry.DEFAULT_JUDGE_THINKING_LEVEL,
    concurrency: int = 4,
    no_function_grounding: bool = False,
    backend: str = "batch",
    checklist_provider: Any = None,
    batch_client: Optional[BatchClient] = None,
    batch_wait: bool = True,
    poll_interval: float = 30.0,
    batch_timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    consistency_runs: int = 1,
    filter_by_annotation: bool = False,
    meaningful_threshold: float = cg.DEFAULT_MEANINGFUL_THRESHOLD,
) -> str:
    if backend not in ("sync", "batch"):
        raise ValueError(f"Unknown backend '{backend}'; use 'sync' or 'batch'")

    records = read_jsonl(input_path)
    use_grounding = not no_function_grounding

    if backend == "batch":
        if batch_client is None:
            batch_client = build_batch_client(checklist_provider_type, api_key=api_key)
        return _drive_single_job_batch(
            "mt-checklist-batch",
            records,
            lambda r: _checklist_batch_request_for_turn(r, use_grounding, checklist_thinking),
            lambda r, text: _apply_checklist_batch_response(
                r, text, filter_by_annotation=filter_by_annotation, meaningful_threshold=meaningful_threshold
            ),
            batch_client,
            checklist_model,
            output_path,
            wait=batch_wait,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            produced=_produced_checklist_batch,
        )

    if checklist_provider is None:
        checklist_provider = registry.build_judge_provider(
            checklist_provider_type, checklist_model, checklist_thinking
        )

    def fn(r: Dict[str, Any]) -> Dict[str, Any]:
        target_lang = r.get("target_lang")
        taxonomy = (
            cg.load_function_taxonomy(target_lang)
            if use_grounding and target_lang and cg.taxonomy_available(target_lang)
            else None
        )
        history_text = ops.render_bilingual_history(r.get("history"))
        cultural_context = cg.get_cultural_context(r.get("lang_a") or "", r.get("lang_b") or "")
        items = cg.generate_turn_checklist(
            checklist_provider,
            target_lang or "",
            r.get("conversation_context") or "",
            r.get("speaker") or "",
            r.get("source_text") or "",
            history_text=history_text,
            taxonomy=taxonomy,
            cultural_context=cultural_context,
            consistency_runs=consistency_runs,
            filter_by_annotation=filter_by_annotation,
            meaningful_threshold=meaningful_threshold,
        )
        out = dict(r)
        errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
        if errs:
            out["checklist_items"] = []
            out["verification_prompt"] = None
            out["checklist_error"] = "; ".join(errs)
            return out
        out["checklist_items"] = [item.model_dump() for item in items]
        out["verification_prompt"] = cg.compose_verification_prompt(items)
        note = cg.checklist_count_note(items, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
        if note:
            out["checklist_count_note"] = note
        return out

    return _drive_flat(
        "mt-checklist-batch", records, fn, output_path, concurrency=concurrency, produced=_produced_checklist_batch
    )


# ---------------------------------------------------------------------------
# Stage: mt-consolidate — one line per conversation + flat results_turns.jsonl
# ---------------------------------------------------------------------------
RESULTS = "results.jsonl"
RESULTS_TURNS = "results_turns.jsonl"


def run_mt_consolidate(
    conv_judged_path: str,
    turn_judged_path: str,
    output_path: str,
    turns_output_path: str,
) -> str:
    """Always rewrites both outputs from scratch (mirrors the single-turn
    consolidate convention: the final files must always reflect the latest
    judge output, not accumulate stale lines across re-runs).
    """
    conv_records = read_jsonl(conv_judged_path)
    turn_records = read_jsonl(turn_judged_path)
    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for r in turn_records:
        by_conv.setdefault(r["conversation_id"], []).append(r)

    conversation_lines = []
    turn_lines: List[Dict[str, Any]] = []
    skipped = 0
    for conv in conv_records:
        turns = by_conv.get(conv["conversation_id"])
        if not turns:
            skipped += 1
            continue
        conv_line, flat_turns = ops.consolidate_conversation(turns, conv)
        conversation_lines.append(conv_line)
        turn_lines.extend(flat_turns)
    if skipped:
        print(f"[mt-consolidate] skipped {skipped} conversation(s) with no judged turns")

    write_jsonl(output_path, conversation_lines)
    write_jsonl(turns_output_path, turn_lines)
    print(
        f"[mt-consolidate] {len(conversation_lines)} conversation(s), {len(turn_lines)} turn(s) "
        f"-> {output_path}, {turns_output_path}"
    )
    return output_path
