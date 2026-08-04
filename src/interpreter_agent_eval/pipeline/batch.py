"""Async batch-API backends for the cloud stages (translate, judge).

OpenAI ``/v1/batches`` and Gemini batch mode both run asynchronously (submit a
request set, poll, then collect) in exchange for a ~50% discount. That doesn't
fit the synchronous ``map`` contract, so batch work is modeled as three steps:

    submit(requests, model) -> job_id
    poll(job_id)            -> "running" | "completed" | "failed"
    collect(job_id, requests) -> {custom_id: response_text}

A stage persists ``job_id`` to a sidecar so it can submit now and collect later
(even across process restarts). The local LM-Studio ``respond`` stage cannot be
batched and always uses the synchronous backend.

These clients are exercised offline via :class:`FakeBatchClient`; live runs
require real API keys and incur the provider's batch turnaround (up to 24h).
"""

import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Normalized lifecycle states the stages reason about.
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
TERMINAL = {COMPLETED, FAILED}


@dataclass
class BatchRequest:
    """One provider-agnostic generation request inside a batch."""

    custom_id: str
    prompt: str
    system: Optional[str] = None
    # Common generation params; each client maps these to its own request shape.
    config: Dict[str, Any] = field(default_factory=dict)


class BatchClient(ABC):
    """Submit/poll/collect interface implemented per provider."""

    @abstractmethod
    def submit(self, requests: List[BatchRequest], model: str) -> str:
        """Upload the request set and create a batch job. Returns a job id."""

    @abstractmethod
    def poll(self, job_id: str) -> str:
        """Return a normalized state: pending/running/completed/failed."""

    def progress(self, job_id: str) -> Optional[str]:
        """Optional human-readable per-request progress (e.g. "83/120 done,
        0 failed"). None if the provider doesn't expose it — Gemini's AI
        Studio batch API only reports completion_stats on Vertex AI."""
        return None

    @abstractmethod
    def collect(
        self, job_id: str, requests: List[BatchRequest]
    ) -> Dict[str, Optional[str]]:
        """Fetch finished outputs as ``{custom_id: response_text}``.

        ``requests`` is passed so providers that match by position (Gemini inline
        responses) can recover the custom_id ordering.
        """


# ---------------------------------------------------------------------------
# OpenAI — explicit custom_id per line, file-based input/output
# ---------------------------------------------------------------------------
_OPENAI_DONE = {"completed"}
_OPENAI_FAIL = {"failed", "expired", "cancelled", "cancelling"}


class OpenAIBatchClient(BatchClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _body(self, req: BatchRequest, model: str) -> Dict[str, Any]:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        body: Dict[str, Any] = {"model": model, "messages": messages}
        cfg = req.config
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if cfg.get("max_tokens") is not None:
            body["max_tokens"] = cfg["max_tokens"]
        if cfg.get("reasoning_effort") is not None:
            body["reasoning_effort"] = cfg["reasoning_effort"]
        if cfg.get("json"):
            body["response_format"] = {"type": "json_object"}
        return body

    def submit(self, requests: List[BatchRequest], model: str) -> str:
        lines = [
            {
                "custom_id": r.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._body(r, model),
            }
            for r in requests
        ]
        payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        buf = io.BytesIO(payload.encode("utf-8"))
        buf.name = "batch_input.jsonl"
        uploaded = self.client.files.create(file=buf, purpose="batch")
        batch = self.client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=uploaded.id,
        )
        return batch.id

    def poll(self, job_id: str) -> str:
        status = self.client.batches.retrieve(job_id).status
        if status in _OPENAI_DONE:
            return COMPLETED
        if status in _OPENAI_FAIL:
            return FAILED
        return RUNNING

    def progress(self, job_id: str) -> Optional[str]:
        rc = self.client.batches.retrieve(job_id).request_counts
        if rc is None:
            return None
        return f"{rc.completed}/{rc.total} done, {rc.failed} failed"

    def collect(
        self, job_id: str, requests: List[BatchRequest]
    ) -> Dict[str, Optional[str]]:
        batch = self.client.batches.retrieve(job_id)
        if batch.status not in _OPENAI_DONE:
            raise RuntimeError(
                f"OpenAI batch {job_id} not complete (status={batch.status})"
            )
        if not batch.output_file_id:
            return {}
        text = self.client.files.content(batch.output_file_id).text
        out: Dict[str, Optional[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            try:
                body = obj["response"]["body"]
                out[cid] = body["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001 — keep going, mark this one missing
                out[cid] = None
        return out


# ---------------------------------------------------------------------------
# Gemini (google-genai) — inline requests, responses returned in order
# ---------------------------------------------------------------------------
def _gemini_state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


class GeminiBatchClient(BatchClient):
    def __init__(
        self, api_key: Optional[str] = None, http_options: Optional[dict] = None
    ):
        from google import genai

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if http_options:
            kwargs["http_options"] = http_options
        self.client = genai.Client(**kwargs)
        # Step 4b (docs/multiturn_gap_diagnosis.md §4b, D1 screening pilot):
        # batch token usage was previously discarded entirely. Accumulated
        # here (not returned from collect(), to keep that method's
        # {custom_id: text} contract unchanged for existing callers) and
        # readable via get_usage_totals() / usage_by_custom_id after collect().
        self.usage_totals: Dict[str, int] = {
            "prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0,
        }
        self.usage_by_custom_id: Dict[str, Dict[str, int]] = {}

    def _inlined(self, req: BatchRequest, model: str):
        from google.genai import types

        cfg_kwargs: Dict[str, Any] = {}
        if req.system:
            cfg_kwargs["system_instruction"] = req.system
        if req.config.get("temperature") is not None:
            cfg_kwargs["temperature"] = req.config["temperature"]
        if req.config.get("max_tokens") is not None:
            cfg_kwargs["max_output_tokens"] = req.config["max_tokens"]
        tlevel = req.config.get("thinking_level")
        if tlevel and tlevel != "none":
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=tlevel, include_thoughts=True
            )
        # Step 4a (docs/multiturn_gap_diagnosis.md §4b): structured output was
        # silently dropped on the batch path — config["json"]/["response_schema"]
        # were set by callers (e.g. checklist_gen.py's batch builders) but never
        # read here, unlike OpenAIBatchClient._body and the sync path
        # (GoogleAIProvider.generate). Honour both now.
        if req.config.get("json"):
            cfg_kwargs["response_mime_type"] = "application/json"
            schema = req.config.get("response_schema")
            if schema is not None:
                cfg_kwargs["response_schema"] = schema
        config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        return types.InlinedRequest(
            model=model,
            contents=req.prompt,
            config=config,
            metadata={"custom_id": req.custom_id},
        )

    def submit(self, requests: List[BatchRequest], model: str) -> str:
        src = [self._inlined(r, model) for r in requests]
        job = self.client.batches.create(model=model, src=src)
        return job.name

    def poll(self, job_id: str) -> str:
        state = _gemini_state_name(self.client.batches.get(name=job_id).state)
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            return COMPLETED
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            return FAILED
        return RUNNING

    def collect(
        self, job_id: str, requests: List[BatchRequest]
    ) -> Dict[str, Optional[str]]:
        job = self.client.batches.get(name=job_id)
        state = _gemini_state_name(job.state)
        if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            raise RuntimeError(f"Gemini batch {job_id} not complete (state={state})")
        responses = getattr(job.dest, "inlined_responses", None) or []
        out: Dict[str, Optional[str]] = {}
        # Inline responses come back in request order; prefer round-tripped
        # metadata.custom_id when present, else fall back to positional match.
        for idx, resp in enumerate(responses):
            cid = None
            meta = getattr(resp, "metadata", None)
            if isinstance(meta, dict):
                cid = meta.get("custom_id")
            if cid is None and idx < len(requests):
                cid = requests[idx].custom_id
            text = None
            try:
                text = resp.response.text
            except Exception:  # noqa: BLE001
                text = None
            if cid is not None:
                out[cid] = text
                # Step 4b: capture usage_metadata alongside text (mirrors
                # research/mt_compare/batch2_common.py's collect()).
                p = c = t = 0
                try:
                    u = getattr(resp.response, "usage_metadata", None)
                    if u is not None:
                        p = getattr(u, "prompt_token_count", 0) or 0
                        c = getattr(u, "candidates_token_count", 0) or 0
                        t = getattr(u, "thoughts_token_count", 0) or 0
                except Exception:  # noqa: BLE001
                    pass
                self.usage_by_custom_id[cid] = {
                    "prompt_tokens": p, "candidates_tokens": c, "thoughts_tokens": t,
                }
                self.usage_totals["prompt_tokens"] += p
                self.usage_totals["candidates_tokens"] += c
                self.usage_totals["thoughts_tokens"] += t
        return out

    def get_usage_totals(self) -> Dict[str, int]:
        """Accumulated token usage across every collect() call on this client."""
        return dict(self.usage_totals)


# ---------------------------------------------------------------------------
# Fake client — deterministic, offline. Drives the stage orchestration in tests.
# ---------------------------------------------------------------------------
class FakeBatchClient(BatchClient):
    """In-memory batch client. ``responder(BatchRequest) -> str`` produces output."""

    def __init__(self, responder, complete_after_polls: int = 0):
        self._responder = responder
        self._jobs: Dict[str, List[BatchRequest]] = {}
        self._polls: Dict[str, int] = {}
        self._complete_after = complete_after_polls
        self._counter = 0

    def submit(self, requests: List[BatchRequest], model: str) -> str:
        self._counter += 1
        job_id = f"fake-batch-{self._counter}"
        self._jobs[job_id] = list(requests)
        self._polls[job_id] = 0
        return job_id

    def poll(self, job_id: str) -> str:
        self._polls[job_id] += 1
        if self._polls[job_id] > self._complete_after:
            return COMPLETED
        return RUNNING

    def collect(
        self, job_id: str, requests: List[BatchRequest]
    ) -> Dict[str, Optional[str]]:
        return {
            r.custom_id: self._responder(r) for r in self._jobs.get(job_id, requests)
        }


def write_usage_totals(clients: List["GeminiBatchClient"], path: str, label: str = "") -> Dict[str, Any]:
    """Merge usage across sharded GeminiBatchClients (one per API key) and
    write ``usage_totals.json`` (Step 4b). Returns the merged dict."""
    merged_totals: Dict[str, int] = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0}
    merged_by_cid: Dict[str, Dict[str, int]] = {}
    for client in clients:
        for key, val in client.get_usage_totals().items():
            merged_totals[key] = merged_totals.get(key, 0) + val
        merged_by_cid.update(client.usage_by_custom_id)
    payload = {"label": label, "totals": merged_totals, "n_responses": len(merged_by_cid), "by_custom_id": merged_by_cid}
    import os as _os
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def build_batch_client(provider_type: str, **kwargs) -> BatchClient:
    """Construct a batch client for a cloud provider (gemini/openai)."""
    import os

    if provider_type == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIBatchClient(api_key=api_key, base_url=kwargs.get("base_url"))
    if provider_type == "gemini":
        return GeminiBatchClient(
            api_key=kwargs.get("api_key"), http_options=kwargs.get("http_options")
        )
    raise ValueError(
        f"Batch backend not supported for provider '{provider_type}'. "
        "Use 'gemini' or 'openai', or run this stage with --backend sync."
    )
