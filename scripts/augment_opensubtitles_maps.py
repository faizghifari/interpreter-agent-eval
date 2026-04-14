import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


LANGS: Dict[str, Dict[str, str]] = {
    "ind": {"name": "Indonesian", "label": "Bahasa Indonesia"},
    "kor": {"name": "Korean", "label": "Korean"},
}

HINT_PATTERNS = [
    r"be ready to",
    r"prepare to",
    r"expected response",
    r"you should respond",
    r"react by",
    r"if .* then",
    r"when .* then",
]

GUIDANCE_PATTERNS = [
    r"\bsaya\s+harus\b",
    r"\baku\s+harus\b",
    r"\bkami\s+harus\b",
    r"\bkita\s+harus\b",
    r"\bsaya\s+akan\b",
    r"\baku\s+akan\b",
    r"\bkami\s+akan\b",
    r"\bkita\s+akan\b",
    r"\bi\s+(must|will|should|need\s+to)\b",
    r"\bwe\s+(must|will|should|need\s+to)\b",
    r"해야\s*해",
    r"해야\s*한다",
    r"해야\s*돼",
    r"할\s*거",
    r"할게",
    r"하겠",
]

ENGLISH_TEMPLATE_PATTERNS = [
    r"the\s+dialogue\s+lead-up",
    r"the\s+dialogue\s+has",
    r"lead-up\s+window",
    r"conversation\s+context\s+should",
    r"context\s+should\s+be\s+interpreted",
    r"previous\s+context\s+only",
]


def _as_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = _as_string(item)
        if s:
            out.append(s)
    return out


def _normalize_generated_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    checklist = raw.get("checklist")
    if not isinstance(checklist, dict):
        checklist = {}

    return {
        "speech_act_intent": _as_string(raw.get("speech_act_intent")),
        "semantic_core": _as_string(raw.get("semantic_core")),
        "mandatory_cultural_constraints": _as_string_list(
            raw.get("mandatory_cultural_constraints")
        ),
        "context_window_summary": _as_string(raw.get("context_window_summary")),
        "conversation_context": _as_string(raw.get("conversation_context")),
        "user_a_context": _as_string(raw.get("user_a_context")),
        "user_b_context": _as_string(raw.get("user_b_context")),
        "checklist": {
            "layer_1_semantic_core": _as_string_list(
                checklist.get("layer_1_semantic_core")
            ),
            "layer_2_pragmatic_function": _as_string_list(
                checklist.get("layer_2_pragmatic_function")
            ),
            "layer_3_cultural_social_constraints": _as_string_list(
                checklist.get("layer_3_cultural_social_constraints")
            ),
        },
        "verification_prompt": _as_string(raw.get("verification_prompt")),
    }


PROMPT_TEMPLATE = """You are an expert MT evaluation data designer.

You will convert bilingual subtitle-window data into evaluation metadata for one-turn interpretation simulation.

Direction:
- source language: {source_language} ({source_language_code})
- target language: {target_language} ({target_language_code})

Source turn:
{source_text}

Reference target turn (for understanding constraints only, do not copy mechanically):
{reference_target_text}

Complexity reason tags from mining pipeline:
{reason_tags}

Context window digest (critical, must be reflected in outputs):
{context_digest}

Previous context window:
{prev_context}

Task requirements:
1) Infer speech_act_intent and semantic_core from source + context.
2) Produce mandatory_cultural_constraints as concrete translator constraints (pragmatic, register, politeness, discourse, implicit assumptions, or culturally-bound adaptation).
3) Build roleplay-ready contexts while de-identifying movie specifics.
4) Do not include actor names, film titles, or scene-specific lore in conversation_context, user_a_context, or user_b_context.
5) user_a_context must be in source language. user_b_context must be in target language.
6) Produce checklist layers where YES means success.
7) Enforce checklist priority: layer_3 count >= layer_2 count >= layer_1 count.
8) conversation_context must be grounded in previous context window only, but must not quote or enumerate transcript-style turn history.
9) Do not include the current source turn inside conversation_context, context_window_summary, user_a_context, or user_b_context.
10) Add context_window_summary as 2-4 English sentences focused only on previous context window.
11) Both user contexts must be fully written in each user's language and provide rich role/situation grounding inferred from previous context (identity, relationship, setting, emotional stance when inferable) without exposing exact past utterances.
12) user_a_context must explicitly state that the user is User A (source-side user); user_b_context must explicitly state that the user is User B (target-side user).
13) Do not include guidance phrasing like "Saya harus", "Saya akan", "I must", "I will", "해야", or "할 거".
14) Do not include target-side plans, expected replies, or strategy hints in user_b_context.
15) verification_prompt must be numbered lines (1., 2., ...).
16) Checklist must include at least one criterion for contextual coherence with surrounding turns.
17) Use reference_target_text only as a soft pragmatic reference, not strict ground truth; do not suggest future responses.

Output JSON only with this schema:
{{
  "speech_act_intent": "string",
  "semantic_core": "string",
  "mandatory_cultural_constraints": ["string"],
    "context_window_summary": "string",
  "conversation_context": "string",
  "user_a_context": "string",
  "user_b_context": "string",
    "checklist": {{
    "layer_1_semantic_core": ["string"],
    "layer_2_pragmatic_function": ["string"],
    "layer_3_cultural_social_constraints": ["string"]
    }},
  "verification_prompt": "string"
}}
"""


def _load_env_file(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_env(repo_root: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(repo_root / ".env")
    except Exception:
        _load_env_file(repo_root)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}")
    return rows


def _load_existing_keys(path: Path) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    if not path.exists():
        return keys

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            seg_file = str(obj.get("segment_file", ""))
            seg_id = str(obj.get("segment_id", ""))
            if seg_file or seg_id:
                keys.add((seg_file, seg_id))

    return keys


def _iter_context(ctx: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for r in ctx:
        sid = r.get("segment_id", "")
        src = str(r.get("source_text", "")).strip()
        tgt = str(r.get("target_text", "")).strip()
        lines.append(f"- [{sid}] src: {src} | tgt: {tgt}")
    return lines


def _ctx_source_text(ctx_row: Dict[str, Any], direction: str) -> str:
    if direction == "id_kor":
        return _as_string(ctx_row.get("source_text", ""))
    return _as_string(ctx_row.get("target_text", ""))


def _ctx_target_text(ctx_row: Dict[str, Any], direction: str) -> str:
    if direction == "id_kor":
        return _as_string(ctx_row.get("target_text", ""))
    return _as_string(ctx_row.get("source_text", ""))


def _shorten(text: str, max_chars: int = 140) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_context_digest(row: Dict[str, Any], direction: str) -> str:
    prev_ctx = row.get("prev_context", []) or []
    current_source = (
        _as_string(row.get("source_text", ""))
        if direction == "id_kor"
        else _as_string(row.get("target_text", ""))
    )

    prev_source = [_ctx_source_text(x, direction) for x in prev_ctx]

    lead_up_last = [_shorten(x) for x in prev_source[-3:] if x]

    first_prev = _shorten(prev_source[0]) if prev_source else "(none)"
    latest_prev = _shorten(prev_source[-1]) if prev_source else "(none)"
    current_source_short = _shorten(current_source) if current_source else "(none)"

    return (
        f"prev_turn_count={len(prev_ctx)}\n"
        f"first_prev_source_turn={first_prev}\n"
        f"latest_prev_source_turn={latest_prev}\n"
        f"current_source_turn={current_source_short}\n"
        f"recent_leadup_last3={lead_up_last if lead_up_last else ['(none)']}"
    )


def _fallback_context_window_summary(row: Dict[str, Any], direction: str) -> str:
    prev_ctx = row.get("prev_context", []) or []
    return (
        f"The prior context window contains {len(prev_ctx)} turns that shape the immediate interaction conditions. "
        "It provides grounding from earlier dialogue only, without predicting upcoming actions or replies."
    )


def _fallback_conversation_context(row: Dict[str, Any], direction: str) -> str:
    prev_ctx = row.get("prev_context", []) or []
    reason_tags = _extract_reason_tags(row)
    reason_text = ", ".join(reason_tags[:3]) if reason_tags else "local pragmatic continuity"
    return (
        f"The interaction is already in progress with {len(prev_ctx)} earlier turns. "
        f"The current source utterance should be interpreted with {reason_text} carried from prior dialogue state. "
        "This context is grounding only and does not imply what either user will say next."
    )


def _fallback_user_a_context(row: Dict[str, Any], direction: str) -> str:
    prev_ctx = row.get("prev_context", []) or []

    if direction == "id_kor":
        return (
            f"Percakapan sudah berlangsung {len(prev_ctx)} giliran sebelum ujaran saat ini. "
            "Anda adalah penutur sumber dalam interaksi berjalan, dengan latar relasi, emosi, dan tingkat formalitas yang dibentuk oleh konteks sebelumnya. "
            "Informasi ini dipakai untuk menjaga kesinambungan makna tanpa memproyeksikan giliran berikutnya."
        )

    return (
        f"현재 발화 이전에 {len(prev_ctx)}개의 발화가 이어진 상태입니다. "
        "당신은 원문 사용자로서 이전 맥락에서 형성된 관계, 정서, 발화 톤, 공손성 수준을 인식하고 있습니다. "
        "이 정보는 의미 연속성 파악을 위한 배경이며 다음 반응 예측은 포함하지 않습니다."
    )


def _fallback_user_b_context(row: Dict[str, Any], direction: str) -> str:
    prev_ctx = row.get("prev_context", []) or []
    if direction == "id_kor":
        return (
            f"현재 발화 이전에 {len(prev_ctx)}개의 발화가 이어져 있습니다. "
            "당신은 대상 언어 사용자로서 앞선 맥락에서 드러난 관계, 정서, 장면 분위기, 공손성 수준을 알고 있습니다. "
            "이 정보는 현재 발화 해석 배경이며 다음 발화 계획은 포함하지 않습니다."
        )

    return (
        f"Percakapan sebelum giliran ini mencakup {len(prev_ctx)} ujaran. "
        "Anda adalah pengguna bahasa target yang memahami latar relasi, emosi, suasana, dan tingkat formalitas dari konteks sebelumnya. "
        "Informasi ini menjadi landasan memahami ujaran saat ini tanpa memproyeksikan giliran berikutnya."
    )


def _split_exchanges(text: str) -> List[str]:
    compact = " ".join((text or "").split())
    if not compact:
        return []

    parts = re.split(r"(?:(?<=\s)|^)-\s*", compact)
    exchanges = [p.strip(" -") for p in parts if p and p.strip(" -")]
    return exchanges if exchanges else [compact]


def _collect_prev_exchanges(row: Dict[str, Any], direction: str, side: str) -> List[str]:
    prev_ctx = row.get("prev_context", []) or []
    exchanges: List[str] = []

    for ctx_row in prev_ctx:
        text = _ctx_target_text(ctx_row, direction) if side == "target" else _ctx_source_text(ctx_row, direction)
        exchanges.extend(_split_exchanges(text))

    return exchanges


def _label_exchanges(exchanges: List[str]) -> List[str]:
    labeled: List[str] = []
    n = len(exchanges)
    if n == 0:
        return labeled

    # Assume alternating exchanges and anchor to current-source speaker = User A,
    # so the immediate previous exchange is User B.
    for i, ex in enumerate(exchanges):
        distance_from_last = (n - 1) - i
        speaker = "User B" if distance_from_last % 2 == 0 else "User A"
        cleaned = _strip_guidance_phrases(_shorten(ex, max_chars=180))
        if cleaned:
            labeled.append(f"{speaker}: {cleaned}")

    return labeled


def _build_last5_labeled_exchanges(
    row: Dict[str, Any],
    direction: str,
    side: str,
) -> List[str]:
    exchanges = _collect_prev_exchanges(row, direction, side=side)
    labeled = _label_exchanges(exchanges)
    return labeled[-5:]


def _build_conversation_history_block(row: Dict[str, Any], direction: str) -> str:
    source_lines = _build_last5_labeled_exchanges(row, direction, side="source")
    target_lines = _build_last5_labeled_exchanges(row, direction, side="target")

    if direction == "id_kor":
        source_header = "Riwayat pertukaran sebelumnya (bahasa User A / sumber - Indonesia):"
        target_header = "이전 발화 교환 기록 (User B 언어 / 대상 - 한국어):"
    else:
        source_header = "이전 발화 교환 기록 (User A 언어 / 원문 - 한국어):"
        target_header = "Riwayat pertukaran sebelumnya (bahasa User B / target - Indonesia):"

    source_block = "\n".join(source_lines) if source_lines else "- (none)"
    target_block = "\n".join(target_lines) if target_lines else "- (none)"
    return f"{source_header}\n{source_block}\n{target_header}\n{target_block}".strip()


def _build_user_history_block(row: Dict[str, Any], direction: str, user: str) -> str:
    if user == "a":
        side = "source"
        if direction == "id_kor":
            header = "Riwayat pertukaran terakhir (konteks sebelumnya):"
        else:
            header = "이전 맥락 최근 발화 교환:"
    else:
        side = "target"
        if direction == "id_kor":
            header = "이전 맥락 최근 발화 교환:"
        else:
            header = "Riwayat pertukaran terakhir (konteks sebelumnya):"

    lines = _build_last5_labeled_exchanges(row, direction, side=side)
    if not lines:
        return f"{header}\n- (none)"
    return header + "\n" + "\n".join(lines)


def _user_role_prefix(direction: str, user: str) -> str:
    if user == "a":
        if direction == "id_kor":
            return "Peran Anda: User A (penutur sumber)."
        return "당신의 역할: User A(원문 사용자)."

    if direction == "id_kor":
        return "당신의 역할: User B(대상 언어 사용자)."
    return "Peran Anda: User B (pengguna bahasa target)."


def _contains_english_template_language(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pat, lowered) for pat in ENGLISH_TEMPLATE_PATTERNS)


def _expected_user_role_marker(direction: str, user: str) -> str:
    if user == "a":
        return "Peran Anda" if direction == "id_kor" else "당신의 역할"
    return "당신의 역할" if direction == "id_kor" else "Peran Anda"


def _extract_reason_tags(row: Dict[str, Any]) -> List[str]:
    reasons = str(row.get("reasons", "")).strip()
    if not reasons:
        return []
    tags = [r.strip() for r in reasons.split(",") if r.strip()]
    seen: Set[str] = set()
    out: List[str] = []
    for t in tags:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _resolve_direction(global_index_1_based: int) -> str:
    split_index_0_based = global_index_1_based - 1
    return "id_kor" if split_index_0_based % 2 == 0 else "kor_id"


def _direction_spec(direction: str) -> Dict[str, str]:
    if direction == "id_kor":
        return {
            "source_code": "ind",
            "target_code": "kor",
            "source_language": LANGS["ind"]["name"],
            "target_language": LANGS["kor"]["name"],
        }
    if direction == "kor_id":
        return {
            "source_code": "kor",
            "target_code": "ind",
            "source_language": LANGS["kor"]["name"],
            "target_language": LANGS["ind"]["name"],
        }
    raise ValueError(f"Unsupported direction: {direction}")


def _render_prompt(row: Dict[str, Any], direction: str) -> str:
    spec = _direction_spec(direction)

    if direction == "id_kor":
        source_text = str(row.get("source_text", "")).strip()
        reference_target_text = str(row.get("target_text", "")).strip()
    else:
        source_text = str(row.get("target_text", "")).strip()
        reference_target_text = str(row.get("source_text", "")).strip()

    prev_lines = _iter_context(row.get("prev_context", []))
    prev_block = "\n".join(prev_lines) if prev_lines else "(none)"

    reason_tags = _extract_reason_tags(row)
    reason_text = ", ".join(reason_tags) if reason_tags else "(none)"
    context_digest = _build_context_digest(row, direction)

    return PROMPT_TEMPLATE.format(
        source_language=spec["source_language"],
        source_language_code=spec["source_code"],
        target_language=spec["target_language"],
        target_language_code=spec["target_code"],
        source_text=source_text,
        reference_target_text=reference_target_text,
        reason_tags=reason_text,
        context_digest=context_digest,
        prev_context=prev_block,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model output does not contain JSON object: {text[:300]}")

    return json.loads(text[start : end + 1])


def _llm_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: float,
) -> Dict[str, Any]:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model_name, safe='')}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=request_timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP error {exc.code}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    parsed = json.loads(body)
    candidates = parsed.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {parsed}")

    parts = (
        (candidates[0].get("content") or {}).get("parts")
        if isinstance(candidates[0], dict)
        else None
    ) or []
    text = ""
    if parts and isinstance(parts[0], dict):
        text = str(parts[0].get("text") or "")
    if not text:
        raise RuntimeError(f"Gemini returned empty text: {parsed}")

    raw = _extract_json(text)
    return _normalize_generated_fields(raw)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _contains_hinting_language(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pat, lowered) for pat in HINT_PATTERNS + GUIDANCE_PATTERNS)


def _strip_hint_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    kept = [s for s in sentences if not _contains_hinting_language(s)]
    return " ".join(kept)


def _strip_guidance_phrases(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in GUIDANCE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _strip_existing_turn_history(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"Turn\s*-?\d+\s*\|[^\n]*", "", text)
    out = re.sub(r"Recent history[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"Riwayat\s*5\s*giliran[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"이전\s*맥락\s*최근\s*5개\s*발화[^\n]*", "", out)
    out = re.sub(r"Konteks\s*percakapan\s*:[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"이전\s*대화\s*:[^\n]*", "", out)
    out = re.sub(r"Previous\s*exchange[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"Previous\s*turns[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"Riwayat\s*percakapan[^\n]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"User\s*[AB]\s*:[^\n]*", "", out)
    out = re.sub(r"\n{2,}", "\n", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _contains_current_source_text(context_text: str, row: Dict[str, Any], direction: str) -> bool:
    current_source = (
        _as_string(row.get("source_text", ""))
        if direction == "id_kor"
        else _as_string(row.get("target_text", ""))
    )
    if not current_source:
        return False

    src_norm = re.sub(r"\W+", "", current_source.lower())
    ctx_norm = re.sub(r"\W+", "", (context_text or "").lower())
    if len(src_norm) < 16:
        return False
    return src_norm[:16] in ctx_norm or src_norm[-16:] in ctx_norm


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        v = item.strip()
        if not v:
            continue
        key = v.lower()
        if key in {"yes", "no", "ya", "tidak", "benar", "salah"}:
            continue
        if re.fullmatch(r"(yes|no)[.!?]?", key):
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(v)
    return result


def _compose_verification_prompt(checklist: Dict[str, List[str]]) -> str:
    ordered_items: List[str] = []
    ordered_items.extend(checklist.get("layer_3_cultural_social_constraints", []))
    ordered_items.extend(checklist.get("layer_2_pragmatic_function", []))
    ordered_items.extend(checklist.get("layer_1_semantic_core", []))
    cleaned = [it.strip() for it in ordered_items if it and it.strip()]
    return "\n".join(f"{i}. {item}" for i, item in enumerate(cleaned, 1))


def _normalize_verification_prompt(raw_prompt: str) -> str:
    if not raw_prompt:
        return raw_prompt

    normalized = " ".join(raw_prompt.split())
    parts = re.split(r"(?=\b\d+\.)", normalized)
    items = [p.strip() for p in parts if p.strip()]
    if not any(re.match(r"^\d+\.", item) for item in items):
        return raw_prompt.strip()
    return "\n".join(items)


def _trim_to_sentence_window(text: str, min_sentences: int, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    trimmed = sentences[:max_sentences]
    if len(trimmed) < min_sentences:
        return " ".join(sentences)
    return " ".join(trimmed)


def _sanitize_non_transcript_context(
    text: str,
    min_sentences: int,
    max_sentences: int,
) -> str:
    out = _as_string(text)
    out = _strip_hint_sentences(out)
    out = _strip_existing_turn_history(out)
    out = re.sub(r"\bUser\s*([AB])\s*:\s*", r"User \1 ", out)
    out = re.sub(r"\b(?:src|tgt)\s*:\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bTurn\s*-?\d+\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    out = _strip_guidance_phrases(out)
    out = _trim_to_sentence_window(out, min_sentences=min_sentences, max_sentences=max_sentences)
    return out.strip()


def _checklist_has_keyword(items: List[str], keywords: List[str]) -> bool:
    lowered = " ".join(i.lower() for i in items)
    return any(k in lowered for k in keywords)


def _repair_generated_sample(
    generated: Dict[str, Any],
    row: Dict[str, Any],
    direction: str,
) -> Dict[str, Any]:
    fallback_window = _fallback_context_window_summary(row, direction)
    fallback_conversation = _fallback_conversation_context(row, direction)
    fallback_user_a = _fallback_user_a_context(row, direction)
    fallback_user_b = _fallback_user_b_context(row, direction)
    user_a_role = _user_role_prefix(direction, user="a")
    user_b_role = _user_role_prefix(direction, user="b")

    context_window_summary = _trim_to_sentence_window(
        _strip_hint_sentences(_as_string(generated.get("context_window_summary", ""))),
        min_sentences=2,
        max_sentences=4,
    )
    context_window_summary = _strip_existing_turn_history(context_window_summary)
    if not context_window_summary or _contains_english_template_language(context_window_summary):
        context_window_summary = fallback_window
    context_window_summary = _strip_guidance_phrases(context_window_summary)
    generated["context_window_summary"] = context_window_summary

    conversation_context = _sanitize_non_transcript_context(
        _as_string(generated.get("conversation_context", "")),
        min_sentences=2,
        max_sentences=5,
    )
    if not conversation_context or _contains_english_template_language(conversation_context):
        conversation_context = fallback_conversation
    generated["conversation_context"] = conversation_context

    user_a_body = _sanitize_non_transcript_context(
        _as_string(generated.get("user_a_context", "")),
        min_sentences=2,
        max_sentences=6,
    )
    user_b_body = _sanitize_non_transcript_context(
        _as_string(generated.get("user_b_context", "")),
        min_sentences=2,
        max_sentences=6,
    )

    if not user_a_body or _contains_english_template_language(user_a_body):
        user_a_body = fallback_user_a
    if not user_b_body or _contains_english_template_language(user_b_body):
        user_b_body = fallback_user_b

    user_a_context = f"{user_a_role} {user_a_body}".strip()
    user_b_context = f"{user_b_role} {user_b_body}".strip()

    generated["user_a_context"] = user_a_context
    generated["user_b_context"] = user_b_context

    generated["mandatory_cultural_constraints"] = _dedupe_keep_order(
        _as_string_list(generated.get("mandatory_cultural_constraints"))
    )

    checklist = generated.get("checklist")
    if not isinstance(checklist, dict):
        checklist = {}

    l1 = _dedupe_keep_order(_as_string_list(checklist.get("layer_1_semantic_core")))
    l2 = _dedupe_keep_order(
        _as_string_list(checklist.get("layer_2_pragmatic_function"))
    )
    l3 = _dedupe_keep_order(
        _as_string_list(checklist.get("layer_3_cultural_social_constraints"))
    )

    if len(l1) > 2:
        l2.extend(l1[2:])
        l1 = l1[:2]

    if not l1:
        l1 = [
            "Does the translation preserve the core factual meaning of the source utterance?"
        ]
    if len(l2) < 2:
        l2.extend(
            [
                "Does the translation preserve the speaker's communicative intent?",
                "Does the translation preserve interpersonal stance and implied intent in context?",
            ]
        )
    if len(l3) < 2:
        l3.extend(
            [
                "Does the translation preserve required register, politeness, and social stance?",
                "Does the translation bridge non-literal constraints required for natural understanding?",
            ]
        )

    if not _checklist_has_keyword(
        l2,
        ["context", "preced", "follow", "coher", "turn", "dialogue flow"],
    ):
        l2.append(
            "Does the translation remain coherent with surrounding dialogue turns and local context?"
        )

    l1 = _dedupe_keep_order(l1)
    l2 = _dedupe_keep_order(l2)
    l3 = _dedupe_keep_order(l3)

    if len(l2) < len(l1):
        l2.extend(l1[: len(l1) - len(l2)])
        l2 = _dedupe_keep_order(l2)
    if len(l3) < len(l2):
        l3.extend(l2[: len(l2) - len(l3)])
        l3 = _dedupe_keep_order(l3)

    generated["checklist"] = {
        "layer_1_semantic_core": l1,
        "layer_2_pragmatic_function": l2,
        "layer_3_cultural_social_constraints": l3,
    }

    generated["verification_prompt"] = _compose_verification_prompt(
        generated["checklist"]
    )
    generated["verification_prompt"] = _normalize_verification_prompt(
        generated["verification_prompt"]
    )

    return generated


def _validate_generated_sample(
    generated: Dict[str, Any],
    row: Optional[Dict[str, Any]] = None,
    direction: Optional[str] = None,
) -> List[str]:
    errors: List[str] = []

    cc = _trim_to_sentence_window(_as_string(generated.get("conversation_context", "")), 1, 10)
    cws = _trim_to_sentence_window(_as_string(generated.get("context_window_summary", "")), 1, 10)

    if not cc:
        errors.append("conversation_context is empty")
    if _contains_hinting_language(cc):
        errors.append("conversation_context contains guidance language")
    if "Turn" in cc:
        errors.append("conversation_context should not include Turn numbering")
    if "User A:" in cc or "User B:" in cc:
        errors.append("conversation_context should not expose transcript-style speaker lines")
    if re.search(r"\b(src|tgt)\s*:", cc, flags=re.IGNORECASE):
        errors.append("conversation_context should not expose src/tgt transcript markers")
    if _contains_english_template_language(cc):
        errors.append("conversation_context contains English template prose")
    if not cws:
        errors.append("context_window_summary is empty")
    if len(_split_sentences(cws)) < 2:
        errors.append("context_window_summary should contain at least two sentences")
    if _contains_hinting_language(cws):
        errors.append("context_window_summary contains guidance language")
    if "Turn" in cws:
        errors.append("context_window_summary should not include Turn numbering")

    if _contains_hinting_language(_as_string(generated.get("user_a_context", ""))):
        errors.append("user_a_context contains hinting language")
    if _contains_hinting_language(_as_string(generated.get("user_b_context", ""))):
        errors.append("user_b_context contains hinting language")
    ua = _as_string(generated.get("user_a_context", ""))
    ub = _as_string(generated.get("user_b_context", ""))
    if "Turn" in ua:
        errors.append("user_a_context should not include Turn numbering")
    if "Turn" in ub:
        errors.append("user_b_context should not include Turn numbering")
    if "User A" not in ua:
        errors.append("user_a_context must explicitly identify the user as User A")
    if "User B" not in ub:
        errors.append("user_b_context must explicitly identify the user as User B")
    if "User A:" in ua or "User B:" in ua:
        errors.append("user_a_context should not include transcript-style speaker lines")
    if "User A:" in ub or "User B:" in ub:
        errors.append("user_b_context should not include transcript-style speaker lines")
    if re.search(r"\b(src|tgt)\s*:", ua, flags=re.IGNORECASE):
        errors.append("user_a_context should not expose src/tgt transcript markers")
    if re.search(r"\b(src|tgt)\s*:", ub, flags=re.IGNORECASE):
        errors.append("user_b_context should not expose src/tgt transcript markers")
    if _contains_english_template_language(ua):
        errors.append("user_a_context contains English template prose")
    if _contains_english_template_language(ub):
        errors.append("user_b_context contains English template prose")
    if direction is not None and _expected_user_role_marker(direction, "a") not in ua:
        errors.append("user_a_context language marker does not match expected user language")
    if direction is not None and _expected_user_role_marker(direction, "b") not in ub:
        errors.append("user_b_context language marker does not match expected user language")

    if row is not None and direction is not None:
        if _contains_current_source_text(cc, row, direction):
            errors.append("conversation_context should not include current source utterance")
        if _contains_current_source_text(cws, row, direction):
            errors.append("context_window_summary should not include current source utterance")
        if _contains_current_source_text(ua, row, direction):
            errors.append("user_a_context should not include current source utterance")
        if _contains_current_source_text(ub, row, direction):
            errors.append("user_b_context should not include current source utterance")

    checklist = generated.get("checklist")
    if not isinstance(checklist, dict):
        checklist = {}

    l1 = len(_as_string_list(checklist.get("layer_1_semantic_core")))
    l2 = len(_as_string_list(checklist.get("layer_2_pragmatic_function")))
    l3 = len(_as_string_list(checklist.get("layer_3_cultural_social_constraints")))

    if l1 < 1:
        errors.append("layer_1_semantic_core must have at least 1 item")
    if l2 < 2:
        errors.append("layer_2_pragmatic_function must have at least 2 items")
    if l3 < 2:
        errors.append("layer_3_cultural_social_constraints must have at least 2 items")
    if not (l3 >= l2 >= l1):
        errors.append("checklist count priority must satisfy layer_3 >= layer_2 >= layer_1")

    l2_items = _as_string_list(checklist.get("layer_2_pragmatic_function"))
    if not _checklist_has_keyword(
        l2_items,
        ["context", "coher", "preced", "follow", "turn"],
    ):
        errors.append("layer_2_pragmatic_function should include context-coherence criterion")

    return errors


def _build_output_record(
    row: Dict[str, Any],
    generated: Dict[str, Any],
    direction: str,
    global_index: int,
) -> Dict[str, Any]:
    spec = _direction_spec(direction)
    source_code = spec["source_code"]
    target_code = spec["target_code"]

    if direction == "id_kor":
        source_text = str(row.get("source_text", "")).strip()
        reference_target_text = str(row.get("target_text", "")).strip()
    else:
        source_text = str(row.get("target_text", "")).strip()
        reference_target_text = str(row.get("source_text", "")).strip()

    reason_tags = _extract_reason_tags(row)

    return {
        "seed_file": row.get("segment_file", ""),
        "seed_split": "opensubtitles_scene_filtered_repaired",
        "seed_row_id": global_index,
        "Category": "MAPS-Dialogue-Pragmatics",
        "Source Concept (Original Source Language)": generated["semantic_core"],
        "Verification Goal (Target Receiver)": generated["semantic_core"],
        'Linguistic/Cultural "Trap"': " | ".join(
            generated["mandatory_cultural_constraints"]
        ),
        "source_language": spec["source_language"],
        "target_language": spec["target_language"],
        "source_language_code": source_code,
        "target_language_code": target_code,
        "direction": direction,
        "segment_file": row.get("segment_file", ""),
        "segment_id": row.get("segment_id", ""),
        "source_text": source_text,
        "reference_target_text": reference_target_text,
        "speech_act_intent": generated["speech_act_intent"],
        "semantic_core": generated["semantic_core"],
        "mandatory_cultural_constraints": generated["mandatory_cultural_constraints"],
        "context_window_summary": generated["context_window_summary"],
        "conversation_context": generated["conversation_context"],
        "user_a_context": generated["user_a_context"],
        "user_b_context": generated["user_b_context"],
        "verification_prompt": generated["verification_prompt"],
        "checklist_layer_1_semantic_core": generated["checklist"]["layer_1_semantic_core"],
        "checklist_layer_2_pragmatic_function": generated["checklist"]["layer_2_pragmatic_function"],
        "checklist_layer_3_cultural_social_constraints": generated["checklist"]["layer_3_cultural_social_constraints"],
        "reasons": reason_tags,
        "source_row": {
            "worthiness_score": row.get("worthiness_score"),
            "complexity_score": row.get("complexity_score"),
            "quality_score": row.get("quality_score"),
            "alignment_risk": row.get("alignment_risk"),
            "embedding_similarity": row.get("embedding_similarity"),
            "n_prev": row.get("n_prev"),
            "n_after": row.get("n_after"),
        },
    }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_augmentation(
    input_jsonl: Path,
    output_id_kor: Path,
    output_kor_id: Path,
    model_name: str,
    max_rows: int,
    start_index: int,
    sleep_s: float,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    append: bool,
    input_index_offset: int,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or environment.")

    rows = _read_jsonl(input_jsonl)
    if start_index > 0:
        rows = rows[start_index:]
    if max_rows > 0:
        rows = rows[:max_rows]

    if not rows:
        raise RuntimeError("No rows to process after applying start/max limits.")

    output_id_kor.parent.mkdir(parents=True, exist_ok=True)
    output_kor_id.parent.mkdir(parents=True, exist_ok=True)

    existing_id_kor = _load_existing_keys(output_id_kor) if append else set()
    existing_kor_id = _load_existing_keys(output_kor_id) if append else set()

    if not append:
        output_id_kor.write_text("", encoding="utf-8")
        output_kor_id.write_text("", encoding="utf-8")

    written_id_kor = 0
    written_kor_id = 0
    skipped_existing = 0
    failed_rows = 0

    for idx, row in enumerate(rows, start=1):
        global_index = input_index_offset + idx
        direction = _resolve_direction(global_index)
        key = (str(row.get("segment_file", "")), str(row.get("segment_id", "")))

        if direction == "id_kor" and key in existing_id_kor:
            skipped_existing += 1
            continue
        if direction == "kor_id" and key in existing_kor_id:
            skipped_existing += 1
            continue

        prompt = _render_prompt(row=row, direction=direction)

        generated: Optional[Dict[str, Any]] = None
        for attempt in range(1, max_retries + 1):
            try:
                generated = _llm_generate(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    request_timeout_s=request_timeout_s,
                )
                generated = _repair_generated_sample(
                    generated=generated,
                    row=row,
                    direction=direction,
                )
                errors = _validate_generated_sample(
                    generated,
                    row=row,
                    direction=direction,
                )
                hard_failures = [
                    e
                    for e in errors
                    if "empty" in e
                    or "at least" in e
                    or "priority" in e
                    or "should include" in e
                ]
                if hard_failures:
                    raise ValueError("; ".join(hard_failures))
                break
            except Exception as exc:
                if attempt >= max_retries:
                    print(
                        f"failed row={global_index} direction={direction} after {max_retries} attempts: {exc}",
                        flush=True,
                    )
                    failed_rows += 1
                    generated = None
                    break
                wait_s = retry_backoff_s * attempt
                print(
                    f"retry row={global_index} direction={direction} attempt={attempt}/{max_retries} wait={wait_s:.1f}s error={exc}",
                    flush=True,
                )
                time.sleep(wait_s)

        if generated is None:
            continue

        record = _build_output_record(
            row=row,
            generated=generated,
            direction=direction,
            global_index=global_index,
        )

        if direction == "id_kor":
            _append_jsonl(output_id_kor, record)
            existing_id_kor.add(key)
            written_id_kor += 1
        else:
            _append_jsonl(output_kor_id, record)
            existing_kor_id.add(key)
            written_kor_id += 1

        if idx % 20 == 0 or idx == len(rows):
            print(
                f"processed={idx}/{len(rows)} id_kor={written_id_kor} kor_id={written_kor_id} skipped_existing={skipped_existing} failed={failed_rows}",
                flush=True,
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    print("done", flush=True)
    print(f"input_rows={len(rows)}", flush=True)
    print(f"written_id_kor={written_id_kor}", flush=True)
    print(f"written_kor_id={written_kor_id}", flush=True)
    print(f"skipped_existing={skipped_existing}", flush=True)
    print(f"failed_rows={failed_rows}", flush=True)
    print(f"output_id_kor={output_id_kor}", flush=True)
    print(f"output_kor_id={output_kor_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment filtered OpenSubtitles windows into MAPS-like data in two directions (ID->KOR and KOR->ID)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/opensubtitles_final_eval/final_filtered_w375_risklt1_labse060_080_windows_prev15_after2_scene_filtered_repaired.jsonl"
        ),
        help="Input repaired OpenSubtitles filtered JSONL.",
    )
    parser.add_argument(
        "--output-id-kor",
        type=Path,
        default=Path("data/enriched/id_kor_maps_from_opensubs.jsonl"),
        help="Output JSONL for Indonesian -> Korean direction.",
    )
    parser.add_argument(
        "--output-kor-id",
        type=Path,
        default=Path("data/enriched/kor_id_maps_from_opensubs.jsonl"),
        help="Output JSONL for Korean -> Indonesian direction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Process only first N rows after start-index (0 = all).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset in input rows (for chunked runs).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to outputs and skip rows already present in each direction file.",
    )
    parser.add_argument(
        "--input-index-offset",
        type=int,
        default=0,
        help="Offset for stable global row numbering and stable direction parity in chunked runs.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.2,
        help="Sleep duration between rows.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Gemini temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Gemini max output tokens.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=90.0,
        help="HTTP timeout per Gemini request in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries per row.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=3.0,
        help="Backoff multiplier in seconds. Actual wait is attempt * multiplier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_augmentation(
        input_jsonl=args.input,
        output_id_kor=args.output_id_kor,
        output_kor_id=args.output_kor_id,
        model_name=args.model,
        max_rows=args.max_rows,
        start_index=args.start_index,
        sleep_s=args.sleep_s,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
        append=args.append,
        input_index_offset=args.input_index_offset,
    )


if __name__ == "__main__":
    main()
