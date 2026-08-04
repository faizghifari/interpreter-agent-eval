"""Deterministic content-safety heuristics shared by data pipelines.

The policy blocks high-confidence PII plus explicit and ordinary offensive
terms. Ambiguous terms and phone/URL candidates are review-only and remain
allowed. These heuristics reduce release risk but cannot guarantee exhaustive
detection of novel, obfuscated, or context-dependent content.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Rule:
    category: str
    name: str
    severity: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class ContentSafetyDecision:
    block_rules: tuple[str, ...]
    review_rules: tuple[str, ...]
    block_categories: tuple[str, ...]
    review_categories: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return bool(self.block_rules)

    @property
    def review_only(self) -> bool:
        return bool(self.review_rules) and not self.blocked

    @property
    def reason(self) -> str:
        if not self.blocked:
            return "pass"
        return "content_safety_block:" + ",".join(self.block_rules)


def _compile_terms(terms: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(set(terms), key=len, reverse=True)
    body = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def _compile_substrings(terms: Iterable[str]) -> re.Pattern[str]:
    """Compile terms for scripts whose grammatical particles attach directly."""
    ordered = sorted(set(terms), key=len, reverse=True)
    return re.compile("|".join(re.escape(term) for term in ordered), re.IGNORECASE)


def _compile_bengali_terms(terms: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(set(terms), key=len, reverse=True)
    body = "|".join(re.escape(term) for term in ordered)
    return re.compile(
        rf"(?<![\u0980-\u09ff])(?:{body})(?![\u0980-\u09ff])",
        re.IGNORECASE,
    )


EXPLICIT_OFFENSIVE = {
    "en": (
        "motherfucker", "motherfucking", "fuck", "fucker", "fucking", "fucked",
        "bullshit", "horseshit", "shithead", "piece of shit", "son of a bitch",
        "bitch", "bastard", "asshole", "cunt", "dickhead", "dumbass", "whore",
        "slut", "nigger", "nigga", "faggot", "retard", "retarded",
    ),
    "ar": (
        "كس أمك", "كس امك", "كسمك", "ابن الكلب", "ابن كلب", "يا كلب",
        "شرموطة", "شرموط", "قحبة", "خرا", "زب", "طيز", "متناك", "عرص",
        "يا حمار",
    ),
    "bn": (
        "হারামজাদা", "শুয়োরের বাচ্চা", "শুয়োরের বাচ্চা", "খানকি", "মাগী",
        "চোদা", "চোদ", "চুদ", "গুদ", "শালা",
    ),
    "id": (
        "kontol", "memek", "ngentot", "bangsat", "bajingan", "lonte", "sundal",
        "jancuk", "cok",
    ),
    "ko": (
        "씨발", "개새끼", "개자식", "병신", "좆", "창녀", "지랄",
        "미친놈", "미친년", "썅",
    ),
}

ORDINARY_OFFENSIVE = {
    "en": ("idiot", "moron", "stupid", "shut up", "piss off", "damn"),
    "bn": ("বেশ্যা", "বোকা", "নির্বোধ", "গাধা"),
    "id": ("goblok", "tolol", "bodoh", "sialan"),
    "ko": ("닥쳐",),
}

REVIEW_OFFENSIVE = {
    "bn": ("বাল",),
    "id": ("anjing", "babi"),
    "ko": ("새끼",),
}


RULES = (
    Rule(
        "pii",
        "email_address",
        "block",
        re.compile(
            r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])",
            re.UNICODE,
        ),
    ),
    Rule(
        "pii",
        "ipv4_address",
        "block",
        re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    ),
    Rule(
        "pii",
        "payment_card_candidate",
        "block",
        re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    ),
    Rule(
        "pii",
        "government_id_with_label",
        "block",
        re.compile(
            r"(?i)(?<!\w)(?:passport|national\s+id|identity\s+(?:card|number)|"
            r"resident(?:\s+registration)?\s+number|nik|nomor\s+induk\s+kependudukan|"
            r"주민등록번호|여권번호|জাতীয়\s+পরিচয়পত্র|رقم\s+(?:الهوية|الجواز))(?!\w)"
            r"\s*[:#-]?\s*"
            r"(?=[a-z0-9\u0660-\u0669\u06f0-\u06f9-]{6,24}(?!\w))"
            r"(?=[a-z\u0660-\u0669\u06f0-\u06f9-]*[0-9\u0660-\u0669\u06f0-\u06f9])"
            r"[a-z0-9\u0660-\u0669\u06f0-\u06f9-]{6,24}"
        ),
    ),
    Rule(
        "pii",
        "phone_candidate",
        "review",
        re.compile(r"(?<!\w)(?:\+\d{1,3}[ .()-]*)?(?:\d[ .()-]*){7,15}(?!\w)"),
    ),
    Rule(
        "pii",
        "web_url",
        "review",
        re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\]\[{}]+"),
    ),
    Rule("offensive", "explicit_offensive_en", "block", _compile_terms(EXPLICIT_OFFENSIVE["en"])),
    Rule("offensive", "explicit_offensive_ar", "block", _compile_terms(EXPLICIT_OFFENSIVE["ar"])),
    Rule("offensive", "explicit_offensive_bn", "block", _compile_bengali_terms(EXPLICIT_OFFENSIVE["bn"])),
    Rule("offensive", "explicit_offensive_id", "block", _compile_terms(EXPLICIT_OFFENSIVE["id"])),
    Rule("offensive", "explicit_offensive_ko", "block", _compile_substrings(EXPLICIT_OFFENSIVE["ko"])),
    Rule("offensive", "ordinary_offensive_en", "block", _compile_terms(ORDINARY_OFFENSIVE["en"])),
    Rule("offensive", "ordinary_offensive_bn", "block", _compile_bengali_terms(ORDINARY_OFFENSIVE["bn"])),
    Rule("offensive", "ordinary_offensive_id", "block", _compile_terms(ORDINARY_OFFENSIVE["id"])),
    Rule("offensive", "ordinary_offensive_ko", "block", _compile_substrings(ORDINARY_OFFENSIVE["ko"])),
    Rule("offensive", "review_offensive_bn", "review", _compile_bengali_terms(REVIEW_OFFENSIVE["bn"])),
    Rule("offensive", "review_offensive_id", "review", _compile_terms(REVIEW_OFFENSIVE["id"])),
    Rule("offensive", "review_offensive_ko", "review", _compile_substrings(REVIEW_OFFENSIVE["ko"])),
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\u0640\u064b-\u065f\u0670]", "", normalized)


def _valid_ipv4(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _luhn(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit() and ord(char) < 128]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _valid_phone_candidate(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15 or len(set(digits)) == 1:
        return False
    separators = sum(not char.isdigit() for char in value.strip())
    return value.strip().startswith("+") or separators >= 2


def _iter_matches_normalized(normalized: str) -> Iterator[tuple[Rule, re.Match[str]]]:
    for rule in RULES:
        for match in rule.pattern.finditer(normalized):
            value = match.group(0)
            if rule.name == "ipv4_address" and not _valid_ipv4(value):
                continue
            if rule.name == "payment_card_candidate" and not _luhn(value):
                continue
            if rule.name == "phone_candidate" and not _valid_phone_candidate(value):
                continue
            yield rule, match


def iter_matches(text: str) -> Iterator[tuple[Rule, re.Match[str]]]:
    yield from _iter_matches_normalized(normalize_text(text))


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_strings(child)


def classify_texts(texts: Iterable[str]) -> ContentSafetyDecision:
    block_rules: set[str] = set()
    review_rules: set[str] = set()
    block_categories: set[str] = set()
    review_categories: set[str] = set()
    for text in texts:
        for rule, _ in iter_matches(text or ""):
            if rule.severity == "block":
                block_rules.add(rule.name)
                block_categories.add(rule.category)
            else:
                review_rules.add(rule.name)
                review_categories.add(rule.category)
    return ContentSafetyDecision(
        block_rules=tuple(sorted(block_rules)),
        review_rules=tuple(sorted(review_rules)),
        block_categories=tuple(sorted(block_categories)),
        review_categories=tuple(sorted(review_categories)),
    )


def classify_value(value: Any) -> ContentSafetyDecision:
    """Classify every string nested in a JSON-like value."""
    return classify_texts(iter_strings(value))
