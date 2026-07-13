"""
Per-language feature definitions for OpenSubtitles scoring.

To add a new language:
  1. Add a script ratio function (if not already present).
  2. Define marker sets as module-level constants.
  3. Define any regex-based feature helpers as named functions.
  4. Create a LanguageFeatures instance and add it to LANG_REGISTRY.
  5. If any cross-lingual features involve this language as target, add entries
     to CROSS_FEATURES.

Framework types (FeatureDef, CrossFeatureDef, LanguageFeatures) are defined here
so that this module has no local imports and can be safely imported by
_opensubs_scoring without creating a circular dependency.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

# ── Framework types ───────────────────────────────────────────────────────────


@dataclass
class FeatureDef:
    """A complexity feature evaluated on one side of a segment pair."""
    name: str
    weight: float
    fn: Callable  # fn(text: str, token_set: Set[str]) -> bool


@dataclass
class CrossFeatureDef:
    """A complexity feature that requires both source and target text."""
    name: str
    weight: float
    fn: Callable  # fn(src_text, tgt_text, src_tokens, tgt_tokens, metrics) -> bool


@dataclass
class LanguageFeatures:
    """Per-language configuration: script detection and intrinsic complexity features.

    `features` are language-specific complexity indicators applied to a text in
    this language, regardless of whether that text is the source or target side
    of a translation pair.
    """
    code: str
    script_ratio_fn: Callable[[str], float]
    min_script_ratio: float    # hard filter: reject below this
    ideal_script_ratio: float  # quality penalty: penalise if script ratio is below this
    pronoun_tokens: Set[str]
    features: List[FeatureDef] = field(default_factory=list)


# ── Script ratio functions ────────────────────────────────────────────────────


def latin_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(
        1 for ch in alpha
        if (0x0041 <= ord(ch) <= 0x007A) or (0x00C0 <= ord(ch) <= 0x024F)
    ) / len(alpha)


def korean_script_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(
        1 for ch in alpha
        if (0xAC00 <= ord(ch) <= 0xD7AF)
        or (0x1100 <= ord(ch) <= 0x11FF)
        or (0x3130 <= ord(ch) <= 0x318F)
        or (0x4E00 <= ord(ch) <= 0x9FFF)
    ) / len(alpha)


def arabic_script_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(
        1 for ch in alpha
        if (0x0600 <= ord(ch) <= 0x06FF)
        or (0x0750 <= ord(ch) <= 0x077F)
        or (0x08A0 <= ord(ch) <= 0x08FF)
        or (0xFB50 <= ord(ch) <= 0xFDFF)
        or (0xFE70 <= ord(ch) <= 0xFEFF)
    ) / len(alpha)


def bengali_script_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for ch in alpha if 0x0980 <= ord(ch) <= 0x09FF) / len(alpha)


def japanese_script_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(
        1 for ch in alpha
        if (0x3040 <= ord(ch) <= 0x309F)   # Hiragana
        or (0x30A0 <= ord(ch) <= 0x30FF)   # Katakana
        or (0x4E00 <= ord(ch) <= 0x9FFF)   # CJK unified ideographs
        or (0x3400 <= ord(ch) <= 0x4DBF)   # CJK extension A
    ) / len(alpha)


def chinese_script_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(
        1 for ch in alpha
        if (0x4E00 <= ord(ch) <= 0x9FFF)
        or (0x3400 <= ord(ch) <= 0x4DBF)
        or (0x20000 <= ord(ch) <= 0x2A6DF)
        or (0xF900 <= ord(ch) <= 0xFAFF)
    ) / len(alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# INDONESIAN (id)
# ═══════════════════════════════════════════════════════════════════════════════

# Pragmatic particles: split by translation difficulty.
# Strong: genuinely untranslatable, carry speaker attitude / social stance.
# loh/lho: mirative / surprising-info marker — missing from earlier set, added per Wouk (2022).
ID_STRONG_DISCOURSE_MARKERS = {"dong", "deh", "sih", "kok", "loh", "lho"}
# Soft: common softeners / confirmers; still pragmatic but easier to approximate.
ID_SOFT_DISCOURSE_MARKERS = {"nih", "lah", "kan", "ya"}
# Union kept for backward-compat references elsewhere in this file.
ID_DISCOURSE_MARKERS = ID_STRONG_DISCOURSE_MARKERS | ID_SOFT_DISCOURSE_MARKERS
ID_REQUEST_MARKERS = {"tolong", "mohon", "harap", "bisa", "bisakah", "jangan", "ayo", "mari"}
ID_NEGATION_MARKERS = {"tidak", "tak", "jangan", "ga", "nggak", "enggak"}
ID_SLANG_MARKERS = {
    "banget", "gitu", "begitu", "nih", "dong", "deh", "sih", "kok", "aja", "udah",
    "nggak", "ga", "kayak", "emang", "beneran",
}
ID_PRONOUN_MARKERS = {
    "aku", "saya", "gue", "gua", "kami", "kita", "kamu", "kau", "anda", "dia", "mereka",
}


def _id_slang_only(text: str, tokens: Set[str]) -> bool:
    return bool(tokens & (ID_SLANG_MARKERS - ID_DISCOURSE_MARKERS))


def _id_reduplication(text: str, tokens: Set[str]) -> bool:
    # Indonesian reduplication (jalan-jalan) often has non-literal meaning.
    lowered = (text or "").lower()
    return bool(re.search(r"\b([a-z]{2,})-\1\b", lowered)
                or re.search(r"\b[a-z]{2,}-[a-z]{2,}\b", lowered))


def _id_affix_complexity(text: str, tokens: Set[str]) -> bool:
    # Derivational affixes (di-, ter-, ke-…-an) change meaning non-transparently.
    lowered = (text or "").lower()
    return bool(
        re.search(r"\bdi[a-z]{3,}\b", lowered)
        or re.search(r"\bter[a-z]{3,}\b", lowered)
        or re.search(r"\bke[a-z]{2,}an\b", lowered)
    )


def _id_men_prefix(text: str, tokens: Set[str]) -> bool:
    # meN- verbal prefix changes verb voice (active/passive/applicative) non-transparently.
    # Nasal assimilation: mem(b/p/f/v), men(d/t/j/l/r), meng(vowel/k/g/h), meny(s), menge(short roots).
    # Most frequent Indonesian derivational prefix (>68% of meN- usage derives verbs, Denistia & Baayen 2021).
    lowered = (text or "").lower()
    return bool(re.search(
        r"\bme(?:mb|mp|nd|nt|ng|ny|nge|nj|nl|nr)[a-z]+\b",
        lowered,
    ))


# ter- prefix: 3-way semantic ambiguity requiring contextual resolution.
# MT systems trained on di-dominated corpora (328 di- vs. 19 ter- tokens in one study)
# systematically mis-render ter-, losing accidental/potential/superlative distinctions
# (Sneddon 2010 Routledge; Udayana 2019 JLTR).
_ID_TER_PREFIX_RE = re.compile(r"\bter[a-z]{2,}\b")

# 2nd-person register sets: Indonesian has no single "you".
# Anda/Bapak/Ibu (formal) vs kamu/kau (informal) vs lo/lu (Jakartan colloquial).
# Informal → formal style transfer degrades up to 20 BLEU (Cahyawijaya et al. 2021).
ID_FORMAL_2P   = {"anda", "saudara", "saudari", "bapak", "ibu", "pak", "bu"}
ID_INFORMAL_2P = {"kamu", "kau", "lo", "lu"}


def _id_ter_prefix_ambiguity(text: str, tokens: Set[str]) -> bool:
    # ter- encodes accidental passive (terbawa = accidentally carried),
    # potential passive (terbeli = can be bought), or superlative (terbesar = biggest).
    # Context required to disambiguate — MT cannot resolve without discourse access.
    return bool(_ID_TER_PREFIX_RE.search((text or "").lower()))


def _id_register_pronoun_mismatch(text: str, tokens: Set[str]) -> bool:
    # Formal 2nd person (Anda/Bapak) co-occurring with informal (kamu/lo) or strong
    # particles (dong/sih) signals register clash requiring a translation target-register
    # decision (Cahyawijaya et al. 2021; Aji et al. ACL 2022).
    has_formal         = bool(tokens & ID_FORMAL_2P)
    has_informal       = bool(tokens & ID_INFORMAL_2P)
    has_particle_clash = has_formal and bool(tokens & ID_STRONG_DISCOURSE_MARKERS)
    return (has_formal and has_informal) or has_particle_clash


_ID_FEATURES = LanguageFeatures(
    code="id",
    script_ratio_fn=latin_ratio,
    min_script_ratio=0.55,
    ideal_script_ratio=0.65,
    pronoun_tokens=ID_PRONOUN_MARKERS,
    features=[
        FeatureDef("id_strong_discourse_marker", 1.0, lambda _, toks: bool(toks & ID_STRONG_DISCOURSE_MARKERS)),
        FeatureDef("id_soft_discourse_marker",   0.3, lambda _, toks: bool(toks & ID_SOFT_DISCOURSE_MARKERS)),
        FeatureDef("request_or_imperative", 1.0, lambda _, toks: bool(toks & ID_REQUEST_MARKERS)),
        FeatureDef("negation", 0.5, lambda _, toks: bool(toks & ID_NEGATION_MARKERS)),
        FeatureDef("id_slang_or_colloquial", 0.5, _id_slang_only),
        FeatureDef("id_reduplication", 0.75, _id_reduplication),   # raised: non-compositionality consistently documented
        FeatureDef("id_affix_complexity", 0.5, _id_affix_complexity),
        FeatureDef("id_men_prefix_complexity",      0.50, _id_men_prefix),              # voice-changing meN- prefix (Denistia & Baayen 2021)
        FeatureDef("id_ter_prefix_ambiguity",       0.75, _id_ter_prefix_ambiguity),    # new: 3-way ter- ambiguity (Sneddon 2010; Udayana 2019)
        FeatureDef("id_register_pronoun_mismatch",  0.50, _id_register_pronoun_mismatch),  # new: Anda vs kamu/lo register clash (Cahyawijaya 2021)
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# KOREAN (ko)
# ═══════════════════════════════════════════════════════════════════════════════

KO_COLLOQUIAL_MARKERS = {
    "진짜", "정말", "대박", "헐", "완전", "제발", "설마", "어쩌라고", "그러니까", "아무튼",
}
KO_PRONOUN_MARKERS = {"나", "저", "우리", "너", "당신", "그", "그녀", "얘", "걔", "쟤"}
# 은/는 = topic, 이/가 = subject — omission of both is a significant translation cue.
_KO_SUBJECT_TOPIC_PARTICLES = {"은", "는", "이", "가"}
KO_SENTENCE_ENDINGS = {
    "잖아", "거든", "거지", "구나", "군요", "더라", "더라고", "지요", "죠", "네요", "나요",
}
KO_COMPLEX_VERB_ENDINGS = {
    "겠습니다", "겠어", "겠네", "겠군", "겠지", "더라고요", "더라고", "던데", "다니까", "잖아요", "거든요",
}
KO_HONORIFIC_MARKERS = {
    "요", "습니다", "님", "씨", "선생", "형", "누나", "언니", "오빠", "아저씨", "아줌마",
}
# Lexical honorifics: parallel vocabulary items that replace base-register words.
# E.g. 먹다 → 드시다, 자다 → 주무시다, 이름 → 성함, 나이 → 연세, 밥 → 진지.
# These are structurally invisible to sentence-ending particle checks and represent
# a separate vocabulary-level difficulty dimension (Song 2010; MDPI Electronics 2021).
KO_LEXICAL_HONORIFICS = {
    "드시", "드세", "드셨", "드십",          # eat/drink (honorific stem)
    "주무시", "주무세", "주무셨",             # sleep (honorific)
    "말씀",                                  # words/speech (honorific noun)
    "여쭤", "여쭙", "여쭈",                  # ask (honorific)
    "뵙", "뵈었", "뵈어", "뵙니다",          # see/meet (honorific)
    "댁",                                    # home/house (honorific)
    "성함",                                  # name (honorific)
    "연세",                                  # age (honorific)
    "진지",                                  # meal (honorific)
    "생신",                                  # birthday (honorific)
}


def _ko_honorific(text: str, tokens: Set[str]) -> bool:
    return any(marker in text.lower() for marker in KO_HONORIFIC_MARKERS)


def _ko_lexical_honorific(text: str, tokens: Set[str]) -> bool:
    # Parallel vocabulary honorifics (드시다, 주무시다, 말씀, etc.) replace base-register words
    # at the lexical level — invisible to SFE-based checks.  Presence signals that the
    # translation must restructure entire lexical items, not just pragmatic particles.
    return any(marker in text for marker in KO_LEXICAL_HONORIFICS)


def _ko_sentence_ending(text: str, tokens: Set[str]) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return any(
        eojeol.endswith(ending)
        for eojeol in re.findall(r"[가-힣]+", compact)
        for ending in KO_SENTENCE_ENDINGS
    )


def _ko_complex_verb(text: str, tokens: Set[str]) -> bool:
    return any(
        eojeol.endswith(ending)
        for eojeol in re.findall(r"[가-힣]+", text or "")
        for ending in KO_COMPLEX_VERB_ENDINGS
    )


def _ko_topic_subject_contrast(text: str, tokens: Set[str]) -> bool:
    # Fires when both topic marking (은/는) and subject marking (이/가) co-occur.
    # The 은/는 vs 이/가 distinction encodes information structure (contrastive topic
    # vs new information) that collapses in English SVO word-order translation.
    ko_words = re.findall(r"[가-힣]{2,}", text or "")
    has_topic = any(w.endswith("은") or w.endswith("는") for w in ko_words)
    has_subject = any(w.endswith("이") or w.endswith("가") for w in ko_words)
    return has_topic and has_subject


# -더- retrospective evidential: speaker witnessed event at prior time.
# English has no grammatical correlate; any rendering loses epistemic commitment
# (Chung 2010, Lingua; Song 2010, PACLIC; Lee et al. 2024, LREC-COLING).
_KO_RETROSPECTIVE_RE = re.compile(
    r"더라(?:고요?)?|더니|더군요?|던데|던가요?|었더니|았더니|였더니"
)
# Quotative-reportative contracted forms: -다고 하다 → 대, -라고 하다 → 래.
# Attribution frame (speaker, verb-of-saying, clause type) is compressed into one morpheme;
# illocutionary force of the original must be inferred (Ceong 2016, CLA).
_KO_QUOTATIVE_RE = re.compile(
    r"[다자나라]고\s*(?:해|했|하네|하더|할게|한다|해요|했어|했는데)"
    r"|[다자나라]는\s*거야|대요?|래요?|댔|랬"
)
# Completive auxiliaries V-아/어 + {버리다|놓다|두다}: same surface structure,
# three distinct stance readings (regret/relief; retention; deliberate storage).
# Park (2003) / Kwon (2012) classify these as aspectual + attitudinal — MT flattens both.
_KO_COMPLETIVE_AUX_RE = re.compile(
    r"[아어]버리|[아어]버렸|버려[서요]?"
    r"|[아어]놓[다고면서았]|놓았"
    r"|[아어]두[다고면서었]|[아어]뒀"
)
# Korean numeral classifiers: two numeral systems (native/Sino-Korean) ×
# obligatory semantic-class suffix.  Mismatch causes systematic semantic class
# errors in MT output (Kim 2009, Language Resources and Evaluation).
_KO_NUMERAL_CLASSIFIER_RE = re.compile(
    r"\d+\s*(?:개|명|마리|권|벌|장|채|대|잔|병|켤레|송이|그루|조각|줄|번|회|살|달)"
    r"|[하두세네다여]\s*(?:개|명|마리|권|벌|장|채|대|잔|병)"
)


def _ko_retrospective_evidential(text: str, tokens: Set[str]) -> bool:
    return bool(_KO_RETROSPECTIVE_RE.search(text or ""))


def _ko_quotative_reportative(text: str, tokens: Set[str]) -> bool:
    return bool(_KO_QUOTATIVE_RE.search(text or ""))


def _ko_completive_auxiliary(text: str, tokens: Set[str]) -> bool:
    return bool(_KO_COMPLETIVE_AUX_RE.search(text or ""))


def _ko_numeral_classifier(text: str, tokens: Set[str]) -> bool:
    return bool(_KO_NUMERAL_CLASSIFIER_RE.search(text or ""))


_KO_FEATURES = LanguageFeatures(
    code="ko",
    script_ratio_fn=korean_script_ratio,
    min_script_ratio=0.35,
    ideal_script_ratio=0.55,
    pronoun_tokens=KO_PRONOUN_MARKERS,
    features=[
        FeatureDef("ko_honorific_or_social_marker", 1.0, _ko_honorific),       # raised: literature's single most translation-critical feature (MDPI 2021)
        FeatureDef("ko_lexical_honorific", 0.75, _ko_lexical_honorific),        # new: parallel honorific vocabulary (드시다, 주무시다, 말씀 …)
        FeatureDef("ko_colloquial_expression", 1.0, lambda _, toks: bool(toks & KO_COLLOQUIAL_MARKERS)),
        FeatureDef("ko_sentence_ending_particle", 0.75, _ko_sentence_ending),  # raised: evidential/mirative SFEs are near-untranslatable (Song 2010)
        FeatureDef("ko_complex_verb_ending", 0.75, _ko_complex_verb),          # raised: same pragmatic-loss class as SFEs
        FeatureDef("ko_topic_subject_contrast",    0.50, _ko_topic_subject_contrast),  # new: 은/는 vs 이/가 information-structure load
        FeatureDef("ko_retrospective_evidential",  0.75, _ko_retrospective_evidential), # new: -더- encodes directly-witnessed past (Chung 2010)
        FeatureDef("ko_quotative_reportative",     0.50, _ko_quotative_reportative),    # new: compressed attribution frame -다고 하다 → 대/래 (Ceong 2016)
        FeatureDef("ko_completive_auxiliary",      0.50, _ko_completive_auxiliary),     # new: V+버리다/놓다/두다 stance readings lost in MT (Kwon 2012)
        FeatureDef("ko_numeral_classifier",        0.30, _ko_numeral_classifier),       # new: two numeral systems + obligatory classifier (Kim 2009)
    ],
)


def _ko_subject_topic_omission(
    src: str, tgt: str, src_tok: Set[str], tgt_tok: Set[str], met: Dict
) -> bool:
    # Fires when the Korean target lacks subject/topic marking on a non-trivial source.
    # Signals that the translator made implicit choices about dropped pronouns.
    has_marking = any(
        len(tok) >= 2 and tok[-1] in _KO_SUBJECT_TOPIC_PARTICLES
        for tok in re.findall(r"[가-힣]+", tgt or "")
    )
    return not has_marking and met["src_tokens"] >= 5


_KO_CROSS = [CrossFeatureDef("ko_subject_topic_omission_likely", 0.5, _ko_subject_topic_omission)]


# ═══════════════════════════════════════════════════════════════════════════════
# ARABIC (ar)
# ═══════════════════════════════════════════════════════════════════════════════
#
# What makes Arabic hard to translate:
#   Discourse particles  يعني / والله / خلاص / يلا are culturally loaded and
#   have no single-word equivalents; translators must adapt the pragmatic intent.
#
#   Diglossia  Subtitles mix MSA and regional colloquials (Egyptian, Levantine,
#   Gulf).  Dialect-specific tokens (مش مو علشان إيه ليه …) signal register that
#   the target translation must preserve.
#
#   Negation variety  Five tense-bound MSA particles (لا لم لن ليس ما) plus
#   colloquial مش/مو/مب require the translator to infer tense from context.
#   Egyptian/Levantine clitic negation (ما-verb-ش, fused into one written word)
#   requires morphological decomposition before rendering.
#
#   Grammatical gender  Every noun, adjective, and 2nd/3rd person pronoun carries
#   gender (أنتَ m / أنتِ f; هو / هي).  Target languages without grammatical
#   gender must resolve gender from discourse context.

# Pragmatic particles / floor-holding devices
AR_DISCOURSE_MARKERS = {
    "يعني",   # ya'ni  — "I mean" / floor-holding filler
    "يلا",    # yalla  — "come on / let's go"
    "يالا",   # variant
    "والله",  # wallah — "I swear / by God" (emphasis, softening)
    "خلاص",  # khalas — "enough / done"
    "ماشي",  # mashi  — "okay / fine" (agreement)
    "حبيبي", # habibi — "dear" (m), social closeness marker
    "حبيبتي",# habibti — "dear" (f)
    "بس",    # bas    — "just / only / enough" (Levantine/Gulf)
    "طيب",   # tayyib — "alright / okay" (consent / topic shift)
}

# MSA negation (tense-dependent) + colloquial variants
AR_NEGATION_MARKERS = {
    "لا",    # la    — present / general negation
    "لم",    # lam   — past negation (MSA)
    "لن",    # lan   — future negation (MSA)
    "ليس",   # laysa — copula negation (MSA)
    "ما",    # ma    — multi-tense negation
    "مش",    # mish  — general negation (Egyptian / Levantine)
    "مو",    # mo    — negation (Gulf)
    "مب",    # mub   — negation (Gulf variant)
    "لأ",    # la'   — "no" (colloquial)
}

# Politeness / request markers
AR_REQUEST_MARKERS = {
    "أرجو",   # arjoo   — "I request / I beg"
    "لازم",   # lazim   — "must / necessary"
    "ممكن",   # mumkin  — "possible / may I"
    "تفضل",   # tafaddal — "please (come / take)" (m)
    "تفضلي",  # tafaddali — (f)
    "تعال",   # ta'al   — "come here" (m)
    "تعالي",  # ta'ali  — (f)
}

# Clear-cut dialectal tokens absent from MSA — unambiguous register signals
AR_COLLOQUIAL_MARKERS = {
    # Egyptian
    "مش",      # mish     — negation
    "إيه",     # eih      — "what"
    "ليه",     # leih     — "why"
    "إزاي",    # izzay    — "how"
    "كمان",    # kaman    — "also / too"
    "علشان",   # 3alshan  — "because / so that"
    "برضو",    # bardu    — "also / anyway"
    "برضه",    # bardu variant
    "دلوقتي",  # dilwa'ti — "now"
    "فين",     # fein     — "where" (Egyptian)
    "مين",     # meen     — "who" (Egyptian)
    "إمتى",    # emta     — "when" (Egyptian)
    # Levantine
    "هيك",     # heik     — "like this"
    "شو",      # shu      — "what"
    "وين",     # wen      — "where"
    "كيفك",    # kifak    — "how are you"
    "منين",    # mnein    — "from where"
    "رح",      # rah      — future marker
    "عم",      # am       — progressive marker
    "هلق",     # hala'    — "now"
    "مسكين",   # miskin   — "poor thing" (sympathy marker)
    # Gulf
    "زين",     # zain     — "okay / good" (Gulf)
    "تو",      # taw      — "just now" (Gulf)
    "ودي",     # widi     — "I want" (Gulf colloquial)
    "شفيه",    # shfih    — "what's wrong with him" (Gulf)
    "چذي",     # chadhi   — "like this" (Gulf)
}

# 3rd-person gender-marked pronouns.  When translating to Indonesian/Korean/Bengali
# (no grammatical gender), the translator must resolve the referent from context.
# Conversely, translating from genderless languages into Arabic requires gender inference.
AR_GENDER_PRONOUNS = {
    "هو",   # huwa — he
    "هي",   # hiya — she
}

# Pronouns with grammatical gender and colloquial variants
AR_PRONOUN_MARKERS = {
    "أنا",   # ana   — I
    "نحن",   # nahnu — we (MSA)
    "أنت",   # anta  — you (m, MSA)
    "أنتِ",  # anti  — you (f, MSA)
    "أنتم",  # antum — you (pl m, MSA)
    "أنتن",  # antunna — you (pl f, MSA)
    "هو",    # huwa  — he
    "هي",    # hiya  — she
    "هم",    # hum   — they (m)
    "هن",    # hunna — they (f)
    "هما",   # huma  — they (dual) / colloquial they
    "إنت",   # inta  — you (m, Egyptian)
    "إنتي",  # inti  — you (f, Egyptian)
    "إنتو",  # intu  — you (pl, Egyptian)
    "إحنا",  # ihna  — we (Egyptian)
}


# Conditional / concessive structures — require context-sensitive rendering
AR_CONDITIONAL_MARKERS = {
    "إذا",   # idha    — "if" (real condition)
    "لو",    # law     — "if" (hypothetical / wish)
    "لولا",  # lawla   — "if it weren't for"
    "لكن",   # lakin   — "but / however"
    "رغم",   # raghm   — "despite"
    "بالرغم",# bilraghm — "in spite of"
    "حتى",   # hatta   — "until / so that / even"
    "كي",    # kay     — "in order to"
    "إلا",   # illa    — "except / unless"
}

# Relative clause pronouns — signal syntactically complex sentences
AR_RELATIVE_MARKERS = {
    "الذي",    # alladhi   — "who/which" (m sg)
    "التي",    # allati    — "who/which" (f sg)
    "الذين",   # alladhina — "who/which" (m pl)
    "اللتان",  # allatan   — "which" (f dual)
    "اللذان",  # alladhani — "which" (m dual)
}

# Degree intensifiers and certainty markers — affect pragmatic force
AR_INTENSIFIER_MARKERS = {
    "جداً",      # jiddan    — "very"
    "جدا",       # jiddan (without tanwin)
    "أبداً",     # abadan    — "never / ever"
    "أبدا",
    "تماماً",    # tamaman   — "completely / exactly"
    "تماما",
    "بالتأكيد",  # certainly
    "بالضبط",    # exactly
    "للغاية",    # to the extreme
    "فعلاً",     # fi'lan    — "really / indeed"
    "فعلا",
    "طبعاً",     # tab'an    — "of course"
    "طبعا",
    "أكيد",      # akid      — "sure / certain" (colloquial)
}


def _ar_clitic_negation(text: str, tokens: Set[str]) -> bool:
    # Egyptian/Levantine discontinuous negation: ما-[verb]-ش fused into one word.
    # Requires morphological decomposition before translation.
    return bool(re.search(r"\bما\w+ش\b", text))


def _ar_dialect_present_prefix(text: str, tokens: Set[str]) -> bool:
    # Egyptian present-tense prefix ب (b-/bi-) before a verb stem of ≥ 3 Arabic
    # chars signals colloquial inflection absent from MSA.
    return bool(re.search(r"\bب[؀-ۿ]{3,}\b", text))


def _ar_gendered_reference(text: str, tokens: Set[str]) -> bool:
    # Presence of gender-marked 3rd-person pronouns (هو/هي) signals that the
    # translation must either preserve or infer grammatical gender.  Languages
    # without grammatical gender (Indonesian, Korean, Bengali) must resolve
    # the referent from discourse context rather than pronoun form.
    return bool(tokens & AR_GENDER_PRONOUNS)


# Broken plurals: >50% of Arabic plurals use internal vowel melody change (no suffix).
# Suffix-stripping stemmers miss all non-concatenative forms, producing singular or
# unrelated tokens (Al-Sughaiyer & Al-Kharashi 2004; Soudi et al. 2012 Benjamins).
AR_BROKEN_PLURALS = {
    "كتب", "رجال", "أولاد", "بيوت", "قلوب", "عيون", "أيدي",
    "أشياء", "مدارس", "مساجد", "عمال", "أسماء", "أحوال", "أفكار", "قضايا",
    "وجوه", "أوقات", "أصحاب", "أطفال", "أسرار", "أصوات", "أحداث", "نساء",
    "كلاب", "بلاد", "أمور", "عقول", "شوارع", "حروف", "ألوان",
    "أبناء", "إخوة", "أنهار", "أسلحة", "رؤوس", "قواعد", "أفراد",
    "جيوش", "بحار", "أيام", "لغات", "علوم", "فنون",
}

# Maf'ūl muṭlaq (cognate accusative): verbal noun in accusative intensifying its verb.
# English has no syntactic slot — must be rendered as adverb or dropped entirely,
# losing the intensification (Meteab & Kamil 2018, Semantic Scholar).
AR_MAF3UL_MARKERS = {
    "ضرباً", "ضربًا", "فرحاً", "فرحًا", "بكاءً", "نوماً", "قتالاً",
    "حباً", "حبًا", "كرهاً", "سرعةً", "شكراً", "شكرًا", "تماماً",
    "قطعياً", "حقاً", "فعلياً",
}

# Dual morphology: Arabic mandates a morphologically distinct form for exactly-two entities.
# English must choose "both X" / "the two X" based on discourse — no structural parallel
# (Abuaiadah et al. 2026 Springer; Soudi et al. 2012).
AR_DUAL_TOKENS = {
    "يومان", "يومين", "شخصان", "شخصين", "طرفان", "طرفين",
    "جانبان", "جانبين", "حالتان", "حالتين", "مرتان", "مرتين",
    "بلدان", "بلدين", "ولدان", "ولدين", "يدان", "يدين",
    "عينان", "عينين", "أخوان", "أخوين",
}


def _ar_broken_plural(text: str, tokens: Set[str]) -> bool:
    # Strip definite article الـ before matching (tokenizer keeps it attached).
    bare = {tok[2:] if tok.startswith("ال") else tok for tok in tokens}
    return bool(bare & AR_BROKEN_PLURALS)


def _ar_maf3ul_mutlaq(text: str, tokens: Set[str]) -> bool:
    return bool(tokens & AR_MAF3UL_MARKERS)


def _ar_dual_morphology(text: str, tokens: Set[str]) -> bool:
    return bool(tokens & AR_DUAL_TOKENS)


def _ar_prepositional_clitic_chain(text: str, tokens: Set[str]) -> bool:
    # Coordination + preposition stacked as a single orthographic token (وبـ / فلـ / وكـ).
    # Clitic tokenization is the single highest-impact preprocessing step for Arabic MT
    # (Habash et al. 2010 arXiv; Al-Sulaiti & Atwell 2006 BCS).
    return any(re.match(r"^[وف][بلك]", tok) for tok in tokens)


_AR_FEATURES = LanguageFeatures(
    code="ar",
    script_ratio_fn=arabic_script_ratio,
    min_script_ratio=0.35,
    ideal_script_ratio=0.50,
    pronoun_tokens=AR_PRONOUN_MARKERS,
    features=[
        FeatureDef("ar_discourse_marker",       1.0,  lambda _, toks: bool(toks & AR_DISCOURSE_MARKERS)),
        FeatureDef("ar_colloquial",             1.0,  lambda _, toks: bool(toks & AR_COLLOQUIAL_MARKERS)),
        FeatureDef("ar_negation",               0.5,  lambda _, toks: bool(toks & AR_NEGATION_MARKERS)),
        FeatureDef("ar_request_marker",         0.5,  lambda _, toks: bool(toks & AR_REQUEST_MARKERS)),
        FeatureDef("ar_conditional",            0.5,  lambda _, toks: bool(toks & AR_CONDITIONAL_MARKERS)),
        FeatureDef("ar_relative_clause",        0.5,  lambda _, toks: bool(toks & AR_RELATIVE_MARKERS)),
        FeatureDef("ar_intensifier",            0.3,  lambda _, toks: bool(toks & AR_INTENSIFIER_MARKERS)),
        FeatureDef("ar_clitic_negation",        1.0,  _ar_clitic_negation),           # raised: requires morphological decomposition (MDPI Arabic NLP 2024)
        FeatureDef("ar_dialect_present_prefix", 0.5,  _ar_dialect_present_prefix),
        FeatureDef("ar_gendered_reference",         0.30, _ar_gendered_reference),        # gender inference for genderless target languages
        FeatureDef("ar_broken_plural",              0.75, _ar_broken_plural),             # new: non-concatenative plurals; missed by suffix-stripping MT (Al-Sughaiyer 2004)
        FeatureDef("ar_maf3ul_mutlaq",              0.50, _ar_maf3ul_mutlaq),             # new: cognate accusative; no English syntactic slot (Meteab & Kamil 2018)
        FeatureDef("ar_dual_morphology",            0.50, _ar_dual_morphology),           # new: mandatory dual form; English must choose paraphrase (Abuaiadah 2026)
        FeatureDef("ar_prepositional_clitic_chain", 0.30, _ar_prepositional_clitic_chain),# new: conjunction+preposition stack in one token (Habash 2010)
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# BENGALI (bn)
# ═══════════════════════════════════════════════════════════════════════════════
#
# What makes Bengali hard to translate:
#   T-V pronoun system  Three address levels — আপনি (apni, formal), তুমি (tumi,
#   familiar), তুই (tui, intimate) — require the translator to infer the social
#   relationship from context.  Choosing the wrong level is face-threatening.
#
#   Discourse particles  Short particles (না তো রে গো আবার) carry heavy pragmatic
#   load: confirmation-seeking, intimacy signalling, contrastive emphasis.  They
#   have no single-word equivalents in most target languages.
#
#   Negation variants  নেই (nei, India) vs. নাই (nai, Bangladesh) are
#   sociolinguistically marked; নি (ni, perfective negation) attaches to verbs and
#   must be rendered as a periphrastic construction in many target languages.
#
#   Echo reduplication  ভেতর-শেতর ("inside and the like"), বাড়ি-টাড়ি ("home
#   etc.") — non-compositional, conveys dismissiveness or generality.
#
#   Gender-neutral 3rd person  সে / তারা cover he, she, and they; translating
#   into gender-marking languages requires contextual inference.

BN_DISCOURSE_MARKERS = {
    "না",     # na     — "no / right?" (topicaliser, tag question)
    "তো",     # to     — common-ground marker ("you know / after all")
    "রে",     # re     — intimate emphatic ("hey / come on")
    "গো",     # go     — emphatic softener (often female speech)
    "আবার",   # abar   — "again" / contrastive ("really? / but")
    "মানে",   # mane   — "I mean" (floor-holding, like Arabic يعني)
    "কিন্তু",  # kintu  — "but" (strong contrastive)
    "এই",     # ei     — "hey / this" (discourse call)
    "তবে",    # tobe   — "then / but" (conditional/concessive)
    "হ্যাঁ",   # hya    — "yes" (emphatic affirmation)
}

# Negation: India/Bangladesh split + register variation + perfective marker
BN_NEGATION_MARKERS = {
    "না",    # na   — general negation / refusal
    "নেই",   # nei  — "not there / doesn't exist" (India standard)
    "নাই",   # nai  — same, Bangladesh variant (sociolinguistically marked)
    "নি",    # ni   — perfective negation (hasn't done X)
    "নয়",   # noy  — copula negation ("is not")
    "নো",    # no   — colloquial copula negation
    "নহে",   # nohe — literary copula negation (Shadhubhasha)
}

# Formal address pronouns — their presence marks register-sensitive text
BN_HONORIFIC_MARKERS = {
    "আপনি",    # apni     — you (formal)
    "আপনার",   # apnar    — your (formal)
    "আপনাকে",  # apnake   — to you (formal)
    "আপনারা",  # apnara   — you all (formal)
    "আপনাদের", # apnader  — your (formal pl)
}

# Full pronoun set; সে/তারা are gender-neutral (asymmetry with gendered targets)
BN_PRONOUN_MARKERS = {
    "আমি",   # ami    — I
    "আমরা",  # amra   — we
    "আমার",  # amar   — my
    "আপনি",  # apni   — you (formal)
    "তুমি",  # tumi   — you (familiar)
    "তুই",   # tui    — you (intimate)
    "সে",    # se     — he / she / they (gender-neutral)
    "তারা",  # tara   — they (pl, gender-neutral)
    "তাকে",  # take   — him / her
    "তোমার", # tomar  — your (familiar)
    "তোর",   # tor    — your (intimate)
    "তাদের", # tader  — their
}


# Shadhubhasha (সাধুভাষা) archaic register vocabulary.  Sadhu forms appear in
# literary quotations, formal/legal text, and religious dialogue.  A translator
# must recognise the elevated register and handle it differently from colloquial
# Chaltibhasha — a dimension entirely separate from discourse particle detection.
BN_SADHU_MARKERS = {
    "করিতেছি", "করিতেছে", "করিতেছেন",  # Sadhu progressive
    "করিয়া", "বলিয়া", "গিয়া",          # Sadhu participle (verbal noun)
    "তাহার", "তাহাকে", "তাহারা", "তাহাদের",  # Sadhu 3rd-person pronouns
    "ইহা", "উহা", "যাহা", "সেহা",       # Sadhu demonstratives
    "কহিলেন", "বলিলেন", "গেলেন",        # Sadhu past-tense verbs
    "হইল", "হইয়া", "হইতে",             # Sadhu forms of হওয়া (be/become)
    "নহে", "নহি",                        # Sadhu copula negation
}

# Bengali light-verb auxiliaries (করা, দেওয়া, নেওয়া, ফেলা, রাখা).
# Compound verbs V+LV encode aktionsart (completion, benefaction, telicity) with
# no structural parallel in English or Korean.  E.g. খেয়ে ফেলা = "eat up (completely,
# often inadvertently)"; দিয়ে দেওয়া = "give away (for other's benefit)".
# Documented in Bengali grammar (Bhattacharja, ACL 2010 code-mixing verbs).
# Roots used for prefix-matching to cover all conjugated forms (ফেলে/ফেলেছে/ফেলব etc.).
# Expanded root set from ACL 2012 automatic extraction (Islam et al.) and
# ACL 2014 verb frames (Chakrabarti et al.): যাওয়া and আসা as directional
# vector verbs added alongside the core completion/benefactive set.
_BN_LIGHT_VERB_ROOTS = (
    "ফেল", "রাখ", "রেখ",           # ফেলা (completion/regret); রাখা/রেখে (retention)
    "দিয়", "দিল", "দিতে",          # দেওয়া: benefactive outward (for others)
    "নিয়", "নিল", "নিতে",          # নেওয়া: benefactive inward (for self)
    "গেল", "গেছ", "গিয়",           # যাওয়া: completion away from speaker
    "এল", "এসে", "এছ",             # আসা: directional toward speaker
)


def _bn_echo_reduplication(text: str, tokens: Set[str]) -> bool:
    # Bengali echo-word reduplication: Bengali-word + hyphen + Bengali-word.
    # E.g. ভেতর-শেতর ("inside and stuff"), বাড়ি-টাড়ি ("home and the like").
    # Non-compositional; conveys dismissiveness or vague generality.
    return bool(re.search(r"[ঀ-৿]+-[ঀ-৿]+", text))


def _bn_honorific(text: str, tokens: Set[str]) -> bool:
    return bool(tokens & BN_HONORIFIC_MARKERS)


def _bn_sadhu_register(text: str, tokens: Set[str]) -> bool:
    # Fires when Shadhubhasha vocabulary is present — signals elevated/literary register
    # that requires register-aware translation decisions separate from colloquial handling.
    return bool(tokens & BN_SADHU_MARKERS)


def _bn_gender_neutral_3p(text: str, tokens: Set[str]) -> bool:
    # সে (se) covers he/she/they (sg); তারা (tara) covers all gendered plurals.
    # When translating to Arabic (grammatical gender required) or Korean (formal
    # gendered forms), the translator must infer gender from discourse context.
    return "সে" in tokens or "তারা" in tokens


def _bn_compound_verb(text: str, tokens: Set[str]) -> bool:
    # Compound verb (V + light verb) construction, typically V+LV at SOV clause end.
    # The light verb modifies aktionsart or social direction of the action with no
    # structural parallel in English or Korean.
    # Root-prefix matching covers conjugated forms (ফেলে/ফেলেছে/ফেলব etc.).
    return any(tok.startswith(root) for tok in tokens for root in _BN_LIGHT_VERB_ROOTS)


# T-V intimate tier: তুই (tui) paradigm — face-threatening/pejorative when directed at
# social superiors or strangers; requires inferring relationship from context to translate
# (Thompson 2010 Academia; Rahman et al. 2024 MDPI Applied Sciences).
BN_INTIMATE_MARKERS = {"তুই", "তোর", "তোকে", "তোরা", "তোদের"}

# Bengali numeral classifiers: obligatory suffix encoding semantic class of counted noun.
# -টা/-টি also marks definiteness (বাড়িটা = "the house" vs. "a/one house"),
# so dropped classifiers lose both count and definiteness distinctions
# (Bengali grammar; Islam et al. 2012 ACL WILDRE; Chakrabarti et al. 2014 ACL).
_BN_CLASSIFIER_RE = re.compile(
    r"একটা|দুটো|তিনটে|চারটা|পাঁচটা|একটি|দুটি|একজন|দুজন|তিনজন|একখানা"
)
_BN_CLASSIFIER_SUFFIX_RE = re.compile(r"[ঀ-৿]+(?:টা|টি|টে|জন|খানা|খানি|গাছা)\b")


def _bn_intimate_register(text: str, tokens: Set[str]) -> bool:
    # তুই paradigm signals intimate/pejorative register — face-threatening outside its
    # licensed relationship; translator must resolve social context to match tone.
    return bool(tokens & BN_INTIMATE_MARKERS)


def _bn_numeral_classifier(text: str, tokens: Set[str]) -> bool:
    # Classifier suffix doubles as definiteness marker — MT dropping it loses both.
    return bool(_BN_CLASSIFIER_RE.search(text or "") or _BN_CLASSIFIER_SUFFIX_RE.search(text or ""))


_BN_FEATURES = LanguageFeatures(
    code="bn",
    script_ratio_fn=bengali_script_ratio,
    min_script_ratio=0.35,
    ideal_script_ratio=0.50,
    pronoun_tokens=BN_PRONOUN_MARKERS,
    features=[
        FeatureDef("bn_discourse_marker",   0.75, lambda _, toks: bool(toks & BN_DISCOURSE_MARKERS)),
        FeatureDef("bn_honorific",          0.75, _bn_honorific),
        FeatureDef("bn_negation",           0.75, lambda _, toks: bool(toks & BN_NEGATION_MARKERS)),  # raised: নি (perfective) requires periphrastic construction; নেই/নাই sociolinguistic split
        FeatureDef("bn_echo_reduplication", 0.5,  _bn_echo_reduplication),
        FeatureDef("bn_sadhu_register",     0.50, _bn_sadhu_register),          # archaic literary register (Shadhubhasha)
        FeatureDef("bn_compound_verb",      0.75, _bn_compound_verb),           # raised: ACL 2012/2014 confirm V+LV non-compositionality; expanded root set
        FeatureDef("bn_gender_neutral_3p",  0.30, _bn_gender_neutral_3p),       # gender inference required for Arabic/Korean targets
        FeatureDef("bn_intimate_register",  0.50, _bn_intimate_register),       # new: তুই paradigm — face-threat register (Thompson 2010)
        FeatureDef("bn_numeral_classifier", 0.30, _bn_numeral_classifier),      # new: classifier + definiteness dual role (Islam et al. ACL 2012)
    ],
)


# ── Language registry ─────────────────────────────────────────────────────────
#
# Add a new LanguageFeatures instance here to activate a new language.
# No changes needed in _opensubs_scoring.py.

LANG_REGISTRY: Dict[str, LanguageFeatures] = {
    "id": _ID_FEATURES,
    "ko": _KO_FEATURES,
    "ar": _AR_FEATURES,
    "bn": _BN_FEATURES,
}

# ── Cross-lingual features ────────────────────────────────────────────────────
#
# Cross features require both sides of a segment pair and are direction-specific.
# Key is (src_lang, tgt_lang).  These contribute to target-side complexity.

CROSS_FEATURES: Dict[Tuple[str, str], List[CrossFeatureDef]] = {
    ("id", "ko"): _KO_CROSS,
    ("ar", "ko"): _KO_CROSS,
    ("bn", "ko"): _KO_CROSS,
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_lang_features(lang: str) -> Optional[LanguageFeatures]:
    return LANG_REGISTRY.get(lang)


def get_cross_features(src_lang: str, tgt_lang: str) -> List[CrossFeatureDef]:
    return CROSS_FEATURES.get((src_lang, tgt_lang), [])
