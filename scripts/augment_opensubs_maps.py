import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from _content_safety import classify_value

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "consistency"))
from consolidate_consistency_runs import consolidate_group, DEDUP_THRESHOLD  # noqa: E402


LANGS: Dict[str, Dict[str, str]] = {
    "arb": {"name": "Arabic", "label": "Arabic"},
    "ben": {"name": "Bengali", "label": "Bengali"},
    "ind": {"name": "Indonesian", "label": "Bahasa Indonesia"},
    "kor": {"name": "Korean", "label": "Korean"},
}

# TMX pair name → (src ISO 639-3, tgt ISO 639-3)
PAIR_LANGS: Dict[str, Tuple[str, str]] = {
    "ar-bn": ("arb", "ben"),
    "ar-id": ("arb", "ind"),
    "ar-ko": ("arb", "kor"),
    "bn-id": ("ben", "ind"),
    "bn-ko": ("ben", "kor"),
    "id-ko": ("ind", "kor"),
}

# Pair-specific cultural context injected into every checklist generation call.
# Distilled from docs/cultural_context_pairs.md (peer-reviewed sources).
# Covers: communication asymmetries, asymmetric cultural concepts, key bridging terms.
CULTURAL_CONTEXT: Dict[str, str] = {
    "ar-bn": """\
Cultural asymmetries for Arabic ↔ Bengali interpretation:

Communication style: Arabic uses emotional expressiveness, rhetorical repetition, and hyperbole as emphasis (not exaggeration). Bengali uses a three-tier pronoun system (apni/tumi/tui) and extensive kinship address forms to encode hierarchy. An Arab's animated, emphatic style can read as angry to a Bengali; a Bengali's quiet deference can read as cold to an Arab.

Key asymmetric concepts — Arabic speakers know, Bengalis may not:
- Wasta (واسطة): using personal connections to access jobs/permits is legitimate social capital, not corruption.
- Inshallah (إن شاء الله): sincere theistic deference ("God willing"), not evasion or "maybe." Bengalis may misread it as non-commitment.
- Fusha vs ammiya diglossia: Bengalis who learn Arabic for religious recitation (tajweed) cannot understand spoken Gulf or Egyptian dialect — a gap that surprises both parties.
- Haram as comprehensive system: not just "forbidden food" but a complete moral-legal framework covering slaughter method, alcohol in cooking, cross-contamination.

Key asymmetric concepts — Bengali speakers know, Arabs may not:
- Lajja (লজ্জা): shame/modesty with a positive valence when displayed — signals appropriate modesty, not disgrace. Not equivalent to Arabic 'aar.
- Adda (আড্ডা): extended, unstructured intellectual conversation as a cultivated social practice. No Arab equivalent.
- Pir/Murshid: Sufi teacher/saint with hereditary devotional authority — not the same as an Arab sheikh. Shrine-visiting (urs) may be seen as bid'a by Gulf Arabs.
- Bengali Islam is Sufi-syncretic (Baul tradition, shrine veneration, Persian-derived ritual vocabulary like namaz instead of salah). Gulf Arab Islam treats some of this as bid'a.
- Ekushey/Language Martyrs (Feb 21): the foundational identity myth of Bangladeshi nationhood. Any Bengali expects this reference to be understood.

Key terms requiring bridging:
- Inshallah: explain as genuine theistic deference, not "maybe"
- Namaz (নামাজ): the Bengali/Persian word for prayer — Arab may not recognize it
- Pir: Sufi saint with hereditary authority — not equivalent to sheikh
- Lajja: no single Arabic equivalent; context-dependent (haya/khajal/aar)
- Adda: "intimate extended intellectual conversation," not small talk
- Wasta: Bengali approximate is joruri manush, lacks institutionalized status""",

    "ar-id": """\
Cultural asymmetries for Arabic ↔ Indonesian interpretation:

Communication style: Arabic communication is expressive, assertive, and uses volume variation and repetition for emphasis — this reads as conflict or anger in Indonesian (especially Javanese) contexts. Indonesian communication is built around rukun (active social harmony maintenance) and face-saving. Saying "yes" (iya) in Indonesian may only mean "I hear you," not agreement. Indonesians use belum ("not yet"), sulit ("difficult"), nanti ("later") as indirect refusals — an Arab interpreting these as honest answers will be repeatedly frustrated.

Key asymmetric concepts — Arabic speakers know, Indonesians may not:
- Wasta: legitimate connection-based access, not corruption.
- Kafala system: Gulf employer-sponsorship for migrant workers — Arabs experience it as employers; Indonesians who have worked in the Gulf know it as a feared lived reality.
- Fusha/ammiya diglossia: pesantren-educated Indonesian Arabic speakers know Quranic Arabic but cannot understand spoken Gulf/Egyptian dialect.
- Gender separation norms: in conservative Gulf contexts, unrelated men and women are not expected to interact professionally — Indonesian mixed-gender workplaces will surprise Arab visitors.

Key asymmetric concepts — Indonesian speakers know, Arabs may not:
- Rukun: social harmony as an ACTIVE obligation — suppressing disagreement and maintaining smooth surfaces is the social contract, not mere politeness.
- Gotong royong: communal labor obligation — neighbors participate in each other's weddings/funerals/building projects as natural communal duty.
- Malu: shame/embarrassment as a pervasive social regulator; closer to Chinese mianzi than Arabic 'aar. Context-positive when signaling modesty.
- Selamatan: communal feast combining Quranic prayer with pre-Islamic ancestral invocations — Gulf Arabs would classify some elements as bid'a/shirk.
- Pesantren: Indonesia's ~26,000 Islamic boarding schools have a kyai (traditional scholar) whose authority derives from lineage and social embeddedness, not formal state certification.
- Halal certification gap: Arabs treat halal as baseline default; Indonesians treat it as a verifiable, certifiable status (BPJPH) because they live in a pluralistic food environment.
- Haji/Hajjah as permanent social title in Indonesia: completing Hajj grants a lifelong honorific. Not the case in Arab countries near Mecca.

Key terms requiring bridging:
- Inshallah → insya Allah (Indonesian): used in stronger literal divine-will register, not routinely as soft refusal
- Rukun: "active social harmony," not just "peace"
- Malu: shame-modesty; positive when signaling appropriate restraint
- Jam karet: "rubber time" — culturally endorsed temporal flexibility
- Selamatan: syncretic ritual feast; requires explanation of pre-Islamic elements
- Kyai: traditional scholar with lineage-based authority; not equivalent to Arab sheikh
- Halal certification: verify, not assume, in Indonesian context""",

    "ar-ko": """\
Cultural asymmetries for Arabic ↔ Korean interpretation:
(These cultures have minimal historical contact — asymmetries are especially large.)

Communication style: Arabic fills silence with conversation, uses emotional expressiveness and repetition for emphasis. Korean nunchi (눈치) — reading unspoken social cues — is a foundational skill; explicitly stating needs signals low social awareness. Korean silence after a question can signal "no" or discomfort. Kibun (기분, collective mood) must be protected; stating an uncomfortable truth damages kibun and is a social violation. An Arab's assertive style is likely to damage kibun; a Korean's silence and indirect deflection will be misread as evasion by an Arab.

Key asymmetric concepts — Arabic speakers know, Koreans may not:
- Inshallah: not "maybe" — sincere theological acknowledgment that the future is in God's hands. Koreans (~96% don't know what halal means) will misread this as non-commitment.
- Halal/haram as system: comprehensive legal-ethical framework covering slaughter method, cross-contamination, cooking alcohol, additives — far beyond "no pork."
- Ramadan restructuring: an entire month that reshapes business hours, meeting availability, and daily schedule. No Korean equivalent calendar disruption.
- Gender separation norms: in conservative Gulf contexts, unrelated men and women do not professionally interact — unexpected to Koreans.
- Wasta: connection-based access as legitimate social capital.

Key asymmetric concepts — Korean speakers know, Arabs may not:
- Nunchi (눈치): the ability to read unspoken social atmosphere — a core social competency. Its absence is a major social failing.
- Kibun (기분): collective mood that must be managed; damaging it even accidentally is a social error.
- Ppalli-ppalli (빨리빨리): urgency culture — speed and efficiency are moral virtues; Arab temporal flexibility reads as unreliability.
- Hoesik (회식): mandatory workplace alcohol-bonding dinners; refusing or leaving early signals disloyalty. Completely incompatible with practicing Muslim's constraints — a genuine structural inclusion barrier.
- Seonbae/hoobae (선배/후배): rigid age-based senior/junior hierarchy governing speech registers. Addressing a Korean by first name without permission is rude.
- Chemyeon (체면): public face/dignity as social currency; deep distress when lost.
- Jeong (정): deep relational bond accumulated over shared time — creates obligation, not just warmth.

Key terms requiring bridging:
- Inshallah: genuine divine deference, not evasion; explain as "I sincerely intend this, God willing"
- Halal/haram: comprehensive system, not just pork prohibition
- Ramadan: explains daytime unavailability and schedule restructuring for a full month
- Nunchi: "reading the room" — indirect signals replace direct speech
- Kibun: mood that must be protected; disrupting it is a social error
- Hoesik: mandatory company drinking — impossible for observant Muslim; interpreter must provide face-saving reformulation
- Ppalli-ppalli: Korean urgency culture — slow responses read as disrespect""",

    "bn-id": """\
Cultural asymmetries for Bengali ↔ Indonesian interpretation:

Communication style: Bengali is emotionally expressive and argument-comfortable — adda culture values vigorous intellectual debate as a social good. Indonesian (especially Javanese) is built around rukun (active harmony maintenance) and malu (shame as regulator). A Bengali engaging in vigorous debate reads as aggressive to a Javanese Indonesian; an Indonesian's indirect belum/sulit/nanti ("not yet"/"difficult"/"later") reads as honest uncertainty to a Bengali, when it is actually a polite "no." Both are Muslim-majority but different enough in Islamic practice that shared vocabulary masks deep practice differences.

Key asymmetric concepts — Bengali speakers know, Indonesians may not:
- Adda (আড্ডা): extended, unstructured intellectual conversation as a cultivated social practice — a core Bengali social form. Indonesians may find it disorganized or pointless.
- Partition memory (1947 and 1971): the Language Movement and Liberation War genocide are foundational to Bangladeshi identity. Any Bengali will expect these references to be understood; an Indonesian has no framework for them.
- Bhadralok: educated, reform-minded Bengali cultural elite defined by literary engagement and secular-nationalist sensibility; often skeptical of Gulf-style orthodoxy.
- Obhiman (Bengali): wounded expectation/silent relational hurt felt toward someone close — neither accusatory nor angry, it expects recognition. No Indonesian equivalent.
- Bengali Islam is Sufi-syncretic (Baul tradition, pir devotion, Persian ritual vocabulary). Indonesian Islam Nusantara is also syncretic but on a Hindu-Buddhist-Animist substrate. A Tablighi-influenced Bangladeshi may find Indonesian syncretism insufficient; an Indonesian may find Bangladeshi Tablighi pietism unnecessarily rigid.

Key asymmetric concepts — Indonesian speakers know, Bengalis may not:
- Rukun: social harmony as an ACTIVE obligation — not just being nice but a structural duty to suppress disagreement.
- Gotong royong: institutionalized communal labor obligation — neighbors participate in each other's events as natural social order.
- Malu: shame as behavior-governor — explains reluctance to speak up, correct others publicly, or stand out inappropriately.
- Jam karet: culturally endorsed temporal flexibility — not disrespect but an explicit named social norm.
- Selamatan: communal feast combining Quranic prayer with pre-Islamic ancestral elements — a Tablighi-influenced Bengali would classify parts as bid'a.
- Islam Nusantara: deliberate framing of Indonesian Islam as contextually inclusive of local adat — structurally different from Bangladeshi Islamic approaches.
- Pancasila: the national ideology of managed religious pluralism — explains why Indonesian Muslims do not demand religious monism in public life.

Key terms requiring bridging:
- Adda: "leisurely extended intellectual conversation as social practice" — not mere chatting
- Obhiman: wounded relational expectation — no Indonesian equivalent; must be paraphrased
- Lojja (লজ্জা): shame/modesty; similar to malu but expressed more verbally/emotionally than behaviorally
- Rukun: active social harmony obligation, not passive goodwill
- Malu: shame as behavior-governor — explains indirect communication patterns
- Belum/Nanti/Sulit: indirect refusal signals — interpreter must bridge as "no" explicitly
- Selamatan: syncretic communal ritual; requires explanation of pre-Islamic elements""",

    "bn-ko": """\
Cultural asymmetries for Bengali ↔ Korean interpretation:
(These cultures have minimal historical contact — asymmetries are significant.)

Communication style: Bengali is emotionally expressive, argument-comfortable, and literary in register. Korean is hierarchically indirect — nunchi (reading unspoken social cues) is a core competency, and kibun (collective mood) must be protected at all times. When a Bengali asks a direct "yes or no" question, a Korean's indirect hedge-filled non-answer means "no" — a Bengali will miss this. Korean silence signals discomfort or disagreement; a Bengali reads silence as thinking. Bengali emotional expressiveness reads as excessive to Koreans; Korean emotional restraint reads as cold to Bengalis.

Key asymmetric concepts — Bengali speakers know, Koreans may not:
- Adda (আড্ডা): extended leisurely intellectual conversation as a social form — no Korean equivalent.
- Obhiman (অভিমান): wounded expectation / silent relational hurt — neither accusatory nor angry, expects recognition. No Korean equivalent.
- Bhadralok: specific colonial-era cultural elite identity defined by literary refinement and secular-nationalist sensibility.
- Partition trauma (1947 and 1971): foundational Bengali/Bangladeshi identity wound. No Korean framework for this.
- Halal requirements: Bengali Muslims have comprehensive halal constraints (slaughter, cross-contamination, additives) — Korean default is pork-and-alcohol-heavy social culture, which is doubly incompatible.
- K-wave asymmetry: Korean pop culture (Hallyu) has massive reach in Bangladesh; Koreans have almost no awareness of Bengali culture.

Key asymmetric concepts — Korean speakers know, Bengalis may not:
- Nunchi (눈치): reading unspoken social atmosphere — core social competency; its absence is a major failing.
- Kibun (기분): collective mood that must be managed; disrupting it is a social error.
- Chemyeon (체면): public face/dignity as social currency; deep distress when damaged.
- Ppalli-ppalli (빨리빨리): urgency as moral virtue — slow responses carry social cost.
- Hoesik (회식): mandatory company alcohol-bonding dinners — career consequences for refusal; impossible for observant Muslims. Interpreter must provide face-saving reformulation.
- Jeong (정): deep relational bond built over time — creates mutual obligation, not just warmth.
- Jesa (제사): ancestral memorial rites — immovable family obligation practiced across religious identities. No Bengali/Muslim equivalent.
- Korean speech levels: systematic grammatical hierarchy in every utterance. Addressing a Korean by first name is disrespectful without explicit permission.

Key terms requiring bridging:
- Adda: extended intellectual social conversation — not small talk
- Obhiman: wounded relational expectation — no Korean equivalent; must be paraphrased with emotional context
- Lajja (লজ্জা): shame/modesty — structurally similar to chemyeon but more familial/communal
- Halal: comprehensive system (slaughter, cross-contamination, additives) — Korean default assumes none of this
- Nunchi: reading the room without being told — Korean core social competency
- Kibun: collective mood requiring protection — disrupting it is a social violation
- Hoesik: mandatory work drinking — explain that refusal is socially costly but impossible for observant Muslim
- Jesa: ancestral rites — immovable family calendar obligation; signals filial piety ethic""",

    "id-ko": """\
Cultural asymmetries for Indonesian ↔ Korean interpretation:
(The most culturally load-bearing pair in this project.)

Communication style: Indonesian (especially Javanese) indirectness is horizontal — it preserves social harmony and avoids imposing. Korean indirectness is vertical — it protects hierarchical order and kibun. Both avoid direct refusal, but differently: Indonesian "tidak enak" (literally "not comfortable/delicious") = "I cannot"; Indonesian belum/nanti/sulit = "no" delivered with maximum face-preservation. Korean "I'll think about it" or prolonged silence also means "no." An Indonesian saying "yes" (iya) may mean "I hear you," not agreement. An Indonesian asking a Korean's age is establishing connection; the Korean is calculating which speech level to use.

Key asymmetric concepts — Indonesian speakers know, Koreans may not:
- Rukun: social harmony as an ACTIVE obligation — not just "no conflict" but the positive cultivation of cooperative, caring social relations. Underlies how Indonesian communities manage all public interactions.
- Gotong royong: institutionalized communal labor — enshrined in Pancasila; no Korean equivalent.
- Malu: shame-embarrassment as pervasive regulator — may prevent an Indonesian employee from reporting a problem to their supervisor even when they know the solution.
- Sungkan: respectful reluctance to impose on or inconvenience others, especially superiors. An Indonesian who repeatedly says "no need, it's fine" while clearly needing help is practicing sungkan — requires gentle insistence.
- Tidak enak: "I feel uncomfortable" = indirect refusal; NOT a comment about food.
- Musyawarah: consensus through deliberation — Indonesian decisions require group agreement, which explains slow and opaque decision-making to Korean interlocutors.
- Halal as DEFAULT (not special request): Indonesian Muslims do not ask for halal as a preference — it is the baseline non-negotiable condition. An Indonesian Muslim cannot eat samgyeopsal (pork belly), drink soju, or eat kimchi with pork-broth base.
- Nasi tumpeng: ceremonial cone-shaped rice — its presence at an event signals formality, celebration, and spiritual significance; not merely "interesting food."
- Batik: UNESCO 2009 heritage textile — wearing it at formal events carries identity significance.

Key asymmetric concepts — Korean speakers know, Indonesians may not:
- Nunchi (눈치): reading unspoken social atmosphere — core social competency. Absence is a major social failing.
- Kibun (기분): collective mood requiring protection; disrupting it is a social violation.
- Ppalli-ppalli (빨리빨리): urgency as cultural virtue — slow responses and deliberative processes read as disrespect.
- Hoesik (회식): mandatory company alcohol-bonding dinners — career consequences for refusal. For Indonesian Muslim employees in Korean workplaces, this is a genuine structural inclusion crisis, not a personal preference.
- Jeong (정): deep relational bond built through shared time and shared meals — an Indonesian who avoids all hoesik may prevent jeong from forming, creating persistent social distance.
- Chemyeon (체면): public face/dignity — governs modesty performances (downplaying achievement) and conflict avoidance.
- Jesa (제사): ancestral memorial rites — immovable family calendar obligation practiced across religions.
- Korean speech levels: systematic grammatical hierarchy. Addressing a Korean by first name without permission is disrespectful.
- K-wave asymmetry: Indonesians often arrive with K-drama-mediated impressions of Korea that are both informed and distorted; Koreans have almost no reciprocal awareness of Indonesian culture.

Key terms requiring bridging:
- Rukun: "active social harmony," not just peace — requires cultivation
- Gotong royong: communal cooperative labor — institutionalized in national ideology
- Malu: shame as behavior-governor; explains reluctance to speak up or stand out
- Sungkan: respectful reluctance to impose — means "I hold back out of respect," not "I don't want"
- Tidak enak: "I feel uncomfortable" = indirect refusal; must be bridged as "no"
- Musyawarah: consensus deliberation — explains why Indonesian decisions seem slow to Koreans
- Halal: non-negotiable baseline for Indonesian Muslims — extends far beyond "no pork and no alcohol"
- Nasi tumpeng: ceremonial dish; its presence signals a formal/spiritual occasion
- Nunchi: reading unspoken social atmosphere — core Korean social competency
- Kibun: collective mood that must be protected
- Hoesik: mandatory company drinking — explain that refusal has career consequences but is religiously impossible for observant Muslim; interpreter must provide face-saving alternative framing
- Ppalli-ppalli: Korean urgency culture — slow responses read as disrespect, not thoughtfulness
- Jeong: deep relational bond built over time; explains why hoesik avoidance has relational consequences""",
}

# Maps mining-pipeline reason tags to actionable evaluation hint text.
# Unmapped tags are shown as-is (no hint).
FEATURE_HINT_MAP: Dict[str, str] = {
    # ── Arabic-specific ──────────────────────────────────────────────────────
    "ar_negation": (
        "Source uses Arabic negation (lā/mā/lan/lam); check that the pragmatic force "
        "(denial, prohibition, soft refusal, hedging) is preserved — not just the grammar."
    ),
    "ar_clitic_negation": (
        "Source uses Arabic clitic negation (e.g., -sh in Egyptian/Levantine); "
        "verify the colloquial negation is rendered with appropriate register in the target."
    ),
    "ar_colloquial": (
        "Source is in Arabic colloquial/dialect register (not fusha); "
        "check whether the colloquial intimacy, informality, or regional marking is preserved or appropriately adapted."
    ),
    "ar_conditional": (
        "Source contains an Arabic conditional (law/in/idha); "
        "check that the conditionality and implied hypothetical or warning force are preserved."
    ),
    "ar_dialect_present_prefix": (
        "Source uses Arabic dialect present-tense prefix (b-/bi-), marking spoken register; "
        "verify register and dialectal informality are reflected in the target."
    ),
    "ar_discourse_marker": (
        "Source uses an Arabic discourse marker (yʿani/hallas/tayeb/masalan); "
        "check that the connective or hedging function is preserved, not literally translated."
    ),
    "ar_intensifier": (
        "Source uses an Arabic intensifier (jiddan/kathiran/wallah); "
        "check that the emotional emphasis or oath-like force is conveyed at an appropriate level in the target."
    ),
    "ar_relative_clause": (
        "Source uses an Arabic relative clause (alladhi/allati); "
        "check that the referential scope and any implied specificity are preserved in the target structure."
    ),
    "ar_request_marker": (
        "Source uses an Arabic request marker (min fadlak/rajāʾan/law samaḥt); "
        "check that the politeness level and face-saving function of the request are preserved in the target."
    ),
    # ── Bengali-specific ────────────────────────────────────────────────────
    "bn_honorific": (
        "Source uses Bengali honorific address (apni/tumi/tui or kinship terms like dada/didi/bhai); "
        "check that the social relationship and hierarchy encoded are preserved in the target language's address system."
    ),
    "bn_negation": (
        "Source uses Bengali negation (na/nei/noy); "
        "check that the pragmatic function (outright denial, soft refusal, existential negation) is preserved."
    ),
    "bn_discourse_marker": (
        "Source uses a Bengali discourse marker (tobe/kintu/tai/aar); "
        "check that the contrastive, additive, or conclusive function is preserved."
    ),
    "bn_echo_reduplication": (
        "Source uses Bengali echo-reduplication (e.g., 'chaay-taay', 'khaoa-daoa'); "
        "check that the generalization or vagueness effect ('tea or something', 'eating and all') is conveyed."
    ),
    # ── Indonesian-specific ─────────────────────────────────────────────────
    "id_affix_complexity": (
        "Source uses Indonesian complex affixation (meN-/di-/ber-/ter-/ke-an/peN-an); "
        "check that the voice, focus, or aspect shift encoded in the affix is preserved in the target."
    ),
    "id_reduplication": (
        "Source uses Indonesian reduplication (jalan-jalan, anak-anak, sayur-mayur); "
        "check that the plurality, variety, or intensity meaning is preserved — not read as a single-word repetition."
    ),
    "id_slang_or_colloquial": (
        "Source uses Indonesian slang or colloquial form (lo/gue, nggak, udah, dll.); "
        "check that the informal register and in-group signal are preserved or appropriately adapted."
    ),
    "id_soft_discourse_marker": (
        "Source uses a soft Indonesian discourse marker (sih, deh, dong, kok); "
        "these are face-saving particles that soften requests or signal frustration — check that their pragmatic function is preserved."
    ),
    "id_strong_discourse_marker": (
        "Source uses a strong Indonesian discourse marker (kan, lah, toh, memang); "
        "these signal assertion, shared knowledge, or mild challenge — check that the rhetorical stance is preserved."
    ),
    # ── Korean-specific ─────────────────────────────────────────────────────
    "ko_honorific_or_social_marker": (
        "Source uses Korean speech-level or honorific marking (formal -습니다, polite -아요/어요, informal -아/어, intimate -야/아); "
        "check that the exact speech level and relational stance are preserved in the target — this is non-optional in Korean."
    ),
    "ko_complex_verb_ending": (
        "Source uses a Korean complex verb ending (-(으)ㄹ게요, -겠-, -(으)ㄹ 것 같다, 아/어야 하다); "
        "check that the modality (volition, conjecture, obligation) is preserved, not flattened to a simple statement."
    ),
    "ko_sentence_ending_particle": (
        "Source uses a Korean sentence-ending particle (요, 죠, 네, 군, 지); "
        "check that the discourse function (confirmation-seeking, new information, shared knowledge) is preserved."
    ),
    "ko_colloquial_expression": (
        "Source uses Korean colloquial expression (애, 걔, 뭐, 어, 진짜); "
        "check that the casual register and in-group familiarity are preserved or adapted appropriately."
    ),
    "ko_subject_topic_omission_likely": (
        "Source likely omits subject/topic (pro-drop); "
        "check that the referent is correctly recovered and the implied subject is consistent with the surrounding context."
    ),
    # ── Cross-language structural ─────────────────────────────────────────────
    "negation": (
        "Source contains negation; check that the pragmatic force (denial, prohibition, soft refusal, hedging) "
        "is preserved in the target — not just the grammatical negation structure."
    ),
    "question": (
        "Source is a question; check the speech act type (genuine inquiry, rhetorical, indirect request, challenge) "
        "and whether the question force is preserved in the target."
    ),
    "request_or_imperative": (
        "Source contains a request or imperative; check that the directness level and face-saving strategy "
        "are appropriate for the target culture's norms around making requests."
    ),
    "exclamation": (
        "Source is an exclamation; check that the emotional register and intensity are preserved — "
        "not over-formalized or under-expressed in the target."
    ),
    "ellipsis": (
        "Source relies on ellipsis or pragmatic omission; check that the implied meaning is fully "
        "recoverable in the target from context, especially where the target language requires more explicit marking."
    ),
    "pronoun_asymmetry": (
        "Source-target pronoun systems are asymmetric; check that person, number, and honorific/social "
        "relationships encoded in pronouns are correctly mapped to the target's address system."
    ),
    "multi_clause_or_turn": (
        "Source spans multiple clauses or turn units; check that all logical and pragmatic relationships "
        "between clauses (cause, contrast, condition, sequence) are preserved in the target."
    ),
    "digit_mismatch": (
        "Source and target have mismatched digit/number representation; "
        "verify that numbers are correctly transferred and any unit or currency localization is appropriate."
    ),
    "qmark_mismatch": (
        "Question mark count or placement differs between source and target; "
        "verify that the interrogative force is preserved as intended."
    ),
    "exclaim_mismatch": (
        "Exclamation mark usage differs between source and target; "
        "verify that the emotional intensity and expressiveness are preserved."
    ),
}


def _expand_reason_hints(tags: List[str]) -> str:
    """Map reason tags to hint text; unmapped tags shown as-is."""
    lines: List[str] = []
    for tag in tags:
        hint = FEATURE_HINT_MAP.get(tag)
        if hint:
            lines.append(f"• [{tag}] {hint}")
        else:
            lines.append(f"• [{tag}]")
    return "\n".join(lines) if lines else "(none)"


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
        "pragmatic_analysis": _as_string(raw.get("pragmatic_analysis")),
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


PROMPT_TEMPLATE = """You are an expert evaluation data designer for cross-cultural interpreter-mediated communication.

Your task: convert a bilingual subtitle segment into structured evaluation metadata for a one-turn interpretation simulation.

Direction: {source_language} ({source_language_code}) → {target_language} ({target_language_code})

{cultural_context_block}Source turn (what the interpreter receives):
{source_text}

Reference target turn (pragmatic reference only — not ground truth):
{reference_target_text}
{reference_alignment_note}

Detected linguistic features and their evaluation implications:
{reason_tags_block}

Context digest (must be reflected in outputs):
{context_digest}

Previous context window:
{prev_context}

════════════════════════════════════════════════════
STEP 1 — Pragmatic analysis (reason before generating)
════════════════════════════════════════════════════

Analyze the source turn and produce a compact analysis covering ALL of:
A. Speech act: primary communicative act (request / refusal / apology / assertion / question / complaint / promise / greeting / challenge / other)
B. Social relationship: power/solidarity dynamic presupposed (superior→subordinate / peer / subordinate→superior / stranger / intimate / etc.)
C. Face stakes: is there a face-threatening act? What mitigation does the source culture use, and what does the target culture expect instead?
D. Cultural failure points: given the pair-specific asymmetries above, name 2–3 concrete ways this specific utterance could fail in translation — grounded in the actual cultural gap, not generic errors.
E. Required target form: what register, honorific level, and grammatical form must the target use?

Output this as "pragmatic_analysis": a 3–5 sentence paragraph that a human judge could use to evaluate the translation.

════════════════════════════════════════════════════
STEP 2 — Generate all output fields
════════════════════════════════════════════════════

Using your Step 1 analysis, generate all fields below.

CHECKLIST FORMAT RULE — mandatory for every item:
Every checklist item must be a yes/no question starting with "Does the" targeting:
• "Does the translation ..."         — Layer 1: fidelity and semantic accuracy
• "Does the interpreter's response ..." — Layers 2 & 3: pragmatic function and communicative goal

The distinction matters: Layer 1 checks whether the translation is *accurate*. Layers 2–3 check whether the interpreter's *response achieves its communicative goal* in the target cultural context — which may require culturally-adapted choices beyond word-for-word accuracy.

TASK REQUIREMENTS:
1)  Infer speech_act_intent (≤8 words) and semantic_core from source + context.
2)  Produce mandatory_cultural_constraints grounded in Step 1 cultural failure points (D).
3)  Build roleplay-ready contexts while de-identifying movie specifics.
4)  Do not include actor names, film titles, or scene-specific lore.
5)  user_a_context must be in source language. user_b_context must be in target language.
6)  Checklist items must use the "Does the translation/interpreter's response ..." format (YES = success).
7)  Enforce checklist priority: layer_3 count >= layer_2 count >= layer_1 count.
8)  conversation_context must be grounded in previous context window only; no transcript-style turn history.
9)  Do not include the current source turn in conversation_context, context_window_summary, or user contexts.
10) context_window_summary: 2–4 English sentences about previous context only.
11) Both user contexts: rich role/situation grounding in each user's language without exposing past utterances.
12) user_a_context must explicitly identify User A (source-side); user_b_context must identify User B (target-side).
13) Do not include guidance phrasing: "Saya harus/akan", "I must/will", "해야", "할 거".
14) Do not include target-side plans, expected replies, or strategy hints in user_b_context.
15) verification_prompt: numbered lines (1., 2., ...).
16) Checklist must include at least one criterion for contextual coherence with surrounding turns.
17) layer_3 must include at least one criterion grounded in the Step 1 cultural failure points (D).

Output JSON only with this schema:
{{
  "pragmatic_analysis": "string",
  "speech_act_intent": "string",
  "semantic_core": "string",
  "mandatory_cultural_constraints": ["string"],
  "context_window_summary": "string",
  "conversation_context": "string",
  "user_a_context": "string",
  "user_b_context": "string",
  "checklist": {{
    "layer_1_semantic_core": ["Does the translation ..."],
    "layer_2_pragmatic_function": ["Does the interpreter's response ..."],
    "layer_3_cultural_social_constraints": ["Does the interpreter's response ..."]
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


def _load_existing_keys(path: Path) -> Set[Tuple[str, str, str]]:
    """Keys are (segment_file, segment_id, direction_label) to allow both directions per row."""
    keys: Set[Tuple[str, str, str]] = set()
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
            direction = str(obj.get("direction", ""))
            if seg_file or seg_id:
                keys.add((seg_file, seg_id, direction))
    return keys


def _iter_context(ctx: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for r in ctx:
        sid = r.get("segment_id", "")
        src = str(r.get("source_text", "")).strip()
        tgt = str(r.get("target_text", "")).strip()
        lines.append(f"- [{sid}] src: {src} | tgt: {tgt}")
    return lines


def _shorten(text: str, max_chars: int = 140) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_context_digest(row: Dict[str, Any], is_fwd: bool) -> str:
    prev_ctx = row.get("prev_context", []) or []
    current_source = _as_string(
        row.get("source_text" if is_fwd else "target_text", "")
    )
    src_field = "source_text" if is_fwd else "target_text"
    prev_source = [_as_string(x.get(src_field, "")) for x in prev_ctx]
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


def _fallback_context_window_summary(row: Dict[str, Any]) -> str:
    n = len(row.get("prev_context", []) or [])
    return (
        f"The prior context window contains {n} turns that shape the immediate interaction conditions. "
        "It provides grounding from earlier dialogue only, without predicting upcoming actions or replies."
    )


def _fallback_conversation_context(row: Dict[str, Any]) -> str:
    n = len(row.get("prev_context", []) or [])
    reason_tags = _extract_reason_tags(row)
    reason_text = ", ".join(reason_tags[:3]) if reason_tags else "local pragmatic continuity"
    return (
        f"The interaction is already in progress with {n} earlier turns. "
        f"The current source utterance should be interpreted with {reason_text} carried from prior dialogue state. "
        "This context is grounding only and does not imply what either user will say next."
    )


def _fallback_user_a_context(row: Dict[str, Any]) -> str:
    n = len(row.get("prev_context", []) or [])
    return (
        f"User A (source-side speaker): The conversation has {n} prior turns "
        "establishing the relationship, register, and emotional tone. "
        "This grounding informs the current utterance without projecting future turns."
    )


def _fallback_user_b_context(row: Dict[str, Any]) -> str:
    n = len(row.get("prev_context", []) or [])
    return (
        f"User B (target-side speaker): The conversation has {n} prior turns "
        "shaping the relational context, formality level, and communicative stance. "
        "This grounding informs interpretation of the current turn without projecting future replies."
    )


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


def _reference_alignment_note(labse_sim: Optional[float]) -> str:
    """Return a prompt note calibrating trust in the reference translation."""
    if labse_sim is None:
        return ""
    if labse_sim >= 0.7:
        label = "High"
        advice = "reference translation is reliable."
    elif labse_sim >= 0.5:
        label = "Moderate"
        advice = "use the reference translation with some caution."
    elif labse_sim >= 0.3:
        label = "Low"
        advice = (
            "reference translation may be partially misaligned; "
            "weight it less heavily and rely more on the source text."
        )
    else:
        label = "Very low"
        advice = (
            "reference translation is likely misaligned or loosely parallel; "
            "treat it as unreliable and derive your analysis primarily from the source text."
        )
    return (
        f"Reference alignment confidence: {label} "
        f"(LaBSE cosine similarity: {labse_sim:.2f}) — {advice}"
    )


def _render_prompt(
    row: Dict[str, Any],
    is_fwd: bool,
    src_lang_name: str,
    tgt_lang_name: str,
    src_lang_code: str,
    tgt_lang_code: str,
    cultural_context: Optional[str] = None,
) -> str:
    if is_fwd:
        source_text = str(row.get("source_text", "")).strip()
        reference_target_text = str(row.get("target_text", "")).strip()
    else:
        source_text = str(row.get("target_text", "")).strip()
        reference_target_text = str(row.get("source_text", "")).strip()

    prev_lines = _iter_context(row.get("prev_context", []))
    prev_block = "\n".join(prev_lines) if prev_lines else "(none)"
    reason_tags = _extract_reason_tags(row)
    reason_tags_block = _expand_reason_hints(reason_tags)
    context_digest = _build_context_digest(row, is_fwd)

    if cultural_context:
        cultural_context_block = (
            "Pair-specific cultural context (anchor all layer_3 criteria and cultural failure points here):\n"
            + cultural_context
            + "\n\n"
        )
    else:
        cultural_context_block = ""

    labse_sim = row.get("labse_similarity")
    alignment_note = _reference_alignment_note(labse_sim)

    return PROMPT_TEMPLATE.format(
        source_language=src_lang_name,
        source_language_code=src_lang_code,
        target_language=tgt_lang_name,
        target_language_code=tgt_lang_code,
        cultural_context_block=cultural_context_block,
        source_text=source_text,
        reference_target_text=reference_target_text,
        reference_alignment_note=alignment_note,
        reason_tags_block=reason_tags_block,
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
    return json.loads(text[start: end + 1])


def _llm_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model_name, safe='')}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )
    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    if seed is not None:
        gen_config["seed"] = seed
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
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


def _contains_current_source_text(
    context_text: str, row: Dict[str, Any], is_fwd: bool
) -> bool:
    current_source = _as_string(
        row.get("source_text" if is_fwd else "target_text", "")
    )
    if not current_source:
        return False
    src_norm = re.sub(r"\W+", "", current_source.lower())
    ctx_norm = re.sub(r"\W+", "", (context_text or "").lower())
    if len(src_norm) < 16:
        return False
    return src_norm[:16] in ctx_norm or src_norm[-16:] in ctx_norm


def _contains_english_template_language(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pat, lowered) for pat in ENGLISH_TEMPLATE_PATTERNS)


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
    is_fwd: bool,
) -> Dict[str, Any]:
    fallback_window = _fallback_context_window_summary(row)
    fallback_conversation = _fallback_conversation_context(row)
    fallback_user_a = _fallback_user_a_context(row)
    fallback_user_b = _fallback_user_b_context(row)

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

    # Ensure "User A" / "User B" markers are always present
    if "User A" not in user_a_body:
        user_a_body = "User A (source-side speaker). " + user_a_body
    if "User B" not in user_b_body:
        user_b_body = "User B (target-side speaker). " + user_b_body

    generated["user_a_context"] = user_a_body
    generated["user_b_context"] = user_b_body

    generated["mandatory_cultural_constraints"] = _dedupe_keep_order(
        _as_string_list(generated.get("mandatory_cultural_constraints"))
    )

    checklist = generated.get("checklist")
    if not isinstance(checklist, dict):
        checklist = {}

    l1 = _dedupe_keep_order(_as_string_list(checklist.get("layer_1_semantic_core")))
    l2 = _dedupe_keep_order(_as_string_list(checklist.get("layer_2_pragmatic_function")))
    l3 = _dedupe_keep_order(_as_string_list(checklist.get("layer_3_cultural_social_constraints")))

    if not l1:
        l1 = ["Does the translation preserve the core factual meaning of the source utterance?"]
    if not l2:
        l2.append("Does the translation preserve the speaker's communicative intent?")
    if not l3:
        l3.append("Does the translation preserve required register, politeness, and social stance?")

    if not _checklist_has_keyword(l2, ["context", "preced", "follow", "coher", "turn", "dialogue flow"]):
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
    generated["verification_prompt"] = _normalize_verification_prompt(
        _compose_verification_prompt(generated["checklist"])
    )
    return generated


def _validate_generated_sample(
    generated: Dict[str, Any],
    row: Optional[Dict[str, Any]] = None,
    is_fwd: Optional[bool] = None,
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

    ua = _as_string(generated.get("user_a_context", ""))
    ub = _as_string(generated.get("user_b_context", ""))

    if _contains_hinting_language(ua):
        errors.append("user_a_context contains hinting language")
    if _contains_hinting_language(ub):
        errors.append("user_b_context contains hinting language")
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

    if row is not None and is_fwd is not None:
        if _contains_current_source_text(cc, row, is_fwd):
            errors.append("conversation_context should not include current source utterance")
        if _contains_current_source_text(cws, row, is_fwd):
            errors.append("context_window_summary should not include current source utterance")
        if _contains_current_source_text(ua, row, is_fwd):
            errors.append("user_a_context should not include current source utterance")
        if _contains_current_source_text(ub, row, is_fwd):
            errors.append("user_b_context should not include current source utterance")

    checklist = generated.get("checklist")
    if not isinstance(checklist, dict):
        checklist = {}

    l1 = len(_as_string_list(checklist.get("layer_1_semantic_core")))
    l2 = len(_as_string_list(checklist.get("layer_2_pragmatic_function")))
    l3 = len(_as_string_list(checklist.get("layer_3_cultural_social_constraints")))

    if l1 < 1:
        errors.append("layer_1_semantic_core must have at least 1 item")
    if l2 < 1:
        errors.append("layer_2_pragmatic_function must have at least 1 item")
    if l3 < 1:
        errors.append("layer_3_cultural_social_constraints must have at least 1 item")
    if not (l3 >= l2 >= l1):
        errors.append("checklist count priority must satisfy layer_3 >= layer_2 >= layer_1")

    l2_items = _as_string_list(checklist.get("layer_2_pragmatic_function"))
    if not _checklist_has_keyword(l2_items, ["context", "coher", "preced", "follow", "turn"]):
        errors.append("layer_2_pragmatic_function should include context-coherence criterion")

    return errors


def _build_output_record(
    row: Dict[str, Any],
    generated: Dict[str, Any],
    src_code: str,
    tgt_code: str,
    is_fwd: bool,
    global_index: int,
    lang_pair: str,
    consistency_run_id: int = 0,
    consistency_temperature: float = 0.0,
) -> Dict[str, Any]:
    if is_fwd:
        source_text = str(row.get("source_text", "")).strip()
        reference_target_text = str(row.get("target_text", "")).strip()
        out_src_code, out_tgt_code = src_code, tgt_code
    else:
        source_text = str(row.get("target_text", "")).strip()
        reference_target_text = str(row.get("source_text", "")).strip()
        out_src_code, out_tgt_code = tgt_code, src_code

    direction_label = f"{out_src_code}_{out_tgt_code}"
    reason_tags = _extract_reason_tags(row)

    record: Dict[str, Any] = {
        "seed_file": row.get("segment_file", ""),
        "seed_split": f"opensubtitles_{lang_pair}",
        "seed_row_id": global_index,
        "Category": "MAPS-Dialogue-Pragmatics",
        "Source Concept (Original Source Language)": generated["semantic_core"],
        "Verification Goal (Target Receiver)": generated["semantic_core"],
        'Linguistic/Cultural "Trap"': " | ".join(generated["mandatory_cultural_constraints"]),
        "source_language": LANGS[out_src_code]["name"],
        "target_language": LANGS[out_tgt_code]["name"],
        "source_language_code": out_src_code,
        "target_language_code": out_tgt_code,
        "direction": direction_label,
        "lang_pair": lang_pair,
        "segment_file": row.get("segment_file", ""),
        "segment_id": row.get("segment_id", ""),
        "source_text": source_text,
        "reference_target_text": reference_target_text,
        "pragmatic_analysis": generated["pragmatic_analysis"],
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
    if consistency_run_id > 0:
        record["consistency_run_id"] = consistency_run_id
        record["consistency_temperature"] = consistency_temperature
    return record


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _row_direction_flags(row: Dict[str, Any], force_directions: Optional[List[str]]) -> List[bool]:
    """Return which is_fwd values to process for this row.

    By default, derived from the row's 'direction' field (fwd/rev/both).
    If force_directions is given, it overrides the per-row field.
    """
    if force_directions is not None:
        flags: List[bool] = []
        if "fwd" in force_directions:
            flags.append(True)
        if "rev" in force_directions:
            flags.append(False)
        return flags
    row_dir = str(row.get("direction") or "both").lower()
    if row_dir == "fwd":
        return [True]
    if row_dir == "rev":
        return [False]
    return [True, False]  # "both" or unknown


def _load_existing_keys_with_runs(path: Path) -> Set[Tuple[str, str, str, int]]:
    """Keys are (segment_file, segment_id, direction, consistency_run_id)."""
    keys: Set[Tuple[str, str, str, int]] = set()
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
            direction = str(obj.get("direction", ""))
            run_id = int(obj.get("consistency_run_id", 0))
            keys.add((seg_file, seg_id, direction, run_id))
    return keys


def _consolidate_multirun_output(
    output_jsonl: Path,
    dedup_threshold: float,
    consolidated_output: Path,
) -> int:
    """Group the multi-run raw records by (segment_id, direction) and merge each
    group into one canonical record via consolidate_consistency_runs.consolidate_group()
    (exact-dedup + LaBSE semantic dedup of checklist items). Always invoked automatically
    after a multi-run (self-consistency) augmentation pass so dedup can't be silently
    skipped by forgetting to run consolidate_consistency_runs.py separately."""
    records = _read_jsonl(output_jsonl)
    safe_records = [record for record in records if not classify_value(record).blocked]
    content_safety_dropped = len(records) - len(safe_records)
    if content_safety_dropped:
        print(
            f"content_safety_dropped_during_consolidation={content_safety_dropped}",
            flush=True,
        )
    records = safe_records
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r.get("segment_id", ""), r.get("direction", ""))].append(r)

    incomplete = [(k, len(v)) for k, v in groups.items() if len(v) < 2]
    if incomplete:
        print(
            f"warning: {len(incomplete)} group(s) with <2 consistency runs "
            f"(consolidating anyway, dedup within a single run only)",
            flush=True,
        )

    merged = [consolidate_group(runs, dedup_threshold) for runs in groups.values()]
    merged.sort(key=lambda r: (str(r.get("segment_id", "")), r.get("direction", "")))

    consolidated_output.parent.mkdir(parents=True, exist_ok=True)
    with consolidated_output.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(merged)


def run_augmentation(
    input_jsonl: Path,
    output_jsonl: Path,
    lang_pair: str,
    force_directions: Optional[List[str]],
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
    use_cultural_context: bool = True,
    consistency_runs: int = 1,
    consistency_temps: Optional[List[float]] = None,
    concurrency: int = 8,
    base_seed: Optional[int] = None,
    consolidate: bool = True,
    dedup_threshold: float = DEDUP_THRESHOLD,
    consolidated_output: Optional[Path] = None,
) -> None:
    if lang_pair not in PAIR_LANGS:
        raise ValueError(f"Unknown lang_pair '{lang_pair}'. Known: {sorted(PAIR_LANGS)}")
    src_code, tgt_code = PAIR_LANGS[lang_pair]

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or environment.")

    cultural_context = CULTURAL_CONTEXT.get(lang_pair) if use_cultural_context else None

    # Resolve per-run temperatures
    n_runs = max(1, consistency_runs)
    if consistency_temps and len(consistency_temps) >= n_runs:
        run_temps = consistency_temps[:n_runs]
    else:
        if n_runs == 1:
            run_temps = [temperature]
        elif n_runs == 2:
            run_temps = [temperature, 0.8]
        else:
            step = (0.9 - temperature) / (n_runs - 1) if temperature < 0.9 else 0.0
            run_temps = [round(temperature + i * step, 2) for i in range(n_runs)]

    rows = _read_jsonl(input_jsonl)
    if start_index > 0:
        rows = rows[start_index:]
    if max_rows > 0:
        rows = rows[:max_rows]
    input_rows_before_safety = len(rows)
    rows = [row for row in rows if not classify_value(row).blocked]
    input_content_safety_dropped = input_rows_before_safety - len(rows)
    if input_content_safety_dropped:
        print(
            f"input_content_safety_dropped={input_content_safety_dropped}",
            flush=True,
        )
    if not rows:
        raise RuntimeError("No safe rows to process after applying start/max limits.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    multi_run = n_runs > 1
    existing_run_keys: Set[Tuple[str, str, str, int]] = set()
    existing_simple_keys: Set[Tuple[str, str, str]] = set()
    if append:
        if multi_run:
            existing_run_keys = _load_existing_keys_with_runs(output_jsonl)
        else:
            existing_simple_keys = {(a, b, c) for a, b, c, _ in _load_existing_keys_with_runs(output_jsonl)}
    if not append:
        output_jsonl.write_text("", encoding="utf-8")

    # Pre-build task list so we can submit all to the thread pool at once.
    all_tasks: List[Dict[str, Any]] = []
    skipped_existing = 0
    for row_idx, row in enumerate(rows, start=1):
        direction_flags = _row_direction_flags(row, force_directions)
        for is_fwd in direction_flags:
            global_index = input_index_offset + row_idx
            out_src, out_tgt = (src_code, tgt_code) if is_fwd else (tgt_code, src_code)
            direction_label = f"{out_src}_{out_tgt}"
            simple_key = (str(row.get("segment_file", "")), str(row.get("segment_id", "")), direction_label)

            for run_idx in range(1, n_runs + 1):
                run_key = (*simple_key, run_idx if multi_run else 0)
                if multi_run and run_key in existing_run_keys:
                    skipped_existing += 1
                    continue
                if not multi_run and simple_key in existing_simple_keys:
                    skipped_existing += 1
                    continue
                all_tasks.append({
                    "row": row,
                    "row_idx": row_idx,
                    "is_fwd": is_fwd,
                    "run_idx": run_idx,
                    "run_temp": run_temps[run_idx - 1],
                    "run_key": run_key,
                    "simple_key": simple_key,
                    "direction_label": direction_label,
                    "global_index": global_index,
                    "out_src": out_src,
                    "out_tgt": out_tgt,
                })

    total_tasks = len(all_tasks)
    print(
        f"tasks_to_process={total_tasks} skipped_existing={skipped_existing} concurrency={concurrency}",
        flush=True,
    )

    stats: Dict[str, int] = {"written": 0, "failed": 0, "completed": 0}
    write_lock = threading.Lock()

    def _execute_task(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = task["row"]
        is_fwd = task["is_fwd"]
        run_idx = task["run_idx"]
        run_temp = task["run_temp"]
        out_src = task["out_src"]
        out_tgt = task["out_tgt"]
        direction_label = task["direction_label"]
        row_idx = task["row_idx"]

        prompt = _render_prompt(
            row=row,
            is_fwd=is_fwd,
            src_lang_name=LANGS[out_src]["name"],
            tgt_lang_name=LANGS[out_tgt]["name"],
            src_lang_code=out_src,
            tgt_lang_code=out_tgt,
            cultural_context=cultural_context,
        )

        generated: Optional[Dict[str, Any]] = None
        for attempt in range(1, max_retries + 1):
            try:
                run_seed = (base_seed + run_idx - 1) if base_seed is not None else None
                generated = _llm_generate(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=run_temp,
                    max_output_tokens=max_output_tokens,
                    request_timeout_s=request_timeout_s,
                    seed=run_seed,
                )
                generated = _repair_generated_sample(generated=generated, row=row, is_fwd=is_fwd)
                errors = _validate_generated_sample(generated, row=row, is_fwd=is_fwd)
                hard_failures = [
                    e for e in errors
                    if "empty" in e or "at least" in e or "priority" in e or "should include" in e
                ]
                if hard_failures:
                    raise ValueError("; ".join(hard_failures))
                safety = classify_value(generated)
                if safety.blocked:
                    raise ValueError(safety.reason)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    print(
                        f"failed row={row_idx} dir={direction_label} run={run_idx} "
                        f"after {max_retries} attempts: {exc}",
                        flush=True,
                    )
                    generated = None
                    break
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                # Use longer backoff for rate limit errors so threads back off together.
                wait_s = max(retry_backoff_s * attempt * 4, 20.0) if is_rate_limit else retry_backoff_s * attempt
                print(
                    f"retry row={row_idx} dir={direction_label} run={run_idx} "
                    f"attempt={attempt}/{max_retries} wait={wait_s:.1f}s error={exc}",
                    flush=True,
                )
                time.sleep(wait_s)

        if generated is None:
            return None

        if sleep_s > 0:
            time.sleep(sleep_s)

        record = _build_output_record(
            row=row,
            generated=generated,
            src_code=src_code,
            tgt_code=tgt_code,
            is_fwd=is_fwd,
            global_index=task["global_index"],
            lang_pair=lang_pair,
            consistency_run_id=run_idx if multi_run else 0,
            consistency_temperature=run_temp if multi_run else 0.0,
        )
        safety = classify_value(record)
        if safety.blocked:
            print(
                f"blocked generated row={row_idx} dir={direction_label} run={run_idx}: "
                f"{safety.reason}",
                flush=True,
            )
            return None
        return record

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {executor.submit(_execute_task, t): t for t in all_tasks}
        for future in as_completed(future_to_task):
            try:
                record = future.result()
            except Exception as exc:
                task = future_to_task[future]
                print(f"unexpected error task row={task['row_idx']}: {exc}", flush=True)
                record = None

            with write_lock:
                stats["completed"] += 1
                if record is not None:
                    _append_jsonl(output_jsonl, record)
                    stats["written"] += 1
                else:
                    stats["failed"] += 1
                cnt = stats["completed"]
                if cnt % 20 == 0 or cnt == total_tasks:
                    print(
                        f"completed={cnt}/{total_tasks} written={stats['written']} "
                        f"failed={stats['failed']}",
                        flush=True,
                    )

    print("done", flush=True)
    print(f"input_rows={len(rows)}", flush=True)
    print(f"total_tasks={total_tasks}", flush=True)
    print(f"written={stats['written']}", flush=True)
    print(f"skipped_existing={skipped_existing}", flush=True)
    print(f"failed_rows={stats['failed']}", flush=True)
    print(f"output={output_jsonl}", flush=True)

    # Finalized checklist-gen setup: dedup is ALWAYS applied, not just when
    # --consistency-runs > 1 — a single run still gets semantic-deduped
    # (strips any within-run near-duplicate criteria the model itself wrote).
    if consolidate:
        out_path = consolidated_output or (output_jsonl.parent / "consolidated.jsonl")
        n_consolidated = _consolidate_multirun_output(
            output_jsonl=output_jsonl,
            dedup_threshold=dedup_threshold,
            consolidated_output=out_path,
        )
        print(
            f"consolidated={n_consolidated} (threshold={dedup_threshold}) -> {out_path}",
            flush=True,
        )
    else:
        print(
            "warning: --no-consolidate set; raw output was NOT deduplicated "
            "(checklist may contain near-duplicate criteria)",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment OpenSubtitles windows into MAPS-like data for any language pair."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input JSONL from 'windows' command.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL path.")
    parser.add_argument("--lang-pair", type=str, required=True, choices=sorted(PAIR_LANGS),
                        help="Language pair (e.g. ar-bn, id-ko).")
    parser.add_argument("--directions", nargs="+", default=None,
                        choices=["fwd", "rev"],
                        help="Override direction per row: fwd, rev, or both. "
                             "Default: use each row's own direction field (fwd/rev/both).")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows to process after start-index (0=all).")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start offset in input rows.")
    parser.add_argument("--append", action="store_true",
                        help="Append to output; skip rows already present.")
    parser.add_argument("--input-index-offset", type=int, default=0,
                        help="Offset for stable global row numbering in chunked runs.")
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-s", type=float, default=3.0)
    parser.add_argument("--no-cultural-context", action="store_true",
                        help="Disable pair-specific cultural context injection.")
    parser.add_argument("--consistency-runs", type=int, default=1, metavar="N",
                        help="Number of self-consistency generation runs per segment (default 1). "
                             "When N>1, each run is stored with a consistency_run_id field.")
    parser.add_argument("--consistency-temps", type=float, nargs="+", default=None, metavar="T",
                        help="Temperatures for each consistency run (space-separated). "
                             "If not given, evenly spaced from --temperature to 0.9.")
    parser.add_argument("--concurrency", type=int, default=8, metavar="N",
                        help="Number of parallel API calls (default: 8). "
                             "Raise to saturate your rate limit; 429s are retried with backoff.")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Base seed for Gemini generationConfig. Run k uses seed+k-1, "
                             "ensuring independent samples across consistency runs. Best-effort.")
    parser.add_argument("--no-consolidate", action="store_true",
                        help="Skip automatic consolidation/dedup entirely (finalized setup: dedup "
                             "always runs, even for --consistency-runs 1, to strip within-run "
                             "near-duplicates). Only use for debugging raw per-run output.")
    parser.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD, metavar="T",
                        help=f"LaBSE cosine similarity threshold for semantic dedup during "
                             f"consolidation (default: {DEDUP_THRESHOLD}). Applied regardless of "
                             f"--consistency-runs.")
    parser.add_argument("--consolidated-output", type=Path, default=None, metavar="PATH",
                        help="Path for the consolidated (deduped) output. "
                             "Default: consolidated.jsonl next to --output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_augmentation(
        input_jsonl=args.input,
        output_jsonl=args.output,
        lang_pair=args.lang_pair,
        force_directions=args.directions,
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
        use_cultural_context=not args.no_cultural_context,
        consistency_runs=args.consistency_runs,
        consistency_temps=args.consistency_temps,
        concurrency=args.concurrency,
        base_seed=args.seed,
        consolidate=not args.no_consolidate,
        dedup_threshold=args.dedup_threshold,
        consolidated_output=args.consolidated_output,
    )


if __name__ == "__main__":
    main()
