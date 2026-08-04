"""Tests for the multi-turn expansion (docs/multiturn_expansion_plan.md).

Mirrors tests/test_basic.py's plain run_all_tests()/sys.exit style rather than
pytest, per repo convention. Mock providers only — no LLM spend. Test
functions are added incrementally as each plan step lands; see the step
comments below.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from interpreter_agent_eval.pipeline.batch import FakeBatchClient  # noqa: E402
from interpreter_agent_eval.pipeline.io import read_jsonl  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn import checklist_gen as cg  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn.checklist_gen import ChecklistItem  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn import operations as mt_ops  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn import stages as mt_stages  # noqa: E402

import generate_multiturn_scenarios as gen  # noqa: E402


class MockChecklistProvider:
    """Returns a fixed, valid structured checklist response."""

    def __init__(self, items):
        self.items = items
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt
        return json.dumps({"items": self.items})


class MockStructuredFailThenPlainProvider:
    """Fails when asked for structured output, succeeds on plain generation."""

    def __init__(self, items):
        self.items = items
        self.structured_calls = 0
        self.plain_calls = 0

    def generate(self, prompt, **kwargs):
        if "response_schema" in kwargs:
            self.structured_calls += 1
            raise RuntimeError("structured output not supported by this mock")
        self.plain_calls += 1
        return json.dumps({"items": self.items})


# ---------------------------------------------------------------------------
# Step 1: checklist_gen.py
# ---------------------------------------------------------------------------
def test_taxonomy_loading():
    print("Testing function taxonomy loading...")

    for lang in ("ind", "kor", "arb", "ben"):
        assert cg.taxonomy_available(lang), f"expected taxonomy file for {lang}"

    cg.assert_taxonomies_available()  # must not raise for the 4 study languages

    entries = cg.load_function_taxonomy("kor")
    assert len(entries) == 53
    assert {"function_id", "layer", "label"} <= set(entries[0].keys())
    layers = {e["layer"] for e in entries}
    assert {"layer_1", "layer_2", "layer_3"} <= layers

    # cached: second call must return the exact same object, not re-read the file
    assert cg.load_function_taxonomy("kor") is entries

    listing = cg.format_taxonomy_listing(entries[:2])
    assert listing.count("\n") == 1
    assert entries[0]["function_id"] in listing

    try:
        cg.load_function_taxonomy("xyz")
        assert False, "expected FileNotFoundError for unknown language"
    except FileNotFoundError:
        pass

    print("  ✓ Taxonomy loading tests passed")


def test_checklist_generation_and_cap_enforcement():
    print("Testing checklist generation + cap enforcement...")

    good_items = [
        {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation preserve the honorific used toward the elder?"},
        {"function_id": "L2_f2", "layer": "layer_2", "text": "Does the translation convey the speaker is making a polite request?"},
        {"function_id": "L1_f6", "layer": "layer_1", "text": "Does the translation preserve the claim that the meeting is tomorrow?"},
    ]
    provider = MockChecklistProvider(good_items)
    items = cg.generate_turn_checklist(provider, "kor", "Two colleagues discuss a meeting.", "A", "Besok kita rapat, kan?")
    assert len(items) == 3
    assert all(isinstance(it, ChecklistItem) for it in items)
    assert provider.call_count == 1
    assert cg.validate_checklist_items(items, cg.TURN_HARD_CEILING) == []

    # the prompt states no upper limit: a genuinely dense turn producing more
    # than the old typical ceiling (7) but under the runaway guard (14) must
    # survive untouched. Text must be genuinely distinct (not templated) since
    # dedup is now always applied; layer counts must satisfy both the floor
    # (>=1 each) and the priority rule (layer_3 >= layer_2 >= layer_1).
    nine_items = [
        {"function_id": "L3_f0", "layer": "layer_3", "text": "Does the translation preserve the formal honorific register expected between strangers?"},
        {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation avoid the direct refusal that would cause loss of face here?"},
        {"function_id": "L3_f2", "layer": "layer_3", "text": "Does the translation reflect the culturally expected indirectness for this request?"},
        {"function_id": "L3_f3", "layer": "layer_3", "text": "Does the translation preserve the deference owed to an elder speaker?"},
        {"function_id": "L2_f0", "layer": "layer_2", "text": "Does the translation convey the speaker is proposing a joint plan rather than issuing a command?"},
        {"function_id": "L2_f1", "layer": "layer_2", "text": "Does the translation preserve the hedged, tentative framing of the suggestion?"},
        {"function_id": "L2_f2", "layer": "layer_2", "text": "Does the translation convey genuine uncertainty about the listener's availability?"},
        {"function_id": "L1_f0", "layer": "layer_1", "text": "Does the translation accurately convey that the meeting is scheduled for tomorrow?"},
        {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation accurately convey the specific location mentioned for the meeting?"},
    ]
    provider_dense = MockChecklistProvider(nine_items)
    items_dense = cg.generate_turn_checklist(provider_dense, "kor", "ctx", "A", "text")
    assert len(items_dense) == 9
    assert cg.validate_checklist_items(items_dense, cg.TURN_HARD_CEILING) == []
    note_dense = cg.checklist_count_note(items_dense, cg.TURN_ITEM_CAP)
    assert note_dense is not None and "above" in note_dense

    # hard ceiling is instrumentation-only, no longer truncates (TODO.md:
    # "hard ceilings 14/12 no longer truncate") — 20 genuinely distinct items
    # for a turn survive untouched, just noted in checklist_count_note().
    # Text must be lexically varied (not a fill-in-the-blank template), since
    # LaBSE dedup is aggressive on near-templated text differing only by a number.
    l1_facts = [
        "the meeting is scheduled for tomorrow", "the venue changed to the north office",
        "the deadline was moved to Friday", "the client already approved the budget",
        "the shipment left the warehouse yesterday", "the contract expires next month",
    ]
    l2_functions = [
        "is making a polite request rather than a demand", "is expressing genuine hesitation about the plan",
        "is offering a face-saving way out of the disagreement", "is signaling gratitude rather than mere acknowledgment",
        "is hedging the suggestion to avoid seeming presumptuous", "is issuing a warning disguised as friendly advice",
        "is subtly declining without a direct refusal",
    ]
    l3_constraints = [
        "the honorific register expected between a subordinate and a superior",
        "the indirectness culturally required for a face-threatening refusal",
        "the kinship term that signals the speaker's closeness to the listener",
        "the formality level expected in a first-time business introduction",
        "the deference owed when addressing a much older relative",
        "the humility marker expected when accepting praise",
        "the avoidance of a taboo topic considered impolite to name directly",
    ]
    too_many = [
        {"function_id": f"L1_f{i}", "layer": "layer_1", "text": f"Does the translation accurately convey that {fact}?"}
        for i, fact in enumerate(l1_facts)
    ] + [
        {"function_id": f"L2_f{i}", "layer": "layer_2", "text": f"Does the translation preserve that the speaker {fn}?"}
        for i, fn in enumerate(l2_functions)
    ] + [
        {"function_id": f"L3_f{i}", "layer": "layer_3", "text": f"Does the translation reflect {constraint}?"}
        for i, constraint in enumerate(l3_constraints)
    ]
    # dedup may legitimately collapse a few near-duplicates even with varied
    # wording (LaBSE catching genuine structural/semantic overlap) — the
    # point of this test is "no truncation to the hard ceiling", not an exact
    # survivor count, so assert well above the ceiling rather than ==.
    provider2 = MockChecklistProvider(too_many)
    items2 = cg.generate_turn_checklist(provider2, "kor", "ctx", "A", "text")
    assert len(items2) > cg.TURN_HARD_CEILING
    note2 = cg.checklist_count_note(items2, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
    assert note2 is not None and "EXCEEDS hard ceiling" in note2

    # conversation-level: same, its own hard ceiling (12) is also instrumentation-only
    provider3 = MockChecklistProvider(too_many)
    conv_items = cg.generate_conversation_checklist(provider3, "kor", "ctx", "transcript basis")
    assert len(conv_items) > cg.CONVERSATION_HARD_CEILING

    # empty items fail validation
    assert cg.validate_checklist_items([], cg.TURN_HARD_CEILING) != []

    # structured-call failure falls back to plain generation (mirrors
    # pipeline.operations.parse_judge_evaluation's fallback convention)
    provider4 = MockStructuredFailThenPlainProvider(good_items)
    items4 = cg.generate_turn_checklist(provider4, "ind", "ctx", "B", "text")
    assert len(items4) == 3
    assert provider4.structured_calls == 1
    assert provider4.plain_calls == 1

    # de-anchoring fix: a below-advisory-typical-max count is valid, not an
    # error, AS LONG AS the per-layer floor (>=1 each) is met — the smallest
    # genuinely valid checklist is 3 items (1 per layer); missing a whole
    # layer is still a hard violation (that's the "layer_N must have at
    # least 1 item" rule, unchanged and enforced regardless of total count).
    minimal_valid = [ChecklistItem(**it) for it in good_items]  # 1 per layer, 3 total
    assert cg.validate_checklist_items(minimal_valid, cg.TURN_HARD_CEILING) == []
    assert cg.checklist_count_note(minimal_valid, cg.TURN_ITEM_CAP) is None  # 3 is within (min_n=3, typical_max=7)

    missing_a_layer = [ChecklistItem(**it) for it in good_items[:2]]  # drops layer_1
    assert cg.validate_checklist_items(missing_a_layer, cg.TURN_HARD_CEILING) != []

    # priority-rule validation: hard ceilings no longer truncate (so there's
    # no "does truncation preserve layer_3 priority" to check anymore — see
    # _enforce_cap), but a checklist with more layer_1 than layer_3 items
    # must still be caught by validate_checklist_items as a priority
    # violation, regardless of how many total items there are.
    priority_probe = [
        {"function_id": "L1_f0", "layer": "layer_1", "text": "Does the translation preserve the specific number mentioned?"},
        {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation preserve the location named in the request?"},
        {"function_id": "L1_f2", "layer": "layer_1", "text": "Does the translation preserve the exact time referenced?"},
        {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation reflect the honorific register expected here?"},
    ]
    provider5 = MockChecklistProvider(priority_probe)
    items5 = cg.generate_turn_checklist(provider5, "kor", "ctx", "A", "text")
    assert len(items5) == 4  # distinct enough to all survive dedup
    errs5 = cg.validate_checklist_items(items5, cg.TURN_HARD_CEILING)
    assert any("priority" in e for e in errs5)

    # cultural-context threading: when given, the pair's asymmetry paragraph
    # must land in the actual prompt sent to the provider.
    provider6 = MockChecklistProvider(good_items)
    cultural_context = cg.get_cultural_context("arb", "kor")
    assert cultural_context, "expected an authored ar-ko cultural-context paragraph"
    cg.generate_turn_checklist(
        provider6, "kor", "ctx", "A", "text", cultural_context=cultural_context
    )
    assert cultural_context in provider6.last_prompt

    # no cultural_context passed -> no block, no crash (unknown/omitted pair)
    provider7 = MockChecklistProvider(good_items)
    cg.generate_turn_checklist(provider7, "kor", "ctx", "A", "text")
    assert cg.get_cultural_context("xyz", "kor") is None

    print("  ✓ Checklist generation + cap enforcement tests passed")


def test_verification_prompt_composition():
    print("Testing verification-prompt composition (L3->L2->L1)...")

    items = [
        ChecklistItem(function_id="L1_f1", layer="layer_1", text="semantic item"),
        ChecklistItem(function_id="L3_f1", layer="layer_3", text="cultural item"),
        ChecklistItem(function_id="L2_f1", layer="layer_2", text="pragmatic item"),
        ChecklistItem(function_id="L3_f2", layer="layer_3", text="second cultural item"),
    ]
    prompt = cg.compose_verification_prompt(items)
    lines = prompt.splitlines()
    assert lines == [
        "1. cultural item",
        "2. second cultural item",
        "3. pragmatic item",
        "4. semantic item",
    ]

    # ungrounded (no taxonomy) items still compose fine
    ungrounded = [ChecklistItem(function_id=None, layer="layer_2", text="fallback item")]
    assert cg.compose_verification_prompt(ungrounded) == "1. fallback item"

    print("  ✓ Verification-prompt composition tests passed")


# ---------------------------------------------------------------------------
# Step 1: generate_multiturn_scenarios.py validators (no spend — hand-written
# good/bad dicts only)
# ---------------------------------------------------------------------------
class _FakeTurn:
    def __init__(self, turn_index, speaker, text):
        self.turn_index = turn_index
        self.speaker = speaker
        self.text = text


def test_validate_num_turns():
    print("Testing num_turns validation...")

    assert gen.validate_num_turns(6) == []
    assert gen.validate_num_turns(4) == []
    assert gen.validate_num_turns(8) == []
    assert gen.validate_num_turns(3) != []
    assert gen.validate_num_turns(9) != []

    print("  ✓ num_turns validation tests passed")


def test_validate_alternation():
    print("Testing A/B alternation validation...")

    good = [_FakeTurn(0, "A", "hi"), _FakeTurn(1, "B", "hello"), _FakeTurn(2, "A", "how are you")]
    assert gen.validate_alternation(good) == []

    # starts with B instead of A
    bad_start = [_FakeTurn(0, "B", "hi"), _FakeTurn(1, "A", "hello")]
    assert gen.validate_alternation(bad_start) != []

    # same speaker twice in a row
    bad_repeat = [_FakeTurn(0, "A", "hi"), _FakeTurn(1, "A", "hello")]
    assert gen.validate_alternation(bad_repeat) != []

    # non-sequential turn_index
    bad_index = [_FakeTurn(0, "A", "hi"), _FakeTurn(2, "B", "hello")]
    assert gen.validate_alternation(bad_index) != []

    # empty text
    bad_text = [_FakeTurn(0, "A", ""), _FakeTurn(1, "B", "hello")]
    assert gen.validate_alternation(bad_text) != []

    # empty list
    assert gen.validate_alternation([]) != []

    print("  ✓ A/B alternation validation tests passed")


def test_generate_one_scripted_scenario_mock():
    print("Testing generate_one_scripted_scenario end-to-end (mock providers)...")

    transcript_json = json.dumps(
        {
            "conversation_context": "Two colleagues discuss a delayed project.",
            "user_a_context": "Anda seorang staf yang menunda laporan.",
            "user_b_context": "당신은 팀장입니다.",
            "turns": [
                {"turn_index": 0, "speaker": "A", "text": "Laporannya belum selesai, maaf."},
                {"turn_index": 1, "speaker": "B", "text": "괜찮아요, 언제까지 가능해요?"},
                {"turn_index": 2, "speaker": "A", "text": "Mungkin dua hari lagi."},
                {"turn_index": 3, "speaker": "B", "text": "알겠습니다, 기다릴게요."},
            ],
        }
    )
    checklist_items = [
        {"function_id": "L2_f1", "layer": "layer_2", "text": "Does the translation convey the speaker is apologizing for a delay?"},
        {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation preserve the claim that the report isn't finished?"},
        {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation use a register appropriate for addressing a team lead?"},
    ]

    class TranscriptProvider:
        model_name = "mock-transcript"

        def generate(self, prompt, **kwargs):
            return transcript_json

    class ChecklistProvider:
        model_name = "mock-checklist"

        def generate(self, prompt, **kwargs):
            return json.dumps({"items": checklist_items})

    scenario = gen.generate_one_scripted_scenario(
        TranscriptProvider(),
        ChecklistProvider(),
        prompt="unused in mock",
        lang_a="ind",
        lang_b="kor",
        num_turns=4,
        glotlid_model=None,
        use_grounding=True,
        seed_file="test_proverbs.xlsx",
        seed_row_id=1,
        category="MAPS-Proverb-Pragmatics-MT",
        conversation_id="indkor_mts_0001",
    )

    assert scenario.conversation_id == "indkor_mts_0001"
    assert scenario.lang_a == "ind" and scenario.lang_b == "kor"
    assert len(scenario.turns) == 4
    assert [t.speaker for t in scenario.turns] == ["A", "B", "A", "B"]
    assert all(t.verification_prompt.startswith("1.") for t in scenario.turns)
    assert len(scenario.conversation_checklist_items) == 3
    assert scenario.conversation_verification_prompt.startswith("1.")

    dumped = scenario.model_dump()
    assert dumped["turns"][0]["checklist_items"][0]["function_id"] == "L2_f1"

    print("  ✓ generate_one_scripted_scenario mock end-to-end test passed")


def test_generate_one_dynamic_seed_mock():
    print("Testing generate_one_dynamic_seed end-to-end (mock providers, guided + free)...")

    guided_json = json.dumps(
        {
            "conversation_context": "Two friends plan a weekend trip.",
            "user_a_context": "Anda ingin mengajak teman lama jalan-jalan.",
            "user_b_context": "당신은 바빠 일정이 있습니다.",
            "intent_outline": [
                {"turn_index": 0, "speaker": "A", "intent": "propose a weekend trip"},
                {"turn_index": 1, "speaker": "B", "intent": "politely decline, citing work"},
                {"turn_index": 2, "speaker": "A", "intent": "suggest an alternative date"},
                {"turn_index": 3, "speaker": "B", "intent": "tentatively agree"},
            ],
        }
    )
    free_json = json.dumps(
        {
            "conversation_context": "Two friends plan a weekend trip.",
            "user_a_context": "Anda ingin mengajak teman lama jalan-jalan.",
            "user_b_context": "당신은 바빠 일정이 있습니다.",
            "intent_outline": None,
        }
    )
    checklist_items = [
        {"function_id": "L2_f3", "layer": "layer_2", "text": "Does the outline show B declining before agreeing?"},
        {"function_id": "L1_f2", "layer": "layer_1", "text": "Does the outline preserve the weekend-trip proposal?"},
        {"function_id": "L3_f2", "layer": "layer_3", "text": "Does the outline preserve B's polite refusal register toward A?"},
    ]

    class SeedProvider:
        model_name = "mock-seed"

        def __init__(self, payload):
            self.payload = payload

        def generate(self, prompt, **kwargs):
            return self.payload

    class ChecklistProvider:
        model_name = "mock-checklist"

        def generate(self, prompt, **kwargs):
            return json.dumps({"items": checklist_items})

    guided_seed = gen.generate_one_dynamic_seed(
        SeedProvider(guided_json),
        ChecklistProvider(),
        prompt="unused in mock",
        lang_a="ind",
        lang_b="kor",
        num_turns=4,
        guidance="guided",
        use_grounding=True,
        seed_file="test_proverbs.xlsx",
        seed_row_id=2,
        category="MAPS-Proverb-Pragmatics-MT",
        conversation_id="indkor_mtd_0001",
    )
    assert guided_seed.guidance == "guided"
    assert guided_seed.intent_outline is not None
    assert [b.speaker for b in guided_seed.intent_outline] == ["A", "B", "A", "B"]
    assert guided_seed.conversation_checklist_items is not None
    assert guided_seed.conversation_verification_prompt.startswith("1.")

    free_seed = gen.generate_one_dynamic_seed(
        SeedProvider(free_json),
        None,
        prompt="unused in mock",
        lang_a="ind",
        lang_b="kor",
        num_turns=4,
        guidance="free",
        use_grounding=True,
        seed_file="test_proverbs.xlsx",
        seed_row_id=3,
        category="MAPS-Proverb-Pragmatics-MT",
        conversation_id="indkor_mtd_0002",
    )
    assert free_seed.guidance == "free"
    assert free_seed.intent_outline is None
    assert free_seed.conversation_checklist_items is None
    assert free_seed.conversation_verification_prompt is None

    print("  ✓ generate_one_dynamic_seed mock end-to-end test passed (guided + free)")


# ---------------------------------------------------------------------------
# Step 2: mt-prepare + turn-unit schema
# ---------------------------------------------------------------------------
def _sample_scripted_conversation(conversation_id="indkor_mts_0001", num_turns=4):
    speakers = ["A" if i % 2 == 0 else "B" for i in range(num_turns)]
    turns = [
        {
            "turn_index": i,
            "speaker": speakers[i],
            "text": f"turn {i} text ({speakers[i]})",
            "checklist_items": [
                {"function_id": "L1_f1", "layer": "layer_1", "text": f"item for turn {i}"}
            ],
            "verification_prompt": f"1. item for turn {i}",
        }
        for i in range(num_turns)
    ]
    return {
        "conversation_id": conversation_id,
        "mode": "scripted",
        "guidance": None,
        "lang_a": "ind",
        "lang_b": "kor",
        "Category": "MAPS-Proverb-Pragmatics-MT",
        "conversation_context": "Two colleagues discuss a project.",
        "user_a_context": "User A persona (Indonesian).",
        "user_b_context": "User B persona (Korean).",
        "turns": turns,
        "conversation_checklist_items": [
            {"function_id": "L2_f1", "layer": "layer_2", "text": "conversation-level item"}
        ],
        "conversation_verification_prompt": "1. conversation-level item",
        "seed_file": "test_proverbs.xlsx",
        "seed_row_id": 1,
        "generation_metadata": {},
    }


def test_conversation_to_turn_units_record_id_uniqueness():
    print("Testing conversation_to_turn_units record_id uniqueness + segment_id/direction rule...")

    conversation = _sample_scripted_conversation(num_turns=6)
    units = mt_ops.conversation_to_turn_units(conversation)
    assert len(units) == 6

    record_ids = [u["record_id"] for u in units]
    assert len(set(record_ids)) == 6, "record_ids must be unique per turn"

    for i, u in enumerate(units):
        assert u["segment_id"] == f"indkor_mts_0001_t{i:02d}"
        expected_direction = "ind-kor" if u["speaker"] == "A" else "kor-ind"
        assert u["direction"] == expected_direction
        # record_id must be exactly {segment_id}_{direction} (io.record_id's rule)
        assert u["record_id"] == f"{u['segment_id']}_{u['direction']}"

    # alternating direction confirms per-turn bidirectionality (D2)
    directions = [u["direction"] for u in units]
    assert directions == ["ind-kor", "kor-ind", "ind-kor", "kor-ind", "ind-kor", "kor-ind"]

    print("  ✓ record_id uniqueness + segment_id/direction rule tests passed")


def test_conversation_to_turn_units_history_correctness():
    print("Testing authored_history correctness...")

    conversation = _sample_scripted_conversation(num_turns=4)
    units = mt_ops.conversation_to_turn_units(conversation)

    assert units[0]["authored_history"] == []
    assert units[1]["authored_history"] == [
        {"turn_index": 0, "speaker": "A", "source_text": "turn 0 text (A)"}
    ]
    assert units[2]["authored_history"] == [
        {"turn_index": 0, "speaker": "A", "source_text": "turn 0 text (A)"},
        {"turn_index": 1, "speaker": "B", "source_text": "turn 1 text (B)"},
    ]
    assert len(units[3]["authored_history"]) == 3

    # each unit's own authored_history list must be independent (no shared mutation)
    units[1]["authored_history"].append({"turn_index": 99, "speaker": "X", "source_text": "poison"})
    assert units[0]["authored_history"] == []
    assert len(units[2]["authored_history"]) == 2

    print("  ✓ authored_history correctness tests passed")


def test_conversation_to_turn_units_listener_side_switching():
    print("Testing listener_context side switching...")

    conversation = _sample_scripted_conversation(num_turns=4)
    units = mt_ops.conversation_to_turn_units(conversation)

    for u in units:
        if u["speaker"] == "A":
            assert u["listener_context"] == conversation["user_b_context"]
        else:
            assert u["listener_context"] == conversation["user_a_context"]

    print("  ✓ listener_context side switching tests passed")


def test_conversation_to_turn_units_checklist_passthrough():
    print("Testing per-turn checklist_items/verification_prompt passthrough...")

    conversation = _sample_scripted_conversation(num_turns=2)
    units = mt_ops.conversation_to_turn_units(conversation)

    assert units[0]["checklist_items"] == [
        {"function_id": "L1_f1", "layer": "layer_1", "text": "item for turn 0"}
    ]
    assert units[0]["verification_prompt"] == "1. item for turn 0"
    assert units[0]["mode"] == "scripted"
    assert units[0]["guidance"] is None
    assert units[0]["num_turns"] == 2
    assert units[0]["category"] == "MAPS-Proverb-Pragmatics-MT"

    print("  ✓ checklist_items/verification_prompt passthrough tests passed")


def test_run_mt_prepare_end_to_end():
    print("Testing run_mt_prepare end-to-end (dedup + rewrite semantics)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "scenarios.jsonl")
        c1 = _sample_scripted_conversation("indkor_mts_0001", num_turns=4)
        c2 = _sample_scripted_conversation("indkor_mts_0002", num_turns=4)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(c1) + "\n")
            f.write(json.dumps(c2) + "\n")
            f.write(json.dumps(c1) + "\n")  # duplicate conversation -> duplicate record_ids

        output_path = os.path.join(tmpdir, "00_units.jsonl")
        mt_stages.run_mt_prepare([data_path], output_path)

        units = read_jsonl(output_path)
        assert len(units) == 8, "8 turns total (4+4), duplicate conversation deduped by record_id"
        conv_ids = {u["conversation_id"] for u in units}
        assert conv_ids == {"indkor_mts_0001", "indkor_mts_0002"}

        # prepare always rewrites: re-running with only c1 must not accumulate old units
        data_path2 = os.path.join(tmpdir, "scenarios2.jsonl")
        with open(data_path2, "w", encoding="utf-8") as f:
            f.write(json.dumps(c1) + "\n")
        mt_stages.run_mt_prepare([data_path2], output_path)
        units2 = read_jsonl(output_path)
        assert len(units2) == 4

    print("  ✓ run_mt_prepare end-to-end test passed")


# ---------------------------------------------------------------------------
# Step 3: mt-translate (scripted, sync waves) + --context-mode
# ---------------------------------------------------------------------------
class MockEchoProvider:
    """Deterministic mock: returns a distinct XLAT-N string per call, in order."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append(prompt)
        # embeds a non-Latin-1 char so the mojibake guard (reused from
        # pipeline.operations._assign_translation) doesn't flag kor-target turns
        return f"XLAT-{len(self.calls)}-테스트"

    def get_provider_name(self):
        return "mock-echo"


def _write_units(units, path):
    with open(path, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")


def test_run_mt_translate_history_injection_and_context_mode():
    print("Testing mt-translate history injection (transcript) + context-mode none...")

    conversation = _sample_scripted_conversation("test_mts_0001", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)

        out_path = os.path.join(tmpdir, "01_translated_transcript.jsonl")
        provider = MockEchoProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="transcript",
            interpreter_provider=provider,
            interpreter_label="mock:mock-model",
        )
        assert len(provider.calls) == 3
        assert "Conversation so far" in provider.calls[1]
        assert "XLAT-1" in provider.calls[1]  # turn 1's prompt sees turn 0's translation
        assert "XLAT-1" in provider.calls[2]
        assert "XLAT-2" in provider.calls[2]  # turn 2's prompt sees turns 0 AND 1

        results = read_jsonl(out_path)
        assert len(results) == 3
        by_turn = {r["turn_index"]: r for r in results}
        assert by_turn[0]["history"] == []
        assert len(by_turn[1]["history"]) == 1
        assert by_turn[1]["history"][0]["translated_text"] == "XLAT-1-테스트"
        assert len(by_turn[2]["history"]) == 2
        assert by_turn[2]["history"][1]["translated_text"] == "XLAT-2-테스트"
        assert all("authored_history" not in r for r in results)

        out_path2 = os.path.join(tmpdir, "01_translated_none.jsonl")
        provider2 = MockEchoProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path2,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="none",
            interpreter_provider=provider2,
            interpreter_label="mock:mock-model",
        )
        assert len(provider2.calls) == 3
        assert "Conversation so far" not in provider2.calls[1]
        assert "Conversation so far" not in provider2.calls[2]
        assert "XLAT-1" not in provider2.calls[2]
        results2 = read_jsonl(out_path2)
        by_turn2 = {r["turn_index"]: r for r in results2}
        assert by_turn2[0]["history"] == []
        assert len(by_turn2[1]["history"]) == 1
        assert by_turn2[1]["history"][0]["translated_text"] == "XLAT-1-테스트"
        assert len(by_turn2[2]["history"]) == 2
        assert by_turn2[2]["history"][1]["translated_text"] == "XLAT-2-테스트"

    print("  ✓ mt-translate history injection + context-mode none tests passed")


def test_run_mt_translate_resume_after_deletion():
    print("Testing mt-translate resume after deleting a mid-turn line...")

    conversation = _sample_scripted_conversation("test_mts_0002", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)

        out_path = os.path.join(tmpdir, "01_translated.jsonl")
        provider = MockEchoProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="transcript",
            interpreter_provider=provider,
            interpreter_label="mock:mock-model",
        )
        assert len(provider.calls) == 3

        results = read_jsonl(out_path)
        kept = [r for r in results if r["turn_index"] != 1]
        assert len(kept) == 2
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

        provider2 = MockEchoProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="transcript",
            interpreter_provider=provider2,
            interpreter_label="mock:mock-model",
        )
        assert len(provider2.calls) == 1, "only the deleted mid-turn should be regenerated"

        final = read_jsonl(out_path)
        assert sorted(r["turn_index"] for r in final) == [0, 1, 2]

    print("  ✓ mt-translate resume-after-deletion test passed")


def test_run_mt_translate_pending_sidecar_withholds_successors():
    print("Testing mt-translate pending sidecar + successor withholding on failure...")

    class MockFailingProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system_prompt=None, **kwargs):
            self.calls += 1
            if "FAIL_MARKER" in prompt:
                raise RuntimeError("simulated translate failure")
            return f"XLAT-{self.calls}-테스트"

        def get_provider_name(self):
            return "mock-fail"

    conversation = _sample_scripted_conversation("test_mts_0003", num_turns=3)
    conversation["turns"][1]["text"] = "FAIL_MARKER turn text"
    units = mt_ops.conversation_to_turn_units(conversation)

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)

        out_path = os.path.join(tmpdir, "01_translated.jsonl")
        provider = MockFailingProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="transcript",
            interpreter_provider=provider,
            interpreter_label="mock:mock-model",
        )

        results = read_jsonl(out_path)
        assert sorted(r["turn_index"] for r in results) == [0], "only turn 0 should succeed"

        pending = read_jsonl(out_path + ".pending.jsonl")
        assert sorted(r["turn_index"] for r in pending) == [1], "turn 1 goes to the pending sidecar"

        assert not any(r["turn_index"] == 2 for r in results)
        assert not any(r["turn_index"] == 2 for r in pending)
        assert provider.calls == 2, "turn 2 must never even be attempted this run"

    print("  ✓ mt-translate pending sidecar + successor withholding test passed")


def test_run_mt_translate_no_context_failure_does_not_withhold_successors():
    print("Testing mt-translate no-context failure does not withhold successors...")

    class MockFailingProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system_prompt=None, **kwargs):
            self.calls += 1
            if "FAIL_MARKER" in prompt:
                raise RuntimeError("simulated translate failure")
            return f"XLAT-{self.calls}-테스트"

        def get_provider_name(self):
            return "mock-fail"

    conversation = _sample_scripted_conversation("test_mts_noctx", num_turns=3)
    conversation["turns"][1]["text"] = "FAIL_MARKER turn text"
    units = mt_ops.conversation_to_turn_units(conversation)

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)

        out_path = os.path.join(tmpdir, "01_translated.jsonl")
        provider = MockFailingProvider()
        mt_stages.run_mt_translate(
            units_path,
            out_path,
            provider_type="mock",
            model_name="mock-model",
            concurrency=1,
            context_mode="none",
            interpreter_provider=provider,
            interpreter_label="mock:mock-model",
        )

        results = read_jsonl(out_path)
        assert sorted(r["turn_index"] for r in results) == [0, 2]
        pending = read_jsonl(out_path + ".pending.jsonl")
        assert [r["turn_index"] for r in pending] == [1]
        assert provider.calls == 3, "context-free turn 2 must still be attempted"

    print("  ✓ mt-translate no-context successor independence test passed")


# ---------------------------------------------------------------------------
# Step 4: mt-respond + mt-verify
# ---------------------------------------------------------------------------
class MockUserSimProvider:
    def __init__(self, response="mock listener response"):
        self.response = response
        self.calls = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append((prompt, system_prompt))
        return self.response


def _translated_unit(conversation_id="test_mtr_0001", num_turns=3, turn_index=1):
    conversation = _sample_scripted_conversation(conversation_id, num_turns=num_turns)
    units = mt_ops.conversation_to_turn_units(conversation)
    unit = dict(units[turn_index])
    # simulate what mt-translate would have attached by this point
    unit["translated_text"] = f"translated turn {turn_index}"
    unit["history"] = [
        {
            "turn_index": h["turn_index"],
            "speaker": h["speaker"],
            "source_text": h["source_text"],
            "translated_text": f"translated turn {h['turn_index']}",
        }
        for h in unit.pop("authored_history", [])
    ]
    unit["model"] = "mock-model"
    unit["interpreter"] = "mock:mock-model"
    unit["context_mode"] = "transcript"
    return unit


def test_build_turn_respond_history_text():
    print("Testing build_turn_respond_history_text listener-side text selection...")

    # turn_index=2, speaker "A" (turns 0=A,1=B,2=A) -> listener is B
    unit = _translated_unit(num_turns=3, turn_index=2)
    text = mt_ops.build_turn_respond_history_text(unit)
    # turn 0 was spoken by A (not the listener B) -> use its translated_text
    assert "translated turn 0" in text
    # turn 1 was spoken by B (the listener) -> use B's own source_text, not a translation
    assert "turn 1 text (B)" in text
    assert "translated turn 1" not in text

    print("  ✓ build_turn_respond_history_text tests passed")


def test_respond_turn_record_success():
    print("Testing respond_turn_record success path...")

    unit = _translated_unit(turn_index=1)  # speaker B, listener A
    provider = MockUserSimProvider(response="listener says hi back")

    def factory(lang):
        assert lang == unit["target_lang"]
        return provider, "mock-listener-model", "MockLang"

    out = mt_ops.respond_turn_record(unit, factory)
    assert out["listener_response"] == "listener says hi back"
    assert out["listener_model"] == "mock-listener-model"
    assert "respond_skipped" not in out
    assert len(provider.calls) == 1
    prompt, system_prompt = provider.calls[0]
    assert prompt == unit["translated_text"]
    assert unit["listener_context"] in system_prompt

    print("  ✓ respond_turn_record success test passed")


def test_respond_turn_record_no_translation():
    print("Testing respond_turn_record with no translation...")

    unit = _translated_unit(turn_index=1)
    unit["translated_text"] = None
    out = mt_ops.respond_turn_record(unit, lambda lang: (MockUserSimProvider(), "m", "L"))
    assert out["listener_response"] is None
    assert out["respond_skipped"] == "no translation"

    print("  ✓ respond_turn_record no-translation test passed")


def test_respond_turn_record_unconfigured_language():
    print("Testing respond_turn_record respond_skipped for an unconfigured language...")

    unit = _translated_unit(turn_index=1)

    def factory(lang):
        raise ValueError(f"No user-simulation provider configured for language '{lang}'")

    out = mt_ops.respond_turn_record(unit, factory)
    assert out["listener_response"] is None
    assert "respond_skipped" in out
    assert "no user-sim model configured" in out["respond_skipped"]
    assert "respond_error" not in out

    print("  ✓ respond_turn_record unconfigured-language test passed")


def test_run_mt_respond_end_to_end():
    print("Testing run_mt_respond end-to-end (mixed configured/unconfigured languages)...")

    conversation = _sample_scripted_conversation("test_mtr_0002", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)
    for u in units:
        u["translated_text"] = f"translated {u['turn_index']}"

    def factory(lang):
        if lang == "kor":
            return MockUserSimProvider(response=f"resp-{lang}"), "mock-model", "Korean"
        raise ValueError(f"No user-simulation provider configured for language '{lang}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "01_translated.jsonl")
        _write_units(units, input_path)
        output_path = os.path.join(tmpdir, "02_responded.jsonl")

        mt_stages.run_mt_respond(input_path, output_path, concurrency=1, user_sim_factory=factory)

        results = read_jsonl(output_path)
        # only turns targeting "kor" (i.e. speaker A, translated into kor) succeed
        assert all(r["target_lang"] == "kor" for r in results)
        assert all(r.get("listener_response") for r in results)

        pending = read_jsonl(output_path + ".pending.jsonl")
        assert all(r["target_lang"] == "ind" for r in pending)
        assert all(r.get("respond_skipped") for r in pending)

    print("  ✓ run_mt_respond end-to-end test passed")


def test_verify_turn_record_with_real_glotlid():
    print("Testing verify_turn_record with the real (local, cached) GlotLID model...")

    from interpreter_agent_eval.utils.language_verification import load_glotlid_model

    glotlid_model = load_glotlid_model()
    if glotlid_model is None:
        print("  (skipped: GlotLID model unavailable in this environment)")
        return

    unit = _translated_unit(turn_index=1)  # speaker B, target_lang "ind"
    unit["translated_text"] = "Halo, apa kabar? Semoga harimu menyenangkan dan penuh kebahagiaan."
    unit["listener_response"] = "Hello, how are you today my friend"  # wrong language on purpose

    out = mt_ops.verify_turn_record(unit, glotlid_model, min_confidence=0.5)
    assert out["translation_language_check"]["is_correct"] is True
    assert out["response_language_check"]["is_correct"] is False
    assert out["language_check_passed"] is False  # response check sets the verdict

    # without a listener_response at all (--skip-respond path), verdict comes
    # from the translation check alone
    unit2 = dict(unit)
    unit2.pop("listener_response")
    out2 = mt_ops.verify_turn_record(unit2, glotlid_model, min_confidence=0.5)
    assert out2["response_language_check"] is None
    assert out2["language_check_passed"] is True

    print("  ✓ verify_turn_record (real GlotLID) tests passed")


# ---------------------------------------------------------------------------
# Step 5: mt-judge-turns + mt-judge-conv
# ---------------------------------------------------------------------------
def _mock_judge_json(items):
    return json.dumps(
        {
            "results": [
                {"id": i + 1, "criteria": c, "met": met, "reasoning": f"reason {i}"}
                for i, (c, met) in enumerate(items)
            ]
        }
    )


class MockJudgeProvider:
    def __init__(self, items=None, fail_structured=False):
        self.items = items or [("criterion 1", True), ("criterion 2", False)]
        self.fail_structured = fail_structured
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self.fail_structured and "response_schema" in kwargs:
            raise RuntimeError("structured output not supported by this mock")
        return _mock_judge_json(self.items)


def test_build_turn_judge_prompt_includes_transcript():
    print("Testing build_turn_judge_prompt: transcript block present, judge-history slot optional...")

    unit0 = _translated_unit(turn_index=0, num_turns=3)
    unit0["history"] = []
    prompt0 = mt_ops.build_turn_judge_prompt(unit0)
    assert "(this is the first turn; no prior turns)" in prompt0

    unit1 = _translated_unit(turn_index=1, num_turns=3)
    prompt1 = mt_ops.build_turn_judge_prompt(unit1)
    assert "translated turn 0" in prompt1  # prior turn's translation present
    assert unit1["verification_prompt"] in prompt1

    # --judge-history slot: absent by default, present only when prior_judgments given
    assert "Prior turns' judged criteria" not in prompt1
    prior = [{"turn_index": 0, "evaluation": {"results": [{"criteria": "c1", "met": True}]}}]
    prompt1_with_history = mt_ops.build_turn_judge_prompt(unit1, prior_judgments=prior)
    assert "Prior turns' judged criteria" in prompt1_with_history
    assert "c1: Yes" in prompt1_with_history

    print("  ✓ build_turn_judge_prompt transcript + judge-history slot tests passed")


def test_judge_turn_record_mock():
    print("Testing judge_turn_record with a mock judge provider...")

    unit = _translated_unit(turn_index=1)
    provider = MockJudgeProvider(items=[("criterion A", True), ("criterion B", False)])
    out = mt_ops.judge_turn_record(unit, provider, "mock:judge")
    assert out["judge"] == "mock:judge"
    assert out["evaluation"]["results"][0]["met"] is True
    assert out["completion_rate"] == "1/2"
    assert out["success_rate"] == 0.5

    # no translation -> judge_skipped, judge never called
    unit_no_trans = dict(unit)
    unit_no_trans["translated_text"] = None
    provider2 = MockJudgeProvider()
    out2 = mt_ops.judge_turn_record(unit_no_trans, provider2, "mock:judge")
    assert out2["evaluation"] is None
    assert out2["judge_skipped"] == "no translation"
    assert len(provider2.prompts) == 0

    # structured-call failure falls back to plain generation
    provider3 = MockJudgeProvider(fail_structured=True)
    out3 = mt_ops.judge_turn_record(unit, provider3, "mock:judge")
    assert out3["evaluation"] is not None
    assert len(provider3.prompts) == 2  # structured attempt + plain fallback

    print("  ✓ judge_turn_record mock tests passed")


def test_run_mt_judge_turns_default_flat_no_history_slot():
    print("Testing run_mt_judge_turns default (flat, no --judge-history) never injects prior verdicts...")

    conversation = _sample_scripted_conversation("test_mtj_0001", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)
    for u in units:
        u["translated_text"] = f"translated {u['turn_index']}"
        u["history"] = [
            {**h, "translated_text": f"translated {h['turn_index']}"} for h in u.pop("authored_history", [])
        ]

    provider = MockJudgeProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "03_verified.jsonl")
        _write_units(units, input_path)
        output_path = os.path.join(tmpdir, "04_turn_judged.jsonl")

        mt_stages.run_mt_judge_turns(
            input_path, output_path, concurrency=1, judge_provider=provider, judge_label="mock:judge"
        )

        results = read_jsonl(output_path)
        assert len(results) == 3
        assert all(r.get("evaluation") is not None for r in results)
        assert all("Prior turns' judged criteria" not in p for p in provider.prompts)

    print("  ✓ run_mt_judge_turns default flat-mode test passed")


def test_run_mt_judge_turns_with_judge_history():
    print("Testing run_mt_judge_turns --judge-history wave loop (experimental, off by default)...")

    conversation = _sample_scripted_conversation("test_mtj_0002", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)
    for u in units:
        u["translated_text"] = f"translated {u['turn_index']}"
        u["history"] = [
            {**h, "translated_text": f"translated {h['turn_index']}"} for h in u.pop("authored_history", [])
        ]

    provider = MockJudgeProvider(items=[("c1", True)])
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "03_verified.jsonl")
        _write_units(units, input_path)
        output_path = os.path.join(tmpdir, "04_turn_judged.jsonl")

        mt_stages.run_mt_judge_turns(
            input_path,
            output_path,
            concurrency=1,
            judge_provider=provider,
            judge_label="mock:judge",
            judge_history=True,
        )

        results = read_jsonl(output_path)
        assert len(results) == 3

        # turn 0's prompt has no history slot; turns 1 and 2 do (prior verdicts fed forward)
        assert not any("Prior turns' judged criteria" in p for p in provider.prompts[:1])
        assert any("Prior turns' judged criteria" in p for p in provider.prompts[1:])

    print("  ✓ run_mt_judge_turns --judge-history test passed")


def test_ensure_conversation_checklist_authored_vs_posthoc():
    print("Testing ensure_conversation_checklist: authored (scripted/guided) vs posthoc (free-flow)...")

    turn_records = [
        {"turn_index": 0, "speaker": "A", "source_text": "s0", "translated_text": "t0"},
        {"turn_index": 1, "speaker": "B", "source_text": "s1", "translated_text": "t1"},
    ]

    authored_unit = {
        "conversation_id": "c1",
        "lang_a": "ind",
        "lang_b": "kor",
        "conversation_context": "ctx",
        "conversation_checklist_items": [{"function_id": "L1_f1", "layer": "layer_1", "text": "authored item"}],
        "conversation_verification_prompt": "1. authored item",
    }
    # floor-satisfying (1 item per layer) so this doesn't trigger a
    # floor-retry, and text is genuinely distinct across items so the
    # always-on dedup doesn't collapse them.
    checklist_provider = MockChecklistProvider(
        [
            {"function_id": "L2_f1", "layer": "layer_2", "text": "posthoc item"},
            {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation preserve the specific fact raised in turn 0?"},
            {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation reflect the appropriate register for this exchange?"},
        ]
    )
    out_authored = mt_ops.ensure_conversation_checklist(authored_unit, turn_records, checklist_provider, "kor")
    assert out_authored["checklist_provenance"] == "authored"
    assert out_authored["conversation_verification_prompt"] == "1. authored item"
    assert checklist_provider.call_count == 0  # never called — checklist already present

    free_unit = {
        "conversation_id": "c2",
        "lang_a": "ind",
        "lang_b": "kor",
        "conversation_context": "ctx",
        "conversation_checklist_items": None,
        "conversation_verification_prompt": None,
    }
    out_posthoc = mt_ops.ensure_conversation_checklist(free_unit, turn_records, checklist_provider, "kor")
    assert out_posthoc["checklist_provenance"] == "posthoc"
    assert out_posthoc["conversation_checklist_items"][0]["text"] == "posthoc item"
    assert "posthoc item" in out_posthoc["conversation_verification_prompt"]
    assert checklist_provider.call_count == 1

    print("  ✓ ensure_conversation_checklist authored-vs-posthoc tests passed")


def test_build_conversation_judge_prompt_failed_turns_note():
    print("Testing build_conversation_judge_prompt failed-turns annotation...")

    conversation_unit = {
        "conversation_context": "ctx",
        "conversation_verification_prompt": "1. cross-turn item",
    }
    turn_records = [
        {"turn_index": 0, "speaker": "A", "source_text": "s0", "translated_text": "t0", "language_check_passed": True},
        {"turn_index": 1, "speaker": "B", "source_text": "s1", "translated_text": "t1", "language_check_passed": False},
    ]
    prompt = mt_ops.build_conversation_judge_prompt(conversation_unit, turn_records)
    assert "turn(s) [1]" in prompt
    assert "1. cross-turn item" in prompt

    # no verification_prompt -> None (nothing to judge against)
    assert mt_ops.build_conversation_judge_prompt({"conversation_verification_prompt": None}, turn_records) is None

    print("  ✓ build_conversation_judge_prompt failed-turns-note tests passed")


def test_judge_conversation_record_mock():
    print("Testing judge_conversation_record with a mock judge provider...")

    conversation_unit = {
        "conversation_context": "ctx",
        "conversation_verification_prompt": "1. cross-turn item",
    }
    turn_records = [{"turn_index": 0, "speaker": "A", "source_text": "s0", "translated_text": "t0"}]
    provider = MockJudgeProvider(items=[("cross-turn item", True)])
    out = mt_ops.judge_conversation_record(conversation_unit, turn_records, provider, "mock:judge")
    assert out["evaluation"]["results"][0]["met"] is True
    assert out["completion_rate"] == "1/1"

    print("  ✓ judge_conversation_record mock test passed")


def test_run_mt_judge_conv_end_to_end():
    print("Testing run_mt_judge_conv end-to-end: authored (scripted) + posthoc (free-flow)...")

    scripted_conv = _sample_scripted_conversation("test_mtjc_scripted", num_turns=2)
    free_conv = {
        "conversation_id": "test_mtjc_free",
        "mode": "dynamic",
        "guidance": "free",
        "lang_a": "ind",
        "lang_b": "kor",
        "conversation_context": "Two friends chat.",
        "conversation_checklist_items": None,
        "conversation_verification_prompt": None,
    }

    turn_judged = []
    for i, speaker in enumerate(["A", "B"]):
        target = "kor" if speaker == "A" else "ind"
        turn_judged.append(
            {
                "record_id": f"test_mtjc_scripted_t{i:02d}_{'ind-kor' if speaker == 'A' else 'kor-ind'}",
                "conversation_id": "test_mtjc_scripted",
                "turn_index": i,
                "speaker": speaker,
                "source_text": f"src {i}",
                "translated_text": f"tgt {i}",
                "target_lang": target,
                "language_check_passed": True,
            }
        )
        turn_judged.append(
            {
                "record_id": f"test_mtjc_free_t{i:02d}_x",
                "conversation_id": "test_mtjc_free",
                "turn_index": i,
                "speaker": speaker,
                "source_text": f"free src {i}",
                "translated_text": f"free tgt {i}",
                "target_lang": target,
                "language_check_passed": True,
            }
        )

    judge_provider = MockJudgeProvider(items=[("conv item", True)])
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L2_f9", "layer": "layer_2", "text": "posthoc conv item", "extra": None}]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        scenarios_path = os.path.join(tmpdir, "scenarios.jsonl")
        with open(scenarios_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(scripted_conv) + "\n")
            f.write(json.dumps(free_conv) + "\n")

        turn_judged_path = os.path.join(tmpdir, "04_turn_judged.jsonl")
        _write_units(turn_judged, turn_judged_path)

        output_path = os.path.join(tmpdir, "05_conv_judged.jsonl")
        mt_stages.run_mt_judge_conv(
            [scenarios_path],
            turn_judged_path,
            output_path,
            concurrency=1,
            judge_provider=judge_provider,
            judge_label="mock:judge",
            checklist_provider=checklist_provider,
        )

        results = {r["conversation_id"]: r for r in read_jsonl(output_path)}
        assert results["test_mtjc_scripted"]["record_id"] == "test_mtjc_scripted_conv"
        assert results["test_mtjc_scripted"]["evaluation"] is not None

        free_result = results["test_mtjc_free"]
        assert free_result["record_id"] == "test_mtjc_free_conv"
        assert free_result["checklist_provenance"] == "posthoc"
        assert free_result["conversation_checklist_items"][0]["text"] == "posthoc conv item"
        assert free_result["evaluation"] is not None

    print("  ✓ run_mt_judge_conv end-to-end test passed (authored + posthoc)")


# ---------------------------------------------------------------------------
# Step 6: mt-converse (dynamic, guided + free)
# ---------------------------------------------------------------------------
def _sample_dynamic_seed(conversation_id="test_mtd_0001", guidance="guided", num_turns=4):
    seed = {
        "conversation_id": conversation_id,
        "mode": "dynamic",
        "guidance": guidance,
        "lang_a": "ind",
        "lang_b": "kor",
        "Category": "MAPS-Proverb-Pragmatics-MT",
        "conversation_context": "Two friends catch up.",
        "user_a_context": "User A persona (Indonesian).",
        "user_b_context": "User B persona (Korean).",
        "num_turns": num_turns,
        "seed_file": "test_proverbs.xlsx",
        "seed_row_id": 1,
    }
    if guidance == "guided":
        seed["intent_outline"] = [
            {"turn_index": i, "speaker": "A" if i % 2 == 0 else "B", "intent": f"beat {i}"}
            for i in range(num_turns)
        ]
        seed["conversation_checklist_items"] = [
            {"function_id": "L2_f1", "layer": "layer_2", "text": "authored conv item"}
        ]
        seed["conversation_verification_prompt"] = "1. authored conv item"
    else:
        seed["intent_outline"] = None
        seed["conversation_checklist_items"] = None
        seed["conversation_verification_prompt"] = None
    return seed


class MockUserSimFactory:
    """lang -> (provider, label, lang_full); records which languages were requested."""

    def __init__(self, responses=None, fail_langs=None):
        self.responses = responses or {}
        self.fail_langs = fail_langs or set()
        self.requested_langs = []
        self.providers = {}

    def __call__(self, lang):
        self.requested_langs.append(lang)
        if lang in self.fail_langs:
            raise ValueError(f"No user-simulation provider configured for language '{lang}'")
        if lang not in self.providers:
            response = self.responses.get(lang, f"utterance in {lang} 테스트")
            self.providers[lang] = MockUserSimProvider(response=response)
        return self.providers[lang], f"mock-{lang}", lang


def test_render_history_for_side():
    print("Testing render_history_for_side (shared by mt-respond and mt-converse)...")

    history = [
        {"turn_index": 0, "speaker": "A", "source_text": "a says hi", "translated_text": "a translated to kor"},
        {"turn_index": 1, "speaker": "B", "source_text": "b replies", "translated_text": "b translated to ind"},
    ]
    text_a = mt_ops.render_history_for_side(history, "A")
    assert "a says hi" in text_a  # A's own words, untranslated
    assert "b translated to ind" in text_a  # B's turn, translated INTO A's language
    assert "a translated to kor" not in text_a
    assert "b replies" not in text_a

    text_b = mt_ops.render_history_for_side(history, "B")
    assert "b replies" in text_b
    assert "a translated to kor" in text_b
    assert "a says hi" not in text_b
    assert "b translated to ind" not in text_b

    print("  ✓ render_history_for_side tests passed")


def test_build_user_turn_prompt_guided_vs_free():
    print("Testing build_user_turn_prompt: guided intent vs free continuation instruction...")

    seed = _sample_dynamic_seed(guidance="guided")
    guided_prompt = mt_ops.build_user_turn_prompt(seed, "A", "history so far", "politely decline")
    assert "politely decline" in guided_prompt
    assert "do not try to" not in guided_prompt.lower() or "wrap up" not in guided_prompt.lower()

    free_prompt = mt_ops.build_user_turn_prompt(seed, "A", "history so far", None)
    assert "do not try to" in free_prompt.lower() and "wrap up" in free_prompt.lower()
    assert "politely decline" not in free_prompt

    print("  ✓ build_user_turn_prompt guided-vs-free tests passed")


def test_converse_next_turn_mock_success():
    print("Testing converse_next_turn happy path (mock user-sim + checklist + interpreter)...")

    seed = _sample_dynamic_seed(guidance="guided", num_turns=4)
    factory = MockUserSimFactory(responses={"ind": "Halo, apa kabar? 테스트"})
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L1_f1", "layer": "layer_1", "text": "checklist item"}]
    )
    interpreter_provider = MockEchoProvider()

    out = mt_ops.converse_next_turn(
        seed,
        0,
        [],
        factory,
        checklist_provider,
        interpreter_provider,
        "mock:interpreter",
        "mock-model",
        {},
        context_mode="transcript",
    )
    assert out["source_text"] == "Halo, apa kabar? 테스트"
    assert out["speaker"] == "A"
    assert out["intent"] == "beat 0"
    assert out["translated_text"] is not None
    assert out["mode"] == "dynamic"
    assert "authored_history" not in out  # translate_turn_record pops it into "history"
    assert out["history"] == []
    assert factory.requested_langs == ["ind"]  # only the speaker's language requested

    print("  ✓ converse_next_turn happy-path test passed")


def test_converse_next_turn_user_sim_failure():
    print("Testing converse_next_turn: user-sim failure -> error, no translation...")

    seed = _sample_dynamic_seed(guidance="free", num_turns=4)
    factory = MockUserSimFactory(fail_langs={"ind"})
    out = mt_ops.converse_next_turn(
        seed, 0, [], factory, MockChecklistProvider([]), MockEchoProvider(), "mock", "mock-model", {}
    )
    assert out["translated_text"] is None
    assert "user-sim" in out["error"]

    print("  ✓ converse_next_turn user-sim-failure test passed")


def test_run_mt_converse_end_to_end_guided_and_free():
    print("Testing run_mt_converse end-to-end (guided + free, full conversations)...")

    guided_seed = _sample_dynamic_seed("test_mtc_guided", guidance="guided", num_turns=4)
    free_seed = _sample_dynamic_seed("test_mtc_free", guidance="free", num_turns=4)

    factory = MockUserSimFactory(responses={"ind": "utterance 테스트 ind", "kor": "발화 테스트 kor"})
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L1_f1", "layer": "layer_1", "text": "checklist item"}]
    )
    interpreter_provider = MockEchoProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = os.path.join(tmpdir, "seeds.jsonl")
        with open(seeds_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(guided_seed) + "\n")
            f.write(json.dumps(free_seed) + "\n")

        output_path = os.path.join(tmpdir, "01_conversed.jsonl")
        mt_stages.run_mt_converse(
            [seeds_path],
            output_path,
            concurrency=2,
            user_sim_factory=factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            checklist_provider=checklist_provider,
        )

        results = read_jsonl(output_path)
        by_conv = {}
        for r in results:
            by_conv.setdefault(r["conversation_id"], []).append(r)

        assert len(by_conv["test_mtc_guided"]) == 4
        assert len(by_conv["test_mtc_free"]) == 4

        guided_turns = sorted(by_conv["test_mtc_guided"], key=lambda r: r["turn_index"])
        assert [t["speaker"] for t in guided_turns] == ["A", "B", "A", "B"]
        assert all(t["intent"] for t in guided_turns)
        assert len(guided_turns[3]["history"]) == 3

        free_turns = sorted(by_conv["test_mtc_free"], key=lambda r: r["turn_index"])
        assert all(t["intent"] is None for t in free_turns)

    print("  ✓ run_mt_converse end-to-end test passed (guided + free)")


def test_run_mt_converse_mid_conversation_resume():
    print("Testing run_mt_converse mid-conversation abandon + resume...")

    seed = _sample_dynamic_seed("test_mtc_resume", guidance="free", num_turns=4)
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L1_f1", "layer": "layer_1", "text": "checklist item"}]
    )
    interpreter_provider = MockEchoProvider()

    # first pass: user-sim fails on "kor" (turn 1, speaker B) -> abandons after turn 0
    failing_factory = MockUserSimFactory(
        responses={"ind": "utterance-FIRST-PASS 테스트 ind"}, fail_langs={"kor"}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = os.path.join(tmpdir, "seeds.jsonl")
        with open(seeds_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(seed) + "\n")
        output_path = os.path.join(tmpdir, "01_conversed.jsonl")

        mt_stages.run_mt_converse(
            [seeds_path],
            output_path,
            concurrency=1,
            user_sim_factory=failing_factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            checklist_provider=checklist_provider,
        )

        results = read_jsonl(output_path)
        assert sorted(r["turn_index"] for r in results) == [0], "only turn 0 should complete before abandon"
        pending = read_jsonl(output_path + ".pending.jsonl")
        assert len(pending) == 1
        assert pending[0]["turn_index"] == 1
        assert "user-sim" in pending[0]["error"]

        # second pass: fixed factory (kor now works) -> resumes from turn 1, completes turns 1-3
        fixed_factory = MockUserSimFactory(
            responses={"ind": "utterance-SECOND-PASS 테스트 ind", "kor": "발화 테스트 kor"}
        )
        mt_stages.run_mt_converse(
            [seeds_path],
            output_path,
            concurrency=1,
            user_sim_factory=fixed_factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            checklist_provider=checklist_provider,
        )

        final = {r["turn_index"]: r for r in read_jsonl(output_path)}
        assert sorted(final.keys()) == [0, 1, 2, 3]
        # turn 0 must NOT have been regenerated by the second pass (still the first pass's utterance)
        assert final[0]["source_text"] == "utterance-FIRST-PASS 테스트 ind"
        # turn 2 (also speaker A / "ind") IS freshly generated in the second pass
        assert final[2]["source_text"] == "utterance-SECOND-PASS 테스트 ind"
        assert not os.path.exists(output_path + ".pending.jsonl")

    print("  ✓ run_mt_converse mid-conversation resume test passed")


# ---------------------------------------------------------------------------
# Step 7: mt-consolidate + CLI + all chains
# ---------------------------------------------------------------------------
def test_consolidate_conversation_function_id_join():
    print("Testing consolidate_conversation: function_id join respects L3->L2->L1 numbering...")

    turn = {
        "record_id": "conv1_t00_ind-kor",
        "conversation_id": "conv1",
        "model": "gemini-3.1-flash-lite-preview",
        "interpreter": "gemini:gemini-3.1-flash-lite-preview",
        "mode": "scripted",
        "guidance": None,
        "context_mode": "transcript",
        "turn_index": 0,
        "speaker": "A",
        "source_lang": "ind",
        "target_lang": "kor",
        "source_text": "src",
        "translated_text": "tgt",
        "intent": None,
        "success_rate": 2 / 3,
        "completion_rate": "2/3",
        "language_check_passed": True,
        # deliberately unsorted (layer_1 first) — consolidate must re-sort L3->L2->L1
        "checklist_items": [
            {"function_id": "L1_f1", "layer": "layer_1", "text": "item1"},
            {"function_id": "L3_f1", "layer": "layer_3", "text": "item2"},
            {"function_id": "L2_f1", "layer": "layer_2", "text": "item3"},
        ],
        "evaluation": {
            "results": [
                {"id": 1, "criteria": "item2", "met": True, "reasoning": "r1"},
                {"id": 2, "criteria": "item3", "met": False, "reasoning": "r2"},
                {"id": 3, "criteria": "item1", "met": True, "reasoning": "r3"},
            ]
        },
    }
    conv_judge_record = {
        "record_id": "conv1_conv",
        "conversation_id": "conv1",
        "mode": "scripted",
        "guidance": None,
        "lang_a": "ind",
        "lang_b": "kor",
        "conversation_context": "ctx",
        "checklist_provenance": "authored",
        "completion_rate": "1/1",
        "success_rate": 1.0,
        "conversation_checklist_items": [{"function_id": "L2_f9", "layer": "layer_2", "text": "conv item"}],
        "evaluation": {"results": [{"id": 1, "criteria": "conv item", "met": True, "reasoning": "cr"}]},
    }

    conv_line, turn_lines = mt_ops.consolidate_conversation([turn], conv_judge_record)

    assert conv_line["conversation_id"] == "conv1"
    assert conv_line["model"] == "gemini-3.1-flash-lite-preview"
    assert conv_line["checklist_provenance"] == "authored"
    assert conv_line["mean_turn_success_rate"] == 2 / 3
    assert conv_line["conversation_success_rate"] == 1.0
    assert conv_line["all_language_checks_passed"] is True

    turn_criteria = conv_line["turns"][0]["criteria"]
    assert turn_criteria[0] == {
        "function_id": "L3_f1", "layer": "layer_3", "criteria": "item2", "met": True, "reasoning": "r1"
    }
    assert turn_criteria[1]["function_id"] == "L2_f1"
    assert turn_criteria[2]["function_id"] == "L1_f1"

    assert conv_line["conversation_criteria"][0]["function_id"] == "L2_f9"

    assert len(turn_lines) == 1
    assert turn_lines[0]["record_id"] == "conv1_t00_ind-kor"
    assert turn_lines[0]["conversation_id"] == "conv1"
    assert turn_lines[0]["model"] == "gemini-3.1-flash-lite-preview"
    assert turn_lines[0]["criteria"] == turn_criteria

    print("  ✓ consolidate_conversation function_id join test passed")


def test_run_mt_consolidate_end_to_end():
    print("Testing run_mt_consolidate end-to-end...")

    turn_a = {
        "record_id": "conv2_t00_ind-kor", "conversation_id": "conv2",
        "model": "m", "interpreter": "i", "mode": "scripted", "guidance": None, "context_mode": "transcript",
        "turn_index": 0, "speaker": "A", "source_lang": "ind", "target_lang": "kor",
        "source_text": "s0", "translated_text": "t0", "intent": None,
        "success_rate": 1.0, "completion_rate": "1/1", "language_check_passed": True,
        "checklist_items": [{"function_id": "L1_f1", "layer": "layer_1", "text": "x"}],
        "evaluation": {"results": [{"id": 1, "criteria": "x", "met": True, "reasoning": "ok"}]},
    }
    turn_b = dict(turn_a, record_id="conv2_t01_kor-ind", turn_index=1, speaker="B")

    conv_judge = {
        "record_id": "conv2_conv", "conversation_id": "conv2", "mode": "scripted", "guidance": None,
        "lang_a": "ind", "lang_b": "kor", "conversation_context": "ctx", "checklist_provenance": "authored",
        "completion_rate": "1/1", "success_rate": 1.0,
        "conversation_checklist_items": [{"function_id": "L2_f1", "layer": "layer_2", "text": "conv item"}],
        "evaluation": {"results": [{"id": 1, "criteria": "conv item", "met": True, "reasoning": "ok"}]},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        turn_judged_path = os.path.join(tmpdir, "04_turn_judged.jsonl")
        conv_judged_path = os.path.join(tmpdir, "05_conv_judged.jsonl")
        _write_units([turn_a, turn_b], turn_judged_path)
        _write_units([conv_judge], conv_judged_path)

        results_path = os.path.join(tmpdir, "results.jsonl")
        results_turns_path = os.path.join(tmpdir, "results_turns.jsonl")
        mt_stages.run_mt_consolidate(conv_judged_path, turn_judged_path, results_path, results_turns_path)

        results = read_jsonl(results_path)
        assert len(results) == 1
        assert results[0]["num_turns"] == 2

        results_turns = read_jsonl(results_turns_path)
        assert len(results_turns) == 2
        assert sorted(r["turn_index"] for r in results_turns) == [0, 1]

    print("  ✓ run_mt_consolidate end-to-end test passed")


def test_end_to_end_scripted_mock():
    print("Testing full scripted chain end-to-end (mock providers, 2 conversations x 3 turns)...")

    c1 = _sample_scripted_conversation("e2e_mts_0001", num_turns=3)
    c2 = _sample_scripted_conversation("e2e_mts_0002", num_turns=3)

    translate_provider = MockEchoProvider()
    user_sim_factory = MockUserSimFactory()
    judge_provider = MockJudgeProvider(items=[("c1", True), ("c2", True), ("c3", False)])
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L2_f1", "layer": "layer_2", "text": "conv item"}]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "scenarios.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(c1) + "\n")
            f.write(json.dumps(c2) + "\n")

        def p(name):
            return os.path.join(tmpdir, name)

        mt_stages.run_mt_prepare([data_path], p("00_units.jsonl"))
        mt_stages.run_mt_translate(
            p("00_units.jsonl"),
            p("01_translated.jsonl"),
            provider_type="mock",
            model_name="mock-model",
            concurrency=2,
            interpreter_provider=translate_provider,
            interpreter_label="mock:mock-model",
        )
        mt_stages.run_mt_respond(
            p("01_translated.jsonl"), p("02_responded.jsonl"), concurrency=1, user_sim_factory=user_sim_factory
        )
        mt_stages.run_mt_verify(p("02_responded.jsonl"), p("03_verified.jsonl"), glotlid_model=None)
        mt_stages.run_mt_judge_turns(
            p("03_verified.jsonl"), p("04_turn_judged.jsonl"), concurrency=2, judge_provider=judge_provider,
            judge_label="mock:judge",
        )
        mt_stages.run_mt_judge_conv(
            [data_path],
            p("04_turn_judged.jsonl"),
            p("05_conv_judged.jsonl"),
            concurrency=2,
            judge_provider=judge_provider,
            judge_label="mock:judge",
            checklist_provider=checklist_provider,
        )
        mt_stages.run_mt_consolidate(
            p("05_conv_judged.jsonl"), p("04_turn_judged.jsonl"), p("results.jsonl"), p("results_turns.jsonl")
        )

        results = read_jsonl(p("results.jsonl"))
        assert len(results) == 2
        assert all(r["mode"] == "scripted" for r in results)
        assert all(r["num_turns"] == 3 for r in results)

        results_turns = read_jsonl(p("results_turns.jsonl"))
        assert len(results_turns) == 6

        responded = read_jsonl(p("02_responded.jsonl"))
        assert len(responded) == 6
        assert all(r.get("listener_response") for r in responded)

    print("  ✓ full scripted chain end-to-end test passed")


def test_end_to_end_dynamic_mock():
    print("Testing full dynamic chain end-to-end (mock providers, guided + free, 4 turns each)...")

    guided_seed = _sample_dynamic_seed("e2e_mtd_guided", guidance="guided", num_turns=4)
    free_seed = _sample_dynamic_seed("e2e_mtd_free", guidance="free", num_turns=4)

    user_sim_factory = MockUserSimFactory(responses={"ind": "utterance 테스트 ind", "kor": "발화 테스트 kor"})
    interpreter_provider = MockEchoProvider()
    checklist_provider = MockChecklistProvider(
        [{"function_id": "L1_f1", "layer": "layer_1", "text": "checklist item"}]
    )
    judge_provider = MockJudgeProvider(items=[("c1", True)])

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "seeds.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(guided_seed) + "\n")
            f.write(json.dumps(free_seed) + "\n")

        def p(name):
            return os.path.join(tmpdir, name)

        mt_stages.run_mt_converse(
            [data_path],
            p("01_conversed.jsonl"),
            concurrency=2,
            user_sim_factory=user_sim_factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            checklist_provider=checklist_provider,
        )
        mt_stages.run_mt_verify(p("01_conversed.jsonl"), p("03_verified.jsonl"), glotlid_model=None)
        mt_stages.run_mt_judge_turns(
            p("03_verified.jsonl"), p("04_turn_judged.jsonl"), concurrency=2, judge_provider=judge_provider,
            judge_label="mock:judge",
        )
        mt_stages.run_mt_judge_conv(
            [data_path],
            p("04_turn_judged.jsonl"),
            p("05_conv_judged.jsonl"),
            concurrency=2,
            judge_provider=judge_provider,
            judge_label="mock:judge",
            checklist_provider=checklist_provider,
        )
        mt_stages.run_mt_consolidate(
            p("05_conv_judged.jsonl"), p("04_turn_judged.jsonl"), p("results.jsonl"), p("results_turns.jsonl")
        )

        results = {r["conversation_id"]: r for r in read_jsonl(p("results.jsonl"))}
        assert len(results) == 2
        assert results["e2e_mtd_guided"]["checklist_provenance"] == "authored"
        assert results["e2e_mtd_free"]["checklist_provenance"] == "posthoc"
        assert all(r["mode"] == "dynamic" for r in results.values())

        results_turns = read_jsonl(p("results_turns.jsonl"))
        assert len(results_turns) == 8

    print("  ✓ full dynamic chain end-to-end test passed")


# ---------------------------------------------------------------------------
# Step 8: Batch support
# ---------------------------------------------------------------------------
def test_run_mt_translate_batch_per_wave():
    print("Testing run_mt_translate batch backend: per-wave sidecar + history threading...")

    conversation = _sample_scripted_conversation("test_mtb_0001", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)

    prompts_seen = []

    def responder(req):
        prompts_seen.append(req.prompt)
        return f"XLAT-{len(prompts_seen)}-테스트"

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)
        output_path = os.path.join(tmpdir, "01_translated.jsonl")

        fake_client = FakeBatchClient(responder=responder, complete_after_polls=0)
        mt_stages.run_mt_translate(
            units_path,
            output_path,
            provider_type="gemini",
            model_name="mock-model",
            thinking_level="high",
            context_mode="transcript",
            backend="batch",
            batch_client=fake_client,
        )

        results = read_jsonl(output_path)
        assert len(results) == 3
        by_turn = {r["turn_index"]: r for r in results}
        assert by_turn[0]["history"] == []
        assert "XLAT-1-테스트" in prompts_seen[1]  # turn 1's prompt includes turn 0's translation
        assert "XLAT-1-테스트" in prompts_seen[2] and "XLAT-2-테스트" in prompts_seen[2]

        for k in range(3):
            assert not os.path.exists(output_path + f".batch.t{k:02d}.json")
        assert not os.path.exists(output_path + ".pending.jsonl")

    print("  ✓ run_mt_translate batch per-wave test passed")


def test_run_mt_translate_aya_retries_wrong_script():
    print("Testing Aya wrong-script output recovery...")

    conversation = _sample_scripted_conversation("test_mta_0001", num_turns=1)
    units = mt_ops.conversation_to_turn_units(conversation)

    class MockAyaWrongScriptProvider:
        def __init__(self):
            self.retry_prompts = []

        def generate_batch(self, requests):
            return ["This answer is not in Korean" for _ in requests]

        def generate(self, prompt, system_prompt=None):
            self.retry_prompts.append(prompt)
            if len(self.retry_prompts) == 1:
                return "Still not a Korean translation"
            return "한국어 번역"

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)
        output_path = os.path.join(tmpdir, "01_translated.jsonl")
        provider = MockAyaWrongScriptProvider()

        mt_stages.run_mt_translate(
            units_path,
            output_path,
            provider_type="aya",
            model_name="mock-aya",
            context_mode="transcript",
            interpreter_provider=provider,
            interpreter_label="aya:mock-aya",
        )

        results = read_jsonl(output_path)
        assert len(results) == 1
        assert results[0]["translated_text"] == "한국어 번역"
        assert len(provider.retry_prompts) == 2
        assert "Korean (kor)" in provider.retry_prompts[0]
        assert "한국어 번역:" in provider.retry_prompts[1]
        assert not os.path.exists(output_path + ".pending.jsonl")

    print("  ✓ Aya wrong-script output recovery test passed")


def test_run_mt_translate_batch_no_wait_then_collect():
    print("Testing run_mt_translate batch backend: --no-batch-wait submit-then-collect...")

    conversation = _sample_scripted_conversation("test_mtb_0002", num_turns=2)
    units = mt_ops.conversation_to_turn_units(conversation)

    def responder(req):
        return f"XLAT-테스트-{req.custom_id[-4:]}"

    with tempfile.TemporaryDirectory() as tmpdir:
        units_path = os.path.join(tmpdir, "00_units.jsonl")
        _write_units(units, units_path)
        output_path = os.path.join(tmpdir, "01_translated.jsonl")

        fake_client = FakeBatchClient(responder=responder, complete_after_polls=0)
        mt_stages.run_mt_translate(
            units_path,
            output_path,
            provider_type="gemini",
            model_name="mock-model",
            context_mode="transcript",
            backend="batch",
            batch_client=fake_client,
            batch_wait=False,
        )
        assert read_jsonl(output_path) == []
        assert os.path.exists(output_path + ".batch.t00.json")

        # re-run, now waiting -> collects wave 0, then submits+collects wave 1
        mt_stages.run_mt_translate(
            units_path,
            output_path,
            provider_type="gemini",
            model_name="mock-model",
            context_mode="transcript",
            backend="batch",
            batch_client=fake_client,
            batch_wait=True,
        )
        results = read_jsonl(output_path)
        assert len(results) == 2
        assert not os.path.exists(output_path + ".batch.t00.json")
        assert not os.path.exists(output_path + ".batch.t01.json")

    print("  ✓ run_mt_translate batch no-wait-then-collect test passed")


def test_run_mt_judge_turns_batch():
    print("Testing run_mt_judge_turns batch backend (single job)...")

    conversation = _sample_scripted_conversation("test_mtjb_0001", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)
    for u in units:
        u["translated_text"] = f"translated {u['turn_index']}"
        u["history"] = [
            {**h, "translated_text": f"translated {h['turn_index']}"} for h in u.pop("authored_history", [])
        ]

    def responder(req):
        return json.dumps({"results": [{"id": 1, "criteria": "c1", "met": True, "reasoning": "ok"}]})

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "03_verified.jsonl")
        _write_units(units, input_path)
        output_path = os.path.join(tmpdir, "04_turn_judged.jsonl")

        fake_client = FakeBatchClient(responder=responder, complete_after_polls=0)
        mt_stages.run_mt_judge_turns(
            input_path, output_path, model_name="mock-model", backend="batch", batch_client=fake_client
        )

        results = read_jsonl(output_path)
        assert len(results) == 3
        assert all(r["evaluation"]["results"][0]["met"] for r in results)
        assert not os.path.exists(output_path + ".batch.json")

    print("  ✓ run_mt_judge_turns batch test passed")


def test_run_mt_judge_turns_batch_rejects_judge_history():
    print("Testing run_mt_judge_turns batch + --judge-history raises (D5: sync-only)...")

    try:
        mt_stages.run_mt_judge_turns(
            "unused.jsonl",
            "unused_out.jsonl",
            backend="batch",
            judge_history=True,
            batch_client=FakeBatchClient(responder=lambda r: "{}"),
        )
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass

    print("  ✓ run_mt_judge_turns batch+judge-history rejection test passed")


def test_run_mt_judge_conv_batch():
    print("Testing run_mt_judge_conv batch backend (single job, posthoc checklist stays sync)...")

    scripted_conv = _sample_scripted_conversation("test_mtjcb_scripted", num_turns=2)
    free_conv = {
        "conversation_id": "test_mtjcb_free",
        "mode": "dynamic",
        "guidance": "free",
        "lang_a": "ind",
        "lang_b": "kor",
        "conversation_context": "ctx",
        "conversation_checklist_items": None,
        "conversation_verification_prompt": None,
    }

    turn_judged = []
    for i, speaker in enumerate(["A", "B"]):
        target = "kor" if speaker == "A" else "ind"
        for cid in ("test_mtjcb_scripted", "test_mtjcb_free"):
            turn_judged.append(
                {
                    "record_id": f"{cid}_t{i:02d}_x",
                    "conversation_id": cid,
                    "turn_index": i,
                    "speaker": speaker,
                    "source_text": f"src {i}",
                    "translated_text": f"tgt {i}",
                    "target_lang": target,
                    "language_check_passed": True,
                }
            )

    checklist_provider = MockChecklistProvider(
        [{"function_id": "L2_f9", "layer": "layer_2", "text": "posthoc conv item"}]
    )

    def responder(req):
        return json.dumps({"results": [{"id": 1, "criteria": "conv item", "met": True, "reasoning": "ok"}]})

    with tempfile.TemporaryDirectory() as tmpdir:
        scenarios_path = os.path.join(tmpdir, "scenarios.jsonl")
        with open(scenarios_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(scripted_conv) + "\n")
            f.write(json.dumps(free_conv) + "\n")

        turn_judged_path = os.path.join(tmpdir, "04_turn_judged.jsonl")
        _write_units(turn_judged, turn_judged_path)

        output_path = os.path.join(tmpdir, "05_conv_judged.jsonl")
        fake_client = FakeBatchClient(responder=responder, complete_after_polls=0)
        mt_stages.run_mt_judge_conv(
            [scenarios_path],
            turn_judged_path,
            output_path,
            judge_model="mock-model",
            checklist_provider=checklist_provider,
            backend="batch",
            batch_client=fake_client,
        )

        results = {r["conversation_id"]: r for r in read_jsonl(output_path)}
        assert results["test_mtjcb_scripted"]["evaluation"] is not None
        free_result = results["test_mtjcb_free"]
        assert free_result["checklist_provenance"] == "posthoc"
        assert free_result["evaluation"] is not None
        assert not os.path.exists(output_path + ".batch.json")

    print("  ✓ run_mt_judge_conv batch test passed")


def test_run_scripted_batch_end_to_end():
    print("Testing run_scripted_batch (generator's batch checklist backend, plan Step 8)...")

    transcript_json = json.dumps(
        {
            "conversation_context": "Two colleagues discuss a delayed project.",
            "user_a_context": "Anda seorang staf.",
            "user_b_context": "당신은 팀장입니다.",
            "turns": [
                {"turn_index": 0, "speaker": "A", "text": "turn0 text"},
                {"turn_index": 1, "speaker": "B", "text": "turn1 text"},
                {"turn_index": 2, "speaker": "A", "text": "turn2 text"},
                {"turn_index": 3, "speaker": "B", "text": "turn3 text"},
            ],
        }
    )

    def checklist_responder(req):
        return json.dumps(
            {
                "items": [
                    {"function_id": "L1_f1", "layer": "layer_1", "text": "item 1"},
                    {"function_id": "L2_f1", "layer": "layer_2", "text": "item 2"},
                    {"function_id": "L3_f1", "layer": "layer_3", "text": "item 3"},
                ]
            }
        )

    # run_scripted_batch now batches BOTH the transcript job (Phase 1) and the
    # checklist job (Phase 2) via build_batch_client — first call gets a
    # transcript-shaped mock client, the next gets a checklist-shaped one.
    transcript_client = FakeBatchClient(responder=lambda req: transcript_json, complete_after_polls=0)
    checklist_client = FakeBatchClient(responder=checklist_responder, complete_after_polls=0)
    call_count = {"n": 0}

    def fake_build_batch_client(*a, **kw):
        call_count["n"] += 1
        return transcript_client if call_count["n"] == 1 else checklist_client

    orig_build_batch_client = gen.build_batch_client
    gen.build_batch_client = fake_build_batch_client

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                pair="id-ko",
                num_scenarios=2,
                num_turns=4,
                seed_xlsx=None,
                seed_split="test_proverbs",
                start_row=1,
                no_seed=True,
                topic_hints="topic a,topic b",
                output_dir=tmpdir,
                tag="batchtest",
                no_function_grounding=False,
                transcript_provider="gemini",
                transcript_model="mock-model",
                transcript_thinking="high",
                checklist_provider="gemini",
                checklist_model="mock-model",
                checklist_thinking="high",
                poll_interval=0.01,
                dry_run=False,
                verify_language=False,
            )
            output_path = gen.run_scripted_batch(args)
            results = read_jsonl(output_path)
            assert len(results) == 2
            for r in results:
                assert len(r["turns"]) == 4
                assert all(t["checklist_items"] for t in r["turns"])
                assert r["conversation_checklist_items"]
                assert r["generation_metadata"]["checklist_backend"] == "batch"
    finally:
        gen.build_batch_client = orig_build_batch_client

    print("  ✓ run_scripted_batch end-to-end test passed")


# ---------------------------------------------------------------------------
# Step 9 optimization: defer + batch dynamic mode's per-turn checklist-gen
# ---------------------------------------------------------------------------
def test_converse_next_turn_skip_checklist():
    print("Testing converse_next_turn skip_checklist=True (deferred checklist-gen)...")

    seed = _sample_dynamic_seed(guidance="free", num_turns=4)
    factory = MockUserSimFactory(responses={"ind": "Halo 테스트"})
    interpreter_provider = MockEchoProvider()

    out = mt_ops.converse_next_turn(
        seed,
        0,
        [],
        factory,
        None,  # checklist_provider unused/None when skip_checklist=True
        interpreter_provider,
        "mock:interpreter",
        "mock-model",
        {},
        skip_checklist=True,
    )
    assert out["checklist_items"] == []
    assert out["verification_prompt"] is None
    assert out["translated_text"] is not None  # translation still happens

    print("  ✓ converse_next_turn skip_checklist test passed")


def test_run_mt_converse_defer_checklist():
    print("Testing run_mt_converse defer_checklist=True (no checklist_provider needed)...")

    seed = _sample_dynamic_seed("test_mtdc_0001", guidance="free", num_turns=4)
    factory = MockUserSimFactory(responses={"ind": "utterance 테스트 ind", "kor": "발화 테스트 kor"})
    interpreter_provider = MockEchoProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = os.path.join(tmpdir, "seeds.jsonl")
        with open(seeds_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(seed) + "\n")
        output_path = os.path.join(tmpdir, "01_conversed.jsonl")

        mt_stages.run_mt_converse(
            [seeds_path],
            output_path,
            concurrency=1,
            user_sim_factory=factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            defer_checklist=True,
        )

        results = read_jsonl(output_path)
        assert len(results) == 4
        assert all(r["checklist_items"] == [] for r in results)
        assert all(r["verification_prompt"] is None for r in results)
        assert all(r["translated_text"] for r in results)

    print("  ✓ run_mt_converse defer_checklist test passed")


def test_run_mt_checklist_batch_sync_and_batch():
    print("Testing run_mt_checklist_batch: sync path + batch path fill in checklists post-hoc...")

    conversation = _sample_scripted_conversation("test_mtcb_0001", num_turns=3)
    units = mt_ops.conversation_to_turn_units(conversation)
    turns = []
    for u in units:
        u["translated_text"] = f"translated {u['turn_index']}"
        u["history"] = [
            {**h, "translated_text": f"translated {h['turn_index']}"} for h in u.pop("authored_history", [])
        ]
        u["checklist_items"] = []
        u["verification_prompt"] = None
        turns.append(u)

    # sync path — floor-satisfying (1 per layer), text genuinely distinct so
    # the always-on dedup doesn't collapse them.
    checklist_provider = MockChecklistProvider(
        [
            {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation accurately convey the specific date mentioned?"},
            {"function_id": "L2_f1", "layer": "layer_2", "text": "Does the translation preserve the speaker's tentative, hedged framing?"},
            {"function_id": "L3_f1", "layer": "layer_3", "text": "Does the translation reflect the culturally appropriate level of formality?"},
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "01_conversed.jsonl")
        _write_units(turns, input_path)
        output_path = os.path.join(tmpdir, "checklist_filled.jsonl")

        mt_stages.run_mt_checklist_batch(
            input_path, output_path, concurrency=1, backend="sync", checklist_provider=checklist_provider
        )
        results = read_jsonl(output_path)
        assert len(results) == 3
        assert all(r["checklist_items"] for r in results)
        assert all(r["verification_prompt"] for r in results)

    # batch path — floor-satisfying (1+ per layer; this mock previously had
    # no layer_1 item, which already failed the floor check that stages.py's
    # _apply_checklist_batch_response applies) and text genuinely distinct so
    # the always-on dedup (now also applied in parse_checklist_batch_response)
    # doesn't collapse them.
    def responder(req):
        # order matters: index 0 must stay "L2_f1" (checked below); 1 item
        # per layer keeps the floor + priority rule satisfied (1 >= 1 >= 1).
        return json.dumps(
            {
                "items": [
                    {"function_id": "L2_f1", "layer": "layer_2", "text": "Does the translation convey the speaker's polite request framing?"},
                    {"function_id": "L1_f1", "layer": "layer_1", "text": "Does the translation accurately convey the specific time mentioned?"},
                    {"function_id": "L3_f2", "layer": "layer_3", "text": "Does the translation reflect the appropriate honorific register?"},
                ]
            }
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "01_conversed.jsonl")
        _write_units(turns, input_path)
        output_path = os.path.join(tmpdir, "checklist_filled.jsonl")

        fake_client = FakeBatchClient(responder=responder, complete_after_polls=0)
        mt_stages.run_mt_checklist_batch(
            input_path, output_path, backend="batch", batch_client=fake_client, checklist_model="mock-model"
        )
        results = read_jsonl(output_path)
        assert len(results) == 3
        assert all(r["checklist_items"][0]["function_id"] == "L2_f1" for r in results)
        assert not os.path.exists(output_path + ".batch.json")

    print("  ✓ run_mt_checklist_batch sync + batch tests passed")


def test_defer_checklist_end_to_end():
    print("Testing full deferred-checklist dynamic flow: converse(defer) -> checklist-batch -> verify -> judge...")

    seed = _sample_dynamic_seed("test_mtdce2e_0001", guidance="free", num_turns=4)
    factory = MockUserSimFactory(responses={"ind": "utterance 테스트 ind", "kor": "발화 테스트 kor"})
    interpreter_provider = MockEchoProvider()

    def checklist_responder(req):
        return json.dumps(
            {
                "items": [
                    {"function_id": "L1_f1", "layer": "layer_1", "text": "item 1"},
                    {"function_id": "L2_f1", "layer": "layer_2", "text": "item 2"},
                    {"function_id": "L3_f1", "layer": "layer_3", "text": "item 3"},
                ]
            }
        )

    def judge_responder(req):
        return json.dumps({"results": [{"id": 1, "criteria": "c1", "met": True, "reasoning": "ok"}]})

    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = os.path.join(tmpdir, "seeds.jsonl")
        with open(seeds_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(seed) + "\n")

        def p(name):
            return os.path.join(tmpdir, name)

        mt_stages.run_mt_converse(
            [seeds_path],
            p("01_conversed.jsonl"),
            concurrency=1,
            user_sim_factory=factory,
            interpreter_provider=interpreter_provider,
            interpreter_label="mock:interpreter",
            defer_checklist=True,
        )

        checklist_client = FakeBatchClient(responder=checklist_responder, complete_after_polls=0)
        mt_stages.run_mt_checklist_batch(
            p("01_conversed.jsonl"), p("02_checklisted.jsonl"), backend="batch", batch_client=checklist_client
        )
        results = read_jsonl(p("02_checklisted.jsonl"))
        assert all(r["checklist_items"] for r in results)

        mt_stages.run_mt_verify(p("02_checklisted.jsonl"), p("03_verified.jsonl"), glotlid_model=None)

        judge_client = FakeBatchClient(responder=judge_responder, complete_after_polls=0)
        mt_stages.run_mt_judge_turns(
            p("03_verified.jsonl"), p("04_turn_judged.jsonl"), backend="batch", batch_client=judge_client
        )
        judged = read_jsonl(p("04_turn_judged.jsonl"))
        assert len(judged) == 4
        assert all(r["evaluation"] is not None for r in judged)

    print("  ✓ full deferred-checklist end-to-end test passed")


def run_all_tests():
    print("\n" + "=" * 80)
    print("Running Multi-Turn Expansion Tests")
    print("=" * 80 + "\n")

    tests = [
        test_taxonomy_loading,
        test_checklist_generation_and_cap_enforcement,
        test_verification_prompt_composition,
        test_validate_num_turns,
        test_validate_alternation,
        test_generate_one_scripted_scenario_mock,
        test_generate_one_dynamic_seed_mock,
        test_conversation_to_turn_units_record_id_uniqueness,
        test_conversation_to_turn_units_history_correctness,
        test_conversation_to_turn_units_listener_side_switching,
        test_conversation_to_turn_units_checklist_passthrough,
        test_run_mt_prepare_end_to_end,
        test_run_mt_translate_history_injection_and_context_mode,
        test_run_mt_translate_resume_after_deletion,
        test_run_mt_translate_pending_sidecar_withholds_successors,
        test_build_turn_respond_history_text,
        test_respond_turn_record_success,
        test_respond_turn_record_no_translation,
        test_respond_turn_record_unconfigured_language,
        test_run_mt_respond_end_to_end,
        test_verify_turn_record_with_real_glotlid,
        test_build_turn_judge_prompt_includes_transcript,
        test_judge_turn_record_mock,
        test_run_mt_judge_turns_default_flat_no_history_slot,
        test_run_mt_judge_turns_with_judge_history,
        test_ensure_conversation_checklist_authored_vs_posthoc,
        test_build_conversation_judge_prompt_failed_turns_note,
        test_judge_conversation_record_mock,
        test_run_mt_judge_conv_end_to_end,
        test_render_history_for_side,
        test_build_user_turn_prompt_guided_vs_free,
        test_converse_next_turn_mock_success,
        test_converse_next_turn_user_sim_failure,
        test_run_mt_converse_end_to_end_guided_and_free,
        test_run_mt_converse_mid_conversation_resume,
        test_consolidate_conversation_function_id_join,
        test_run_mt_consolidate_end_to_end,
        test_end_to_end_scripted_mock,
        test_end_to_end_dynamic_mock,
        test_run_mt_translate_batch_per_wave,
        test_run_mt_translate_batch_no_wait_then_collect,
        test_run_mt_judge_turns_batch,
        test_run_mt_judge_turns_batch_rejects_judge_history,
        test_run_mt_judge_conv_batch,
        test_run_scripted_batch_end_to_end,
        test_converse_next_turn_skip_checklist,
        test_run_mt_converse_defer_checklist,
        test_run_mt_checklist_batch_sync_and_batch,
        test_defer_checklist_end_to_end,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
