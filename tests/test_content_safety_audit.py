import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _content_safety import classify_texts, iter_matches, normalize_text  # noqa: E402
from _opensubs_scoring import step1_filter  # noqa: E402


def rules(text: str) -> set[str]:
    return {rule.name for rule, _ in iter_matches(text)}


def test_detects_direct_pii_without_flagging_plain_numbers() -> None:
    assert "email_address" in rules("Contact jane.doe@example.org now")
    assert "ipv4_address" in rules("server 192.168.1.8")
    assert "payment_card_candidate" in rules("card 4111 1111 1111 1111")
    assert "phone_candidate" in rules("call +82 (10) 1234-5678")
    assert "phone_candidate" not in rules("There were 1234567 people in the story")
    assert "payment_card_candidate" not in rules("number 1234 5678 9012 3456")
    assert "government_id_with_label" in rules("NIK: 3173051209870001")
    assert "government_id_with_label" not in rules("pernikahan dan teknik yang menarik")
    assert "government_id_with_label" not in rules("passport invalid")


def test_detects_supported_offensive_languages() -> None:
    assert "explicit_offensive_en" in rules("What the fuck?")
    assert "explicit_offensive_ar" in rules("يا ابن الكلب")
    assert "explicit_offensive_bn" in rules("ওই হারামজাদা")
    assert "explicit_offensive_id" in rules("dasar bangsat")
    assert "explicit_offensive_ko" in rules("이 개새끼야")


def test_ordinary_insults_block_but_literal_terms_are_review_severity() -> None:
    matches = [(rule.name, rule.severity) for rule, _ in iter_matches("anjing bodoh")]
    assert matches == [
        ("ordinary_offensive_id", "block"),
        ("review_offensive_id", "review"),
    ]
    assert ("ordinary_offensive_bn", "block") in [
        (rule.name, rule.severity) for rule, _ in iter_matches("তুমি বোকা")
    ]
    assert ("ordinary_offensive_ko", "block") in [
        (rule.name, rule.severity) for rule, _ in iter_matches("닥쳐")
    ]


def test_word_boundaries_reduce_substring_false_positives() -> None:
    assert "explicit_offensive_id" not in rules("kecokelatan")
    assert "ordinary_offensive_en" not in rules("damnation")
    assert "review_offensive_bn" not in rules("বাল্বটা")


def test_arabic_diacritics_and_tatweel_are_normalized() -> None:
    assert normalize_text("كـِس أُمك") == "كس أمك"


def test_review_matches_remain_allowed() -> None:
    decision = classify_texts(("anjing", "call +82 (10) 1234-5678"))
    assert not decision.blocked
    assert decision.review_only


def test_opensubs_hard_filter_rejects_blocking_content() -> None:
    row = {
        "source_text": "Please email jane.doe@example.org now",
        "target_text": "지금 이메일을 보내 주세요",
        "source_lang": "en",
        "target_lang": "kor",
    }
    passed, reason, metrics = step1_filter(row)
    assert not passed
    assert reason == "content_safety_block:email_address"
    assert metrics == {}
