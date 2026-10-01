# Human Annotation Guideline

This is the full annotation guideline (version 2.5) given to the native-speaker annotators in
the human calibration study of [*Evaluating Communicative Success in Machine-Translated
Conversation*](https://arxiv.org/abs/2609.19885) (Appendix E of the paper). The protocol received
institutional review board approval at KAIST. Participation was voluntary, annotators gave
informed consent, and each annotator received 50,000 KRW for at most 5 hours of work.

Annotators worked in one spreadsheet per person with two sheets, Task A (checklist function
quality) and Task B (judge calibration).

## Overview

You will evaluate translations produced by AI interpreter agents that translate conversations
between people speaking different languages. You only need to read the **target language**
(your native language). You do **not** need to know the source language.

This round is organized around **evaluation functions** — recurring things a good translation
must do (e.g. "use the right level of politeness", "soften a harsh command", "choose a pronoun
that fits the relationship"). The same function shows up across many different scenarios. By
judging a handful of examples of each function, your work is automatically extended to hundreds
of scenarios you never have to read.

Each example comes from a real film/TV subtitle wrapped with an AI-generated conversation
context. Ten different interpreter models are evaluated — which model produced any given
example is **hidden** while you answer (see Task B below), so your judgment isn't influenced by
which system you think you're looking at.

---

## Your file: two sheets

| Sheet | Task | What you do |
|---|---|---|
| **Task_A** | Function Quality | Rate whether each function is a meaningful, fair thing to check — **do this first** |
| **Task_B** | Judge Calibration | For each example, say whether the criterion is actually satisfied in the translation |

Complete **Task_A first**, then **Task_B**.

---

## About the reference & context (please read once)

- **The reference translation is NOT the correct answer.** It is a subtitle translation,
  often shortened or freely adapted. Use it only to recognise natural target-language wording.
- **The context is partly AI-generated** around a real utterance. It may feel over-specified.
- The **"Context / In English"** field tells you what the source utterance means — rely on it;
  you never need to read the source text.

---

## Task A — Function Quality *(do this first)*

Each row is one **function** (a yes/no question describing what the translation should do).
Hover the **Function** cell to see **example criteria** (6 similar-sounding phrasings) and
**example situations** (3 real cases this function was checked against, each with a short
English context — never the source-language text or the translation). The example situations
are deliberately picked to be *different from each other* — use them to judge A2 and A4 below;
if the 2-3 situations look like completely unrelated topics, that's a sign the function is
situation-specific or mixed, not general, even if its criteria sound alike.

### Step 0 — Calibration Round (do this before rating the full sheet)

Before rating your full set of 53 functions, rate these 8 anchor functions on your own, then
compare against the reference answers and reasoning below. If more than 1-2 of your answers are
off by more than 1 point (A1) or land on a different category than the reference reasoning would
support (A4), re-read the worked reasoning before continuing — the point isn't to match the
reference exactly, it's to catch a systematically different reading of the scale early, before
it colors all 53 answers the same way.

| # | Function | Reference A1 | Reasoning | Reference A4 | Reasoning |
|---|---|---|---|---|---|
| 1 | Does the translation accurately convey the intended meaning and truth of the statement? | 5 | Core semantic fidelity, applies to literally every utterance — as important as it gets. | General | No scenario-specific content in the wording at all. |
| 2 | Are the address terms used in the translation appropriate for the speakers' relationship? | 5 | Getting this wrong actively misrepresents the relationship — a real, checkable failure mode. | General | Applies whenever there's a relationship between speakers, i.e. almost always. |
| 3 | Does the translation accurately convey the speaker's refusal or denial of a claim? | 4 | Important when it applies, but narrower in scope than #1 — only relevant to refusal/denial turns. | Situation-specific | Only meaningful when the turn actually contains a refusal or denial. |
| 4 | Does the translation use natural, culturally appropriate phrasing instead of stiff, literal words? | 4 | A real, recurring failure mode (literal-translation-ese), not redundant with #1. | General | The check itself (naturalness) applies to any utterance, even though *what* counts as natural varies by scenario. |
| 5 | Does the translation correctly frame the speaker's intent as a formal or direct statement? | 3 | Useful but narrower and closer to a phrasing detail than a communicative failure. | Situation-specific | Formal-vs-direct framing is only a live question in specific registers. |
| 6 | Does the translation capture the speaker's emotional tone without sounding unnatural? | *(rate independently — real annotators split 2/3/4 on this one)* | This combines two things: "captures emotional tone" (situation-specific — depends on what emotion is present) and "without sounding unnatural" (general — a naturalness check that applies to everything). | **Mixed** | This is the canonical Mixed case: the function bundles a general naturalness check with a situation-dependent emotional-content check. If your instinct is General or Situation-specific here, re-read the Mixed definition below before continuing. |
| 7 | Does the translation reflect the speaker's emotional state and intent accurately? | *(rate independently — real annotators split 1/3/5 on this one)* | Wave-1's single widest-disagreement function. There's no single right answer here; the point is to notice *why* you land where you do (is "accurately" doing a lot of work for you? does "intent" feel redundant with other functions you've seen?) and to calibrate against your co-annotators' reasoning if you're annotating alongside others for the same language. | — | — |
| 8 | Does the translation use culturally appropriate language to avoid causing unnecessary offense or shame? | — | — | *(rate independently — real annotators split Mixed/Situation-specific/General on this one)* | Ask yourself: could this exact wording apply to a totally unrelated scenario (offense/shame is always a live risk in *some* form) — if yes, that pulls toward General; does what counts as "culturally appropriate" here depend heavily on the specific relationship/context in view — if yes, that pulls toward Situation-specific or Mixed. Either answer is defensible; what matters is applying the same reasoning consistently across the rest of your sheet. |

For each function answer:

- **A1 Meaningful? (1–5)** — Is this a meaningful, important thing to check in the translation?
  Use the anchors above to calibrate your scale, not just the labels:
  `1` = pointless, redundant with another function you've already seen, or not actually checkable
  · `2` = marginally useful, but weakly phrased or very narrow · `3` = somewhat useful, a
  reasonable but not critical check (anchor #5 above) · `4` = a real, recurring, important check
  (anchors #3, #4) · `5` = clearly important, core to communicative success (anchors #1, #2). Try
  to use the full range across your 53 functions rather than clustering near one end — if you
  find yourself rating almost everything 4 or 5, stop and ask whether you're rating "is this a
  coherently-written question" (almost always yes, since these already passed an automated
  filter) rather than "is this worth the space it takes in the checklist relative to the other
  52" (the actual question).
- **A2 Answerable from target + context gloss?** — Could you judge this using the translation
  and the short English context gloss you'll see in Task B, *without* needing the actual source
  text? `Target only` = yes · `Needs source` = you'd have to read the original source text to
  judge it.
- **A3 Notes / what's missing** *(optional)* — anything poorly worded, or an important thing
  this function fails to capture. One short sentence.
- **A4 Generality** — Does this function apply broadly to almost any text, or only to specific
  kinds of situations? `General` = the check itself makes sense for nearly any utterance, even if
  what counts as satisfying it varies by scenario (anchors #1, #2, #4) · `Situation-specific` = it
  only really applies to a particular kind of scenario or phrasing (anchors #3, #5) ·
  `Mixed` = the function bundles a general check with a situation-dependent one, like anchor #6 —
  this is a real, common category, not a fallback for "I'm not sure": if you're tempted to treat
  Mixed as a last resort, re-read anchor #6 and ask whether the function you're rating has that
  same bundled structure. There's no wrong answer for genuinely ambiguous cases — we're using
  this to understand which functions are broad rules-of-thumb versus narrower, context-dependent
  checks, and a well-reasoned Mixed answer is exactly as valid as a confident General or
  Situation-specific one.

You do **not** see any translations in Task A — that's intentional, so your view of the
checklist isn't biased by how well a particular model did.

---

## Task B — Judge Calibration

Rows are **grouped by function**, so you judge the same kind of thing several times in a row.
Each row shows: the **Function**, the specific **Criterion**, a short **Context** gloss, and the
**Translation** in your language. **Which model produced the translation is hidden** — it's in
the same reveal-after group as the judge's verdict (see below).

For each row:

- **★ Satisfied? (YES / NO)** — Reading the translation, is *this criterion* actually satisfied?
  - `YES` = the translation does what the criterion asks.
  - `NO` = it does not.
- **Confidence (Sure / Unsure)** — Mark `Unsure` only if you genuinely cannot tell without
  seeing the source language. `Unsure` rows are set aside (not counted), but they tell us where
  judging needs bilingual knowledge — so don't guess, mark `Unsure`. **`Unsure` is not a hedge on
  a negative answer.** If you're fairly (not perfectly) confident the criterion is *not*
  satisfied, that's still `Satisfied? = NO` with `Confidence = Sure` — normal, everyday judgment
  calls should be `Sure` regardless of which way they go. Reserve `Unsure` specifically for "I
  cannot make this call at all without reading the source text," not for "I'm leaning NO but not
  100% certain." (In wave 1, one annotator marked every single NO as `Unsure` and every `Sure`
  answer YES — if that's happening, it's this distinction, not your judgment, that needs
  re-reading.)

> **Important — answer before peeking.** The interpreter **model name**, the AI judge's own
> verdict, and its reasoning are all kept in **hidden columns on the far right** of Task_B.
> Answer **★ Satisfied?** from your own reading first. You may un-hide those columns afterward
> out of curiosity, but your answer must be your independent judgment — we are measuring
> whether the AI judge agrees with *you*, not the other way around, and we don't want which
> model you think you're looking at to color that.

---

## Important Notes

1. **Task_A before Task_B.**
2. **Answer Task_B from the translation alone, before revealing the judge's verdict.**
3. **You judge the target language only** — the Context / In English gloss tells you the meaning.
4. **The reference is not the answer** — it's only for recognising natural wording.
5. **`Unsure` is a valid answer** when you'd need the source language. Don't guess.
6. **Functions repeat** — once you've understood a function, the following rows of the same
   function go quickly. Stay consistent within a function.

---

## Volume and Compensation

- **Task A:** 53 functions (one quick rating each) — same 53 for all 3 annotators of your
  language.
- **Task B:** ~295–335 examples (your individual sheet; final count set per language after a
  timing pilot — see below), grouped by function, spread across your 3 source directions and
  all 10 interpreter models.
- **Estimated time:** to be confirmed by a short pilot before you start the full sheet: you'll
  first do a **10-row timing pilot** and self-time it. At ≤ ~30 s/row, your sheet takes roughly
  **4.5–5.5 hours** total (Task A + Task B); if you're slower than that, the sheet will be
  trimmed down first so it still fits the 5-hour cap.
- **Compensation:** up to 50,000 KRW (10,000 KRW/h × 5h cap).

---

## Briefing Checklist

- [ ] Do the **Step 0 Calibration Round** (8 anchor functions) before rating your full Task A sheet.
- [ ] There are **two sheets**: finish Task_A before Task_B.
- [ ] The reference translation is **not ground truth**.
- [ ] You judge **target-language output only** — you don't read the source.
- [ ] In Task_B, answer **★ Satisfied?** from your own reading **before** un-hiding the model name or the judge's verdict.
- [ ] Marking **Unsure** is correct when you'd need the source language — it is **not** a hedge on a negative answer; "leaning NO but not 100% sure" is still `NO` + `Sure`.
- [ ] Rows are grouped by function — judge the same kind of thing consistently.
- [ ] Task A also has an **A4 Generality** rating (General/Situation-specific/Mixed) — **Mixed is a real category**, not a last resort; see the calibration anchors.
- [ ] In Task A, hover the Function cell for **example criteria and example situations** — use the situations to judge A2/A4.
- [ ] Try to **use the full 1-5 range** on A1 Meaningful rather than clustering near one end.

---

## Questions or Issues

If an example seems broken (context clearly wrong, translation in the wrong language), note it
in **A3 Notes** for that function (or leave the row `Unsure`) and continue.
