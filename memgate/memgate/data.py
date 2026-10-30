"""Synthetic multi-session dataset for Step 0.

This is a SANITY HARNESS, not a result. It exists so the pipeline and metrics
can be validated with zero downloads. Phase 1 replaces it with LoCoMo /
LongMemEval via `load_locomo()`.

Design goals (so the experiment is actually informative):
  * many durable facts, so at small budgets a policy MUST choose between them
    -> produces a real accuracy-per-token curve rather than a flat line
  * facts planted across all sessions and probed at the end, so recency alone
    recovers only the most recent few -> P0 gets partial, not zero, credit
  * heavy filler and generic chatter, the realistic signal-to-noise case
  * one fact update, to exercise supersession
"""
import ast
import json
import os
import random
import re
from typing import List, Tuple
from .types import Turn, Question
from .utils import answer_tokens

# --- fact templates: (category, statement, probe) ---------------------------
TEMPLATES = [
    ("project", "The {thing} is a {adj} app for {who}, we called it {name}.",
     "What did we call the {thing}?"),
    ("deadline", "The deadline for the {thing} is {month} {day}.",
     "When is the deadline for the {thing}?"),
    ("budget", "My budget for the {thing} is {num} rupees.",
     "What is my budget for the {thing}?"),
    ("decision", "I decided to use {tech} for the {thing} instead of {alt}.",
     "Which technology did I choose for the {thing}?"),
    ("person", "{name} is handling the {thing} for us.",
     "Who is handling the {thing}?"),
    ("pref", "I always prefer {pref} when working on the {thing}.",
     "What do I prefer when working on the {thing}?"),
    ("contact", "You can reach {name} at {name2}@example.com about the {thing}.",
     "What is the contact email for the {thing}?"),
    ("location", "The {thing} review happens in {place} every {month}.",
     "Where does the {thing} review happen?"),
]

NAMES = ["Priya", "Ravi", "Ananya", "Karthik", "Meera", "Arjun", "Divya",
         "Rohan", "Sneha", "Vikram", "Nisha", "Aditya", "Kavya", "Manish"]
_THING_ADJ = ["bakery", "billing", "mobile", "analytics", "payment",
              "onboarding", "search", "admin", "notification", "report"]
_THING_NOUN = ["site", "module", "client", "board", "flow", "page",
               "feature", "panel", "service", "export"]
# 100 unique combinations. Each fact MUST get a unique subject, otherwise two
# different fact_ids can render identical text, the deduper correctly collapses
# them, and the second becomes permanently unanswerable — a dataset artefact
# that would otherwise be misread as a memory failure.
THINGS = [f"{a} {n}" for a in _THING_ADJ for n in _THING_NOUN]
TECH = ["Postgres", "Redis", "FastAPI", "Django", "Next.js", "Kotlin",
        "Flutter", "GraphQL", "Kafka", "SQLite", "Vue", "Svelte"]
ALT = ["MongoDB", "Memcached", "Flask", "Rails", "Nuxt", "Java", "Ionic",
       "REST", "RabbitMQ", "MySQL", "Angular", "Ember"]
ADJ = ["React", "Android", "internal", "offline-first", "dashboard"]
WHO = ["my dad's bakery", "our college fest", "a local clinic", "my uncle's shop"]
NAMEZ = ["Sweet Crumb", "FestTrack", "ClinicLog", "ShopMate", "QuickTally"]
MONTHS = ["March", "April", "June", "July", "September", "November"]
PREFS = ["dark mode", "typed interfaces", "small commits", "written specs",
         "morning standups", "automated tests"]
PLACES = ["the Indore campus", "the annexe lab", "the third floor room",
          "the online call", "the library block"]

FILLER = ["ok", "thanks", "cool", "great", "got it", "sure", "hi", "hey",
          "good morning", "sounds good", "nice", "yeah", "no problem", "bye",
          "right", "understood", "perfect", "alright"]

CHATTER = [
    "Let me think about that for a moment.",
    "That makes sense to me overall.",
    "I was reading about this yesterday evening.",
    "We can look at that a little later if you want.",
    "It has been a fairly busy week for everyone.",
    "I am not entirely sure what the best approach here is.",
    "Let us come back to this topic quite soon.",
    "That is an interesting way of putting it honestly.",
    "I might need to review the whole thing once more.",
    "We should probably keep moving along for now.",
    "There is quite a lot going on this month.",
    "I have been meaning to bring that up actually.",
]

ASSISTANT = [
    "Understood, I can definitely help with that.",
    "Sure, what would you like to do next here?",
    "That sounds like a reasonable plan to me.",
    "Noted. Anything else on your mind today?",
    "Happy to help you work through all of it.",
    "Got it, I will keep that in mind going forward.",
]


def build_dataset(n_sessions: int = 20, turns_per_session: int = 30,
                  n_facts: int = 40, seed: int = 7) -> Tuple[List[Turn], List[Question]]:
    rng = random.Random(seed)
    facts = []          # (fact_id, statement, probe)

    for i in range(n_facts):
        cat, stmt, probe = TEMPLATES[i % len(TEMPLATES)]
        slots = {
            "adj": rng.choice(ADJ), "who": rng.choice(WHO),
            "name": rng.choice(NAMES), "name2": rng.choice(NAMES).lower(),
            "thing": THINGS[i % len(THINGS)],   # unique per fact
            "month": rng.choice(MONTHS), "day": str(rng.randint(2, 28)),
            "num": str(rng.choice([15000, 22000, 40000, 65000, 90000])),
            "tech": TECH[i % len(TECH)], "alt": ALT[i % len(ALT)],
            "pref": rng.choice(PREFS), "place": rng.choice(PLACES),
        }
        if cat == "project":
            slots["name"] = rng.choice(NAMEZ)
        facts.append((f"f{i:02d}_{cat}", stmt.format(**slots),
                      probe.format(**slots), slots["thing"]))

    # spread facts evenly across sessions
    per_session = max(1, n_facts // n_sessions)
    plant = {}
    for i in range(n_facts):
        plant.setdefault(min(i // max(1, per_session), n_sessions - 1), []).append(i)

    turns: List[Turn] = []
    for s in range(n_sessions):
        idx = 0
        for fi in plant.get(s, []):
            fid, stmt, _, _ = facts[fi]
            turns.append(Turn(s, idx, "user", stmt, fact_id=fid)); idx += 1
            turns.append(Turn(s, idx, "assistant", rng.choice(ASSISTANT))); idx += 1

        # One supersession event, mid-run.
        #
        # The update NAMES ITS SUBJECT. It used to read "we pushed that deadline
        # to April 2" -- an anaphor with no antecedent, fired ten sessions after
        # its target, competing with four other deadline facts. No mechanism
        # could resolve that, and semantic supersession duly retired the wrong
        # record. An unresolvable probe measures nothing, so it named the
        # subject instead; `_assert_wellformed` now enforces it.
        if s == n_sessions // 2 and facts:
            fid, _, _, thing = facts[1]
            turns.append(Turn(s, idx, "user",
                              f"Actually, we pushed the {thing} deadline "
                              f"to April 2 instead.",
                              fact_id=fid)); idx += 1

        while idx < turns_per_session:
            r = rng.random()
            if r < 0.40:
                turns.append(Turn(s, idx, "user", rng.choice(FILLER), is_filler=True))
            elif r < 0.72:
                turns.append(Turn(s, idx, "user", rng.choice(CHATTER)))
            else:
                turns.append(Turn(s, idx, "assistant", rng.choice(ASSISTANT)))
            idx += 1

    # probe every fact at the very end: maximum distance from where it was said
    questions = [Question(text=p, gold_fact_id=fid,
                          asked_after_session=n_sessions - 1)
                 for fid, _, p, _ in facts]

    _assert_wellformed(facts, turns, questions)
    return turns, questions


def _assert_wellformed(facts, turns, questions):
    """Guard against dataset artefacts that would be misread as model failures."""
    stmts = [s for _, s, _, _ in facts]
    if len(set(stmts)) != len(stmts):
        raise AssertionError(
            "duplicate fact statements: two fact_ids share identical text, so "
            "the deduper will collapse them and one becomes unanswerable")
    probes = [p for _, _, p, _ in facts]
    if len(set(probes)) != len(probes):
        raise AssertionError("duplicate probe questions: gold answer is ambiguous")

    # An update must name the subject it corrects. A bare anaphor ("that
    # deadline") has no resolvable antecedent when several facts share the
    # category, so supersession cannot be scored -- it silently measures which
    # distractor happens to sit closest in embedding space.
    upd = [t for t in turns if t.text.startswith("Actually,")]
    for t in upd:
        thing = next((th for fid, _, _, th in facts if fid == t.fact_id), None)
        if thing and thing not in t.text:
            raise AssertionError(
                f"update turn does not name its subject {thing!r}: {t.text!r} "
                "-- unresolvable, so supersession cannot be evaluated")
    planted = {t.fact_id for t in turns if t.fact_id}
    missing = {q.gold_fact_id for q in questions} - planted
    if missing:
        raise AssertionError(f"probed but never planted: {sorted(missing)}")
    return True


# --------------------------------------------------------------------- LoCoMo

DEFAULT_LOCOMO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "locomo", "locomo10.json")

_SESSION_KEY = re.compile(r"^session_(\d+)$")

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain",
                  4: "single_hop", 5: "adversarial"}
ADVERSARIAL = 5


def load_locomo(path: str = DEFAULT_LOCOMO_PATH, include_adversarial: bool = False):
    """Load LoCoMo (Maharana et al., 2024) into Turn/Question.

    Returns [(sample_id, turns, questions), ...] -- one entry per conversation.
    The ten conversations are NOT concatenated: they have different speakers and
    no shared referents, so joining them would invent cross-conversation
    distractors that the benchmark never intended.

    Two data hazards, both found by inspection and both silent if unhandled:

    1. `dia_id` is unique only WITHIN a conversation. Across the file there are
       5882 turns but just 1033 distinct ids -- "D1:1" exists in every
       conversation. Ids are namespaced as "<sample_id>/<dia_id>" so evidence
       can never match a turn from a different conversation.

    2. Nine evidence ids reference turns that do not exist in their
       conversation. Those questions are unanswerable by construction, so they
       are dropped rather than left to depress every policy's recall equally --
       the same class of dataset artefact that Step 0's `_assert_wellformed`
       was added to catch.

    Adversarial questions (category 5) are excluded by default. They do carry
    evidence, but the correct behaviour is to decline to answer, so retrieving
    the evidence does not indicate success and mixing them into a retrieval
    metric makes it meaningless. Pass include_adversarial=True to study them.

    Speaker names are kept verbatim ("Caroline", "Melanie") rather than mapped
    onto user/assistant. LoCoMo is two humans talking; labelling one of them
    "assistant" would trip the scorer's assistant down-weight arbitrarily.
    """
    with open(path) as f:
        raw = json.load(f)

    out, dropped_dangling, dropped_empty, dropped_adv = [], 0, 0, 0
    for sample in raw:
        sid = sample["sample_id"]
        conv = sample["conversation"]

        turns: List[Turn] = []
        known_ids = set()
        by_id = {}
        sessions = sorted(
            ((int(m.group(1)), k) for k in conv
             for m in [_SESSION_KEY.match(k)] if m))
        for s_pos, (_, key) in enumerate(sessions):
            for idx, t in enumerate(conv[key]):
                uid = f"{sid}/{t['dia_id']}"
                known_ids.add(uid)
                by_id[uid] = t.get("text", "")
                turns.append(Turn(session=s_pos, index=idx,
                                  speaker=t.get("speaker", "user"),
                                  text=t.get("text", ""), fact_id=uid))

        last_session = len(sessions) - 1
        questions: List[Question] = []
        for q in sample.get("qa", []):
            cat = int(q["category"])
            if cat == ADVERSARIAL and not include_adversarial:
                dropped_adv += 1
                continue
            ev = q.get("evidence") or []
            if isinstance(ev, str):              # defensive: some dumps stringify it
                ev = ast.literal_eval(ev)
            ev = [f"{sid}/{e}" for e in ev]
            if not ev:
                dropped_empty += 1
                continue
            if any(e not in known_ids for e in ev):
                dropped_dangling += 1
                continue
            ans = str(q.get("answer", "") or "")
            ev_text = " ".join(by_id.get(e, "") for e in ev)
            aw = answer_tokens(ans)
            recoverable = bool(aw) and set(aw) <= set(answer_tokens(ev_text))
            questions.append(Question(
                text=q["question"], gold_fact_id=ev[0], gold_evidence=ev,
                asked_after_session=last_session,
                kind=CATEGORY_NAMES.get(cat, "unknown"), category=cat,
                answer=ans, answer_recoverable=recoverable))

        _assert_locomo_wellformed(sid, turns, questions)
        out.append((sid, turns, questions))

    if not out:
        raise ValueError(f"no conversations parsed from {path}")
    print(f"  LoCoMo: {len(out)} conversations, "
          f"{sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} questions "
          f"(dropped {dropped_dangling} dangling-evidence, "
          f"{dropped_empty} empty-evidence"
          f"{'' if include_adversarial else f', {dropped_adv} adversarial'})")
    return out


def _assert_locomo_wellformed(sid, turns, questions):
    ids = [t.fact_id for t in turns]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{sid}: duplicate turn ids after namespacing")
    known = set(ids)
    for q in questions:
        missing = set(q.evidence_ids) - known
        if missing:
            raise AssertionError(f"{sid}: evidence not in conversation: {sorted(missing)}")
    return True


# --------------------------------------------------------------- LongMemEval
DEFAULT_LME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "longmemeval", "longmemeval_s_cleaned.json")


def _lme_has_answer(turn) -> bool:
    """Is this turn marked as evidence?

    READ THIS BEFORE SIMPLIFYING IT. `has_answer` is serialised as the STRINGS
    "True", "False" and "None", not as JSON booleans. `bool("False")` is True,
    so the obvious `if turn.get("has_answer")` marks every turn in the haystack
    as evidence -- and the failure is silent, because every policy then scores
    near 100% and the benchmark looks easy instead of broken. The same class of
    bug as §13's leak: ground truth quietly flowing somewhere it should not.
    """
    v = turn.get("has_answer")
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def load_longmemeval(path: str = DEFAULT_LME_PATH, limit: int = 0):
    """Load LongMemEval into the same (sid, turns, questions) shape as LoCoMo.

    WHY THIS BENCHMARK EXISTS IN THIS PROJECT
    -----------------------------------------
    §24.1 tested the rate-distortion crossover by concatenating up to ten LoCoMo
    conversations, and stated the caveat plainly: a concatenated stream is not a
    natural long conversation, because each question concerns one constituent
    and the others act as distractors. LongMemEval supplies the real thing --
    38 to 62 sessions and ~400-600 turns of genuine single-thread history per
    question -- so the retention claim can be checked on length that was not
    manufactured.

    SHAPE DIFFERENCE, AND WHY IT IMPROVES THE STATISTICS
    ----------------------------------------------------
    LoCoMo is ten conversations each carrying many questions, so intervals are
    clustered over ten units and are correspondingly wide (§24.4, and the one
    claim that did not survive clustering). LongMemEval is one haystack PER
    question: 500 independent items, each returned here as its own
    (sid, turns, questions) triple with exactly one question. Clustering by
    conversation therefore means clustering by question, and 500 independent
    units is a far better basis for an interval than ten.

    GROUND TRUTH IS CARRIED, NEVER CONSULTED
    ----------------------------------------
    Evidence turns are marked by `has_answer` (see `_lme_has_answer` for the
    string trap). That flag is an EVALUATION LABEL of exactly the kind
    store.py's invariant forbids any policy decision from reading, so it is
    mapped onto `Turn.fact_id` -- the same channel LoCoMo's evidence ids use,
    already covered by the label-invariance tests.

    Items whose haystack contains no marked evidence turn are dropped (21 of
    500 in the cleaned release). They are unanswerable by construction, the same
    treatment `load_locomo` gives dangling evidence.

    `category` carries LongMemEval's own question type as a string rather than
    being forced onto LoCoMo's 1-5 integers. The taxonomies are genuinely
    different, and mapping one onto the other would invent equivalences that the
    per-category breakdown would then report as fact.
    """
    with open(path) as f:
        raw = json.load(f)

    out, dropped_no_ev, dropped_empty_ans = [], 0, 0
    for sample in raw[:limit] if limit else raw:
        qid = sample["question_id"]
        turns: List[Turn] = []
        evidence, by_id = [], {}
        for s_pos, session in enumerate(sample.get("haystack_sessions", [])):
            for idx, t in enumerate(session):
                uid = f"{qid}/s{s_pos}t{idx}"
                text = t.get("content", "") or ""
                by_id[uid] = text
                turns.append(Turn(session=s_pos, index=idx,
                                  speaker=t.get("role", "user"),
                                  text=text, fact_id=uid))
                if _lme_has_answer(t):
                    evidence.append(uid)

        if not evidence:
            dropped_no_ev += 1
            continue
        ans = str(sample.get("answer", "") or "")
        if not ans.strip():
            dropped_empty_ans += 1
            continue

        ev_text = " ".join(by_id.get(e, "") for e in evidence)
        aw = answer_tokens(ans)
        recoverable = bool(aw) and set(aw) <= set(answer_tokens(ev_text))
        qtype = sample.get("question_type", "unknown")
        question = Question(
            text=sample["question"], gold_fact_id=evidence[0],
            gold_evidence=evidence,
            asked_after_session=max(0, len(sample.get("haystack_sessions", [])) - 1),
            kind=qtype, category=qtype, answer=ans,
            answer_recoverable=recoverable)

        _assert_locomo_wellformed(qid, turns, [question])
        out.append((qid, turns, [question]))

    if not out:
        raise ValueError(f"no items parsed from {path}")
    n_turns = sum(len(t) for _, t, _ in out)
    n_rec = sum(1 for _, _, qs in out for q in qs if q.answer_recoverable)
    print(f"  LongMemEval: {len(out)} items, {n_turns} turns "
          f"({n_turns // max(1, len(out))} per item), {n_rec} answer-recoverable "
          f"(dropped {dropped_no_ev} without evidence, "
          f"{dropped_empty_ans} without an answer)")
    return out


# ------------------------------------------------------------------ REALTALK
DEFAULT_REALTALK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "realtalk")

REALTALK_CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "commonsense"}
_RT_RANGE = re.compile(r"^D(\d+):(\d+)-D(\d+):(\d+)$")
_RT_ID = re.compile(r"^D\d+:\d+$")


def _realtalk_evidence(raw_ids):
    """Normalise REALTALK's hand-typed evidence strings into dia_ids.

    The annotators typed evidence free-form, so roughly one id in ten is not a
    plain "D5:3": ranges ("D12:3-D12:5"), several ids in one string joined by
    ";" or ". ", and stray trailing dots. Ranges are expanded only within one
    day; anything else unparseable is returned as-is so the caller's dangling
    check drops the question rather than guessing.
    """
    out = []
    for raw in raw_ids:
        for part in re.split(r"[;,]\s*|\.\s+|\s+", str(raw).strip()):
            part = part.strip().rstrip(".")
            if not part:
                continue
            m = _RT_RANGE.match(part)
            if m and m.group(1) == m.group(3) and int(m.group(2)) <= int(m.group(4)):
                out += [f"D{m.group(1)}:{i}"
                        for i in range(int(m.group(2)), int(m.group(4)) + 1)]
            else:
                out.append(part)
    return out


def load_realtalk(path: str = DEFAULT_REALTALK_DIR):
    """Load REALTALK (Lee et al., 2025) into the LoCoMo shape.

    REALTALK is 10 conversations between real people chatting daily for 21
    days, annotated with LoCoMo's schema (`session_N` turn lists, `qa` with
    `evidence` dia_ids). It is the closest public corpus to LoCoMo that is
    NOT synthetic: LoCoMo's dialogues are LLM-generated, REALTALK's are typed
    by humans, so a scorer that has learned LoCoMo's style rather than what
    makes a turn worth keeping should show it here. Source:
    https://github.com/danny911kr/REALTALK (data/Chat_*.json; data/locomo/ in
    that repo is a copy of LoCoMo and is deliberately not loaded).

    Same hazards and the same treatment as `load_locomo`: ids are namespaced by
    conversation, and questions whose evidence does not resolve to a turn are
    dropped, after `_realtalk_evidence` has normalised the free-form strings.
    Turn text is `clean_text`; image captions are ignored, as LoCoMo's are.
    REALTALK has no adversarial category.
    """
    files = sorted(f for f in os.listdir(path)
                   if f.startswith("Chat_") and f.endswith(".json"))
    out, dropped_dangling, dropped_empty, n_ev_fixed = [], 0, 0, 0
    for fname in files:
        with open(os.path.join(path, fname)) as f:
            raw = json.load(f)
        sid = "rt-" + fname[:-len(".json")]
        turns: List[Turn] = []
        known_ids, by_id = set(), {}
        sessions = sorted(
            ((int(m.group(1)), k) for k in raw
             for m in [_SESSION_KEY.match(k)] if m))
        for s_pos, (_, key) in enumerate(sessions):
            for idx, t in enumerate(raw[key]):
                uid = f"{sid}/{t['dia_id']}"
                text = t.get("clean_text", "") or ""
                known_ids.add(uid)
                by_id[uid] = text
                turns.append(Turn(session=s_pos, index=idx,
                                  speaker=t.get("speaker", "user"),
                                  text=text, fact_id=uid))

        last_session = len(sessions) - 1
        questions: List[Question] = []
        for q in raw.get("qa", []):
            cat = int(q["category"])
            ev_raw = q.get("evidence") or []
            ev = _realtalk_evidence(ev_raw)
            if ev != list(ev_raw):
                n_ev_fixed += 1
            ev = [f"{sid}/{e}" for e in dict.fromkeys(ev)]
            if not ev:
                dropped_empty += 1
                continue
            if any(e not in known_ids for e in ev):
                dropped_dangling += 1
                continue
            ans = str(q.get("answer", "") or "")
            ev_text = " ".join(by_id.get(e, "") for e in ev)
            aw = answer_tokens(ans)
            recoverable = bool(aw) and set(aw) <= set(answer_tokens(ev_text))
            questions.append(Question(
                text=q["question"], gold_fact_id=ev[0], gold_evidence=ev,
                asked_after_session=last_session,
                kind=REALTALK_CATEGORIES.get(cat, "unknown"), category=cat,
                answer=ans, answer_recoverable=recoverable))

        _assert_locomo_wellformed(sid, turns, questions)
        out.append((sid, turns, questions))

    if not out:
        raise ValueError(f"no conversations parsed from {path}")
    print(f"  REALTALK: {len(out)} conversations, "
          f"{sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} questions "
          f"(dropped {dropped_dangling} dangling-evidence, {dropped_empty} "
          f"empty-evidence; {n_ev_fixed} evidence lists normalised)")
    return out


# ---------------------------------------------------------------------- BEAM
DEFAULT_BEAM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "beam")


def _beam_ids(src):
    """Flatten BEAM's `source_chat_ids` into a list of turn ids.

    Most question types give a list of ints, but the two-sided ones nest them
    in a dict: knowledge_update {"original_info": [86], "updated_info": [114]},
    contradiction_resolution {"first_statement": ..., "second_statement": ...},
    temporal_reasoning {"first_event": ..., "second_event": ...}. Every side
    is evidence, since answering needs both."""
    vals = list(src.values()) if isinstance(src, dict) else list(src)
    out = []
    for v in vals:
        out += v if isinstance(v, list) else [v]
    return out


def _beam_segments(chat):
    """Split a BEAM chat into id segments.

    A segment is a run of turns whose `id`s count up. At 100K and 500K a chat
    is one segment. At 1M, 3 of 35 chats restart their ids partway (chat 26
    goes 0..1993 then starts again at 0), which is where the 1,720 duplicate
    ids come from. At 10M every plan numbers its own turns, so each plan is a
    segment. Returns [(session batches, ...)] as a list of turn lists."""
    segs, cur, last = [], [], None
    for batch in chat:
        for session in batch["turns"]:
            if session and last is not None and session[0]["id"] <= last:
                segs.append(cur)
                cur = []
            cur.append(session)
            if session:
                last = session[-1]["id"]
    segs.append(cur)
    return segs


def _resolve_segment(ev, segs, key):
    """Which segment does a question's `source_chat_ids` point into?

    Only a segment holding every cited id is a candidate. With one candidate
    that is the answer. With several, each candidate's cited turns are scored
    by how many of the question's and reference answer's content words they
    contain, and the best wins if it leads by at least 0.1 (a tenth of the
    words). Otherwise the question is ambiguous and dropped: a wrong guess
    would label an unrelated turn as evidence. Returns the index or None."""
    cands = [k for k, by in enumerate(segs) if all(e in by for e in ev)]
    if len(cands) <= 1:
        return cands[0] if cands else None
    key = set(key)
    if not key:
        return None
    score = sorted(((sum(len(key & set(answer_tokens(segs[k][e]))) for e in ev)
                     / (len(key) * len(ev)), k) for k in cands), reverse=True)
    return score[0][1] if score[0][0] - score[1][0] >= 0.1 else None


def _beam_chat(sid, segs, probes, counts):
    """One BEAM chat (already split into segments) in the LoCoMo shape."""
    turns: List[Turn] = []
    by_id = {}
    seg_ids = []
    s_pos = 0
    multi = len(segs) > 1
    for k, seg in enumerate(segs):
        ids = {}
        for session in seg:
            for idx, t in enumerate(session):
                uid = f"{sid}/{k}:{t['id']}" if multi else f"{sid}/{t['id']}"
                by_id[uid] = t.get("content", "") or ""
                ids[t["id"]] = uid
                turns.append(Turn(session=s_pos, index=idx,
                                  speaker=t.get("role", "user"),
                                  text=by_id[uid], fact_id=uid))
            s_pos += 1
        seg_ids.append(ids)
    seg_text = [{i: by_id[u] for i, u in ids.items()} for ids in seg_ids]
    questions: List[Question] = []
    for qtype, items in probes.items():
        for q in items:
            raw = list(dict.fromkeys(_beam_ids(q.get("source_chat_ids") or [])))
            if not raw:
                counts["empty"] += 1
                continue
            ans = str(q.get("answer") or q.get("ideal_answer")
                      or q.get("ideal_summary") or "")
            k = _resolve_segment(raw, seg_text,
                                 answer_tokens(q["question"] + " " + ans))
            if k is None:
                counts["dangling" if not any(all(e in s for e in raw)
                                              for s in seg_text)
                       else "ambiguous"] += 1
                continue
            ev = [seg_ids[k][e] for e in raw]
            aw = answer_tokens(ans)
            ev_text = " ".join(by_id[e] for e in ev)
            recoverable = bool(aw) and set(aw) <= set(answer_tokens(ev_text))
            questions.append(Question(
                text=q["question"], gold_fact_id=ev[0], gold_evidence=ev,
                asked_after_session=s_pos - 1, kind=qtype, category=qtype,
                answer=ans, answer_recoverable=recoverable))
    _assert_locomo_wellformed(sid, turns, questions)
    return turns, questions


def load_beam(path: str = DEFAULT_BEAM_DIR, scales=("100K", "500K")):
    """Load BEAM (Tavakoli et al., 2025) into the LoCoMo shape.

    BEAM is LLM-generated user-assistant chats of 100K to 10M tokens about
    coding, maths and general projects, each with 20 probing questions over
    ten memory abilities. Source: github.com/mohammadtavakoli78/BEAM (MIT),
    chats/<scale>/<n>/chat.json and probing_questions/probing_questions.json.
    It is the opposite of LoCoMo in style: assistant turns are long markdown
    answers (~2,100 characters a turn, against LoCoMo's ~130), which is the
    regime where §30 saw the hand-written scorer's separation flip sign.

    Evidence is `source_chat_ids`, the `id` of each cited turn. Abstention
    questions cite nothing and are dropped as empty-evidence, the same
    treatment LoCoMo's empty evidence gets.

    1M: three chats restart their ids partway, so 1,720 ids name two turns.
    Each id run is a segment (_beam_segments) and a question is pinned to the
    segment its words match (_resolve_segment), or dropped as ambiguous.
    100K and 500K have one segment per chat, so they load exactly as before.
    10M has its own layout: see load_beam10m.

    `answer` is the reference answer where the type has one (answer,
    ideal_answer, ideal_summary); instruction- and preference-following
    questions have none, so they count for strict recall only.
    """
    out = []
    counts = {"dangling": 0, "empty": 0, "ambiguous": 0}
    for scale in scales:
        if scale == "10M":
            out += load_beam10m(path, _counts=counts)
            continue
        sdir = os.path.join(path, scale)
        for name in sorted(os.listdir(sdir), key=lambda s: (len(s), s)):
            cdir = os.path.join(sdir, name)
            with open(os.path.join(cdir, "chat.json")) as f:
                chat = json.load(f)
            with open(os.path.join(cdir, "probing_questions.json")) as f:
                probes = json.load(f)
            sid = f"beam-{scale}-{name}"
            turns, questions = _beam_chat(sid, _beam_segments(chat), probes, counts)
            out.append((sid, turns, questions))
    if not out:
        raise ValueError(f"no chats parsed from {path}")
    print(f"  BEAM {'+'.join(scales)}: {len(out)} chats, "
          f"{sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} questions "
          f"(dropped {counts['dangling']} dangling-evidence, "
          f"{counts['empty']} empty-evidence, {counts['ambiguous']} ambiguous)")
    return out


def load_beam10m(path: str = DEFAULT_BEAM_DIR, _counts=None):
    """BEAM 10M: 10 chats of about 20,000 turns each.

    The layout differs from the other scales. chats/10M/<n>/chat.json is a
    list of ten {"plan-k": [batches]} objects, each plan numbering its own
    turns from 0, and probing_questions/probing_questions.json cites ids
    without saying which plan (its `plan_reference` is free text such as
    "Plan 2-3", and does not match the keys). Each plan is a segment and every
    question is pinned by _resolve_segment, as for 1M. Of the 200 questions,
    24 are abstentions with no evidence and about 70 cannot be pinned; they
    are dropped and counted. Expects the chats/10M folder at <path>/10M_raw.
    """
    counts = _counts if _counts is not None else {"dangling": 0, "empty": 0,
                                                  "ambiguous": 0}
    root = os.path.join(path, "10M_raw")
    out = []
    for name in sorted((d for d in os.listdir(root) if d.isdigit()), key=int):
        cdir = os.path.join(root, name)
        with open(os.path.join(cdir, "chat.json")) as f:
            chat = json.load(f)
        with open(os.path.join(cdir, "probing_questions", "probing_questions.json")) as f:
            probes = json.load(f)
        segs = [[s for b in batches for s in b["turns"]]
                for plan in chat for batches in plan.values()]
        sid = f"beam-10M-{name}"
        turns, questions = _beam_chat(sid, segs, probes, counts)
        out.append((sid, turns, questions))
    if _counts is None:
        print(f"  BEAM 10M: {len(out)} chats, {sum(len(t) for _, t, _ in out)} turns, "
              f"{sum(len(q) for _, _, q in out)} questions (dropped "
              f"{counts['dangling']} dangling, {counts['empty']} empty, "
              f"{counts['ambiguous']} ambiguous)")
    return out


# ------------------------------------------------------------------ MemBench
DEFAULT_MEMBENCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "membench")

MEMBENCH_KINDS = ("simple", "aggregative", "comparative", "conditional",
                  "knowledge_update", "noisy", "post_processing")


def load_membench(path: str = DEFAULT_MEMBENCH_DIR, agents=("FirstAgent", "ThirdAgent"),
                  per_kind: int = 0):
    """Load MemBench's factual sets (Tan et al., 2025) into the LoCoMo shape.

    Source: github.com/import-myself/Membench (MIT), MemData/<agent>/<kind>.json,
    each {"roles": ..., "events": [trajectory, ...]} with one multiple-choice
    question per trajectory. The reflective and recommendation sets have
    other layouts and no step evidence, so they are not loaded.

    FirstAgent ("participation"): a user tells an assistant about people and
    events over ~110 user-assistant exchanges. message_list is a list of
    sessions of steps; each step's `sid` is its position in the flattened
    list, and `target_step_id` is [[sid, _], ...]. Each step becomes a user
    turn and an assistant turn; the fact is in the user's message, so that
    turn is the evidence.

    ThirdAgent ("observation"): ~15 third-person messages, no replies;
    `target_step_id` is [mid, ...].

    The question is asked after the last turn. `answer` is the correct
    choice's text. per_kind > 0 keeps the first per_kind trajectories of
    each kind and agent (a deterministic subset).
    """
    out = []
    for agent in agents:
        for kind in MEMBENCH_KINDS:
            fn = os.path.join(path, agent, f"{kind}.json")
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                d = json.load(f)
            events = d["events"] if isinstance(d, dict) else d
            if per_kind:
                events = events[:per_kind]
            tag = "first" if agent == "FirstAgent" else "third"
            for e in events:
                sid = f"membench-{tag}-{kind}-{e['tid']}"
                turns: List[Turn] = []
                ev_of = {}
                if tag == "first":
                    for s_pos, session in enumerate(e["message_list"]):
                        for st in session:
                            u = f"{sid}/{st['sid']}u"
                            ev_of[st["sid"]] = u
                            turns.append(Turn(session=s_pos, index=0,
                                              speaker="user", text=st["user_message"] or "",
                                              fact_id=u))
                            turns.append(Turn(session=s_pos, index=0,
                                              speaker="assistant",
                                              text=st["assistant_message"] or "",
                                              fact_id=f"{sid}/{st['sid']}a"))
                    raw = [x[0] if isinstance(x, list) else x
                           for x in e["QA"]["target_step_id"]]
                else:
                    for m in e["message_list"]:
                        u = f"{sid}/{m['mid']}"
                        ev_of[m["mid"]] = u
                        turns.append(Turn(session=0, index=len(turns), speaker="user",
                                          text=m["message"] or "", fact_id=u))
                    raw = [x[0] if isinstance(x, list) else x
                           for x in e["QA"]["target_step_id"]]
                pos = {}
                for t in turns:                   # index within the session
                    t.index = pos[t.session] = pos.get(t.session, -1) + 1
                qa = e["QA"]
                ev = [ev_of[r] for r in dict.fromkeys(raw) if r in ev_of]
                questions: List[Question] = []
                if ev and len(ev) == len(set(raw)):
                    ans = str(qa.get("choices", {}).get(qa.get("ground_truth"), "")
                              or qa.get("answer", ""))
                    aw = answer_tokens(ans)
                    by_id = {t.fact_id: t.text for t in turns}
                    ev_text = " ".join(by_id[x] for x in ev)
                    questions.append(Question(
                        text=qa["question"], gold_fact_id=ev[0], gold_evidence=ev,
                        asked_after_session=turns[-1].session, kind=kind,
                        category=kind, answer=ans,
                        answer_recoverable=bool(aw) and set(aw) <= set(answer_tokens(ev_text))))
                _assert_locomo_wellformed(sid, turns, questions)
                out.append((sid, turns, questions))
    print(f"  MemBench {'+'.join(agents)}: {len(out)} trajectories, "
          f"{sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} questions")
    return out


# --------------------------------------------------------------- PersonaChat
DEFAULT_PERSONACHAT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "personachat", "personachat_self_original.json")

# Words too common in persona lines to show that a turn states one.
_PERSONA_GENERIC = {"like", "love", "enjoy", "favorite", "really", "also", "just",
                    "am", "have", "has", "got", "get", "go", "going", "one", "lot",
                    "i", "my", "me", "to", "do"}


def _persona_words(text):
    return {w for w in answer_tokens(text) if w not in _PERSONA_GENERIC}


def _states(line, turn_text):
    """Does a turn state this persona line? At least half of the line's
    content words (and at least two, or all if it has fewer) appear in it."""
    c = _persona_words(line)
    hit = c & set(answer_tokens(turn_text))
    return bool(c) and len(hit) >= min(2, len(c)) and len(hit) / len(c) >= 0.5


def load_personachat(path: str = DEFAULT_PERSONACHAT_PATH, split="train",
                     per_unit: int = 20, limit: int = 0):
    """Everyday chit-chat: PersonaChat (Zhang et al., 2018), as long chats.

    Source: personachat_self_original.json, the copy the transfer-learning-
    conv-ai code downloads from s3.amazonaws.com/datasets.huggingface.co.
    Two crowdworkers chat casually, one of them told to play five persona
    lines ("i like to go hunting ."). No questions are asked, so labels are
    weak: a turn of the persona speaker is evidence when it states one of
    their lines (_states: half the line's content words, at least two). Of
    the persona speaker's turns, about 22% are. The partner's own facts are
    not labelled (their persona is not in this file), so their turns all
    count as negatives, which understates what is worth keeping.

    per_unit dialogues are joined as the sessions of one long chat, so a
    chat has ~280 turns. The persona speaker is "user" (the one whose facts
    count, as in the app) and the partner is "assistant".

    Questions, for held-out testing only: each persona line some turn
    stated is asked as the query, with the line's words that turn said as
    the answer and the turns that stated it as evidence. That is a proxy, a query that shares
    wording with its evidence, and it treats every memory the same.
    """
    with open(path) as f:
        data = json.load(f)[split]
    if limit:
        data = data[:limit]
    out = []
    for u0 in range(0, len(data), per_unit):
        sid = f"persona-{split}-{u0 // per_unit}"
        turns: List[Turn] = []
        questions: List[Question] = []
        for s, x in enumerate(data[u0:u0 + per_unit]):
            last = x["utterances"][-1]
            dia = last["history"] + [last["candidates"][-1]]
            ids = []
            for i, t in enumerate(dia):
                uid = f"{sid}/{s}:{i}"
                ids.append(uid)
                turns.append(Turn(session=s, index=i,
                                  speaker="user" if i % 2 else "assistant",
                                  text=t, fact_id=uid))
            for line in x["personality"]:
                said = [i for i, t in enumerate(dia) if i % 2 and _states(line, t)]
                if said:
                    # the answer is the line's words the person actually said
                    words = _persona_words(line) & set(answer_tokens(dia[said[0]]))
                    questions.append(Question(
                        text=line, gold_fact_id=ids[said[0]],
                        gold_evidence=[ids[i] for i in said],
                        asked_after_session=s, kind="persona", category="persona",
                        answer=" ".join(sorted(words)), answer_recoverable=True))
        for q in questions:            # asked after the whole chat
            q.asked_after_session = turns[-1].session
        _assert_locomo_wellformed(sid, turns, questions)
        out.append((sid, turns, questions))
    print(f"  PersonaChat {split}: {len(out)} chats, {sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} persona facts stated")
    return out
