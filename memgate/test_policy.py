#!/usr/bin/env python3
"""Memory-policy tests through the app's own Session: keep, compress, forget
and recall under a token budget, for P1, P1-S and P1-A, with both scorers.
Run: python test_policy.py"""
import os
import sys

import app as A
# The policy as the experiments run it: without the app's entry gate,
# which forgets small talk before the policy sees it (test_app.py covers it).
A.GATE = False
from memgate.utils import count_tokens

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


FACTS = ["My sister Priya lives in Pune and works at Infosys.",
         "My budget for the laptop is 85,000 rupees.",
         "My flight to Delhi is on 14 December at 6:40 am.",
         "I'm allergic to peanuts.",
         "My locker code is 4417."]
FILLER = ["ok", "haha nice", "thanks!", "cool", "lol", "hmm", "sure", "yeah",
          "nice weather today", "how are you", "good morning", "brb",
          "that's funny", "okay got it", "wow", "see you", "great 👍",
          "no worries", "alright", "fine"]
REPLIES = ["Noted.", "Sure thing!", "Got it.", "Okay!", "Ha, yes.",
           "Of course.", "Glad to help.", "Understood.", "Right.", "Sounds good."]
QUESTIONS = [("Where does my sister live?", "Pune"),
             ("What is my laptop budget?", "85,000"),
             ("When is my flight?", "14 December"),
             ("What am I allergic to?", "peanuts"),
             ("What is my locker code?", "4417")]


def chat(mode, scale, scorer, store_budget=None):
    os.environ["MEMGATE_SCORER"] = scorer
    s = A.Session(mode, store_budget, scale)
    trace, chit, n = [], iter(FILLER), 0
    for f in FACTS:
        for t in [f, next(chit), next(chit), next(chit), next(chit)]:
            for spk, txt in (("user", t), ("assistant", REPLIES[n % 10])):
                s.add(spk, txt)
                # enforcement runs when Tier 1 hands a message on, so the
                # store may sit over S by at most the message just added
                trace.append(s.policy.store.stored_cost()
                             - count_tokens(f"[{spk}] {txt}"))
            n += 1
    return s, trace


def recalled(s):
    out = []
    for q, key in QUESTIONS:
        ctx, items = s.context(q)
        out.append(any(key in i["text"] for i in items))
    return out


for scorer in ("learned", "heuristic"):
    print(f"\n-- P1-S select, {scorer} scorer --")
    s, trace = chat("p1s", "full", scorer)
    check(f"[{scorer}] scorer in use is the one asked for",
          s.policy.scorer.name == ("P3-learned" if scorer == "learned"
                                   else "P1-heuristic"), s.policy.scorer.name)
    check(f"[{scorer}] full sizes: every fact recalled", all(recalled(s)),
          str(recalled(s)))

    s, trace = chat("p1s", "compact", scorer)
    S = s.store_budget
    over = [t for t in trace if t > S]
    check(f"[{scorer}] compact: store stays within S = {S} (+ the newest message)",
          not over, f"peaks {over[:5]}")
    drops = [e for e in s.log if e["type"] == "drop"
             and e["msg"].startswith("store over budget")]
    check(f"[{scorer}] compact: the budget forced forgetting", len(drops) > 0)
    t3 = s.state()["t3"]
    kept_u = [i["utility"] for i in t3 if not i["superseded"]]
    dropped_u = [float(e["msg"].rsplit("u = ", 1)[1]) for e in drops]
    check(f"[{scorer}] compact: what was forgotten scored no higher than "
          "what was kept", max(dropped_u) <= max(kept_u),
          f"dropped max {max(dropped_u)}, kept max {max(kept_u)}")
    kept = " ".join(i["text"] for i in t3)
    facts_kept = sum(f in kept for f in FACTS)
    chit_kept = sum(f"[user] {x}" in kept for x in FILLER)
    check(f"[{scorer}] compact: more facts than small talk survive",
          facts_kept / len(FACTS) > chit_kept / len(FILLER),
          f"facts {facts_kept}/5, small talk {chit_kept}/20")
    n_hit = sum(recalled(s))
    check(f"[{scorer}] compact: most facts still recalled under S = {S}",
          n_hit >= 3, f"{n_hit}/5")

print("\n-- context budget --")
for scale in ("full", "compact"):
    s, _ = chat("p1s", scale, "heuristic")
    budget = s.context_budget
    worst = max(s.context(q)[0].tokens for q, _ in QUESTIONS)
    check(f"{scale}: context never exceeds its {budget}-token budget",
          worst <= budget, f"worst {worst}")
s = A.Session("p1s", None, "full")
s.add("user", "word " * 4000)
ctx, items = s.context("what did I say?")
check("a message larger than the context budget does not overflow it",
      ctx.tokens <= s.context_budget, f"{ctx.tokens}")
check("context tokens agree with the tokenizer",
      ctx.tokens >= sum(count_tokens(i["text"]) for i in items) * 0.9)

print("\n-- P1 threshold routing --")
s, _ = chat("p1", "full", "heuristic")
st = s.state()
check("P1 has no store budget", st["store_budget"] is None)
check("P1 sends most facts to Tier 3",
      sum(any(f in i["text"] for i in st["t3"]) for f in FACTS) >= 3)
check("P1 forgets no fact outright",
      all(any(f in i["text"] for k in ("t1", "t2", "t3") for i in st[k])
          for f in FACTS))
check("P1 drops filler below tau_low",
      any(e["type"] == "drop" for e in s.log))
check("P1 compresses mid-score messages into Tier 2", len(st["t2"]) > 0)
check("P1 log explains each decision with its score",
      all("u = " in e["msg"] for e in s.log if e["type"] in ("t3", "t2", "drop")
          and not e["msg"].startswith(("store over budget", "very high"))))

print("\n-- P1-A adaptive --")
s = A.Session("p1a", None, "compact")
for spk, txt in A.SAMPLE:
    s.add(spk, txt)
st = s.state()
check("P1-A demotes under storage pressure", st["counts"]["demoted"] > 0,
      str(st["counts"]))
check("P1-A demotions show in the log",
      any(e["type"] == "demote" for e in s.log))
check("P1-A gists are shorter than what they came from",
      all(i["tokens"] <= 40 for i in st["t2"]))
s2 = A.Session("p1a", None, "full")
for spk, txt in A.SAMPLE:
    s2.add(spk, txt)
check("P1-A compresses nothing while storage is free",
      s2.state()["counts"]["demoted"] == 0)

print("\n-- the sample chat (the jury demo) --")
for scorer in ("learned", "heuristic"):
    os.environ["MEMGATE_SCORER"] = scorer
    s = A.Session("p1s", None, "compact")
    for spk, txt in A.SAMPLE:
        s.add(spk, txt)
    st = s.state()
    live = [i["text"] for i in st["t3"] if not i["superseded"]]
    check(f"[{scorer}] the correction (4 November) is kept",
          any("4 November" in t for t in live))
    check(f"[{scorer}] the name is kept", any("Simar" in t for t in live))
    check(f"[{scorer}] 'You're welcome!' is forgotten",
          not any("welcome" in t for t in live))
    ctx, items = s.context("When is my report due?")
    check(f"[{scorer}] asking for the deadline retrieves the correction",
          any("4 November" in i["text"] for i in items))
    check(f"[{scorer}] supersession retires the old deadline",
          any(e["type"] == "supersede" for e in s.log))

print("\n-- determinism --")
a, _ = chat("p1s", "compact", "learned")
b, _ = chat("p1s", "compact", "learned")
check("the same chat gives the same memory twice",
      [i["text"] for i in a.state()["t3"]] == [i["text"] for i in b.state()["t3"]]
      and [e["msg"] for e in a.log] == [e["msg"] for e in b.log])

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
