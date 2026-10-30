#!/usr/bin/env python3
"""MemGPT-style and Mem0-style baselines: they keep to the budget, they do
what their designs say, and the committed comparison says what the docs say.
Run: python test_baselines.py"""
import csv
import os
import sys

from memgate.types import Turn
from memgate.utils import count_tokens
from memgate.policies import RAGPolicy
from memgate.external_baselines import MemGPTStylePolicy, Mem0StylePolicy

HERE = os.path.dirname(os.path.abspath(__file__))
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
FILLER = ["ok", "haha nice", "thanks!", "cool", "lol"]


def turns(n_rounds=12):
    out, i = [], 0
    for r in range(n_rounds):
        for t in FACTS + FILLER:
            out.append(Turn(session=r, index=i, speaker="user" if i % 2 else "Simar",
                            text=f"{t} (round {r})" if t in FACTS else t,
                            fact_id=f"f{i}"))
            i += 1
    return out


# ---- budgets
for S in (256, 1024):
    g = MemGPTStylePolicy(store_budget=S, budget=2048)
    m = Mem0StylePolicy(store_budget=S, budget=2048)
    worst_g = worst_m = 0
    for t in turns():
        g.observe(t); m.observe(t)
        worst_g = max(worst_g, g.stored_tokens())
        worst_m = max(worst_m, m.stored_tokens())
    check(f"MemGPT-style never holds more than S = {S}", worst_g <= S, worst_g)
    check(f"Mem0-style never holds more than S = {S}", worst_m <= S, worst_m)
    for p in (g, m):
        ctx = p.build_context("Where does my sister live?", 300)
        check(f"{p.name} context keeps to its budget at S = {S}",
              ctx.tokens <= 300 and count_tokens(ctx.text) <= 300 + len(ctx.items))

# ---- MemGPT: evicted messages fold into a summary that stays in context
g = MemGPTStylePolicy(store_budget=512, budget=2048)
for t in turns():
    g.observe(t)
check("MemGPT-style folds evicted messages into its summary",
      len(g.fragments) > 0 and g.summary.text)
check("the summary only claims ids whose gist it still holds",
      len(g.summary.covered_ids) <= len(g.fragments))
check("the summary fits its share of the budget",
      count_tokens(g.summary.text) <= g.summary_budget)
ctx = g.build_context("What is my locker code?", 1024)
check("the summary is placed first in the context",
      ctx.items and ctx.items[0] is g.summary)

# Unbounded, nothing is evicted, so MemGPT-style is RAG plus an empty summary.
g, r = MemGPTStylePolicy(budget=2048), RAGPolicy(budget=2048)
for t in turns(3):
    g.observe(t); r.observe(t)
q = "When is my flight to Delhi?"
check("with no store cap MemGPT-style retrieves exactly what RAG does",
      [i.text for i in g.build_context(q, 400).items]
      == [i.text for i in r.build_context(q, 400).items])

# ---- Mem0: NOOP, ADD, UPDATE
m = Mem0StylePolicy(budget=2048, update_threshold=0.9)
m.observe(Turn(0, 0, "user", "ok", fact_id="a"))
check("Mem0-style stores nothing for filler (NOOP)",
      m.ops["noop"] == 1 and not m.memories)
m.observe(Turn(0, 1, "user", "My locker code is 4417.", fact_id="b"))
m.observe(Turn(0, 2, "user", "My locker code is 4417.", fact_id="c"))
check("a repeated fact UPDATEs the stored one instead of adding a copy",
      m.ops["update"] == 1 and len(m.memories) == 1)
check("the newest version wins the update",
      m.memories[0].fact_id == "c")
m.observe(Turn(0, 3, "user", "I'm allergic to peanuts.", fact_id="d"))
check("an unrelated fact is ADDed", m.ops["add"] == 2 and len(m.memories) == 2)
check("Mem0-style facts are compressed to fact_tokens",
      all(count_tokens(i.text) <= m.fact_tokens + 3 for i in m.memories))

# ---- the committed results say what the docs claim
path = os.path.join(HERE, "results", "external_baselines.csv")
if os.path.exists(path):
    rows = list(csv.DictReader(open(path)))
    by = {(r["policy"], int(r["store_budget"])): float(r["answer_recall"]) for r in rows}
    budgets = sorted({int(r["store_budget"]) for r in rows})
    check("the comparison covers S = 2k to 16k", budgets == [2048, 4096, 8192, 16384])
    check("MemGate's learned scorer beats both baselines at every budget",
          all(by[("MemGate P1-S learned", S)] > max(by[("MemGPT-style", S)],
                                                     by[("Mem0-style", S)])
              for S in budgets))
    check("the P1-S row reproduces the headline 37.5% at S = 4096",
          abs(by[("MemGate P1-S", 4096)] - 37.48) < 0.05)
    sig = list(csv.DictReader(open(os.path.join(HERE, "results",
                                                "external_baselines_sig.csv"))))
    check("every learned-vs-baseline answer gap is significant",
          all(r["significant"] == "True" for r in sig
              if r["a"] == "MemGate P1-S learned" and r["metric"] == "answer"))
else:
    check("results/external_baselines.csv is committed", False)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
