#!/usr/bin/env python3
"""Stress test 1: the hard questions, broken down by what makes them hard.

The headline numbers average over every question. This script splits answer
recall by the things that make a memory question hard, on all three corpora
the project can reach from the cloud:

  category     each corpus's own question types (LoCoMo multi-hop / temporal /
               open-domain / single-hop; REALTALK's; BEAM's ten abilities)
  hops         how many separate turns the answer needs (1, 2, 3+)
  age          where the oldest evidence turn sits in the conversation
               (first third = asked about long ago)
  q length     how long the question itself is (words)

and compares four memory set-ups at the same store budget S:

  rag          plain retrieval, FIFO eviction (the baseline)
  heuristic    P1-S with the hand-written P1 scorer (the paper's system)
  learned      P1-S with the shipped scorer (scorer_pooled_lr.json)
  app          what the app actually runs: the shipped scorer inside
               GuardedScorer, every turn cut by ingest.chunk(), every question
               reduced by ingest.focus()

The shipped scorer was fitted on every LoCoMo and REALTALK turn, so on those
two corpora `learned` and `app` are in-sample (an upper bound, as the paper's
"pooled" row is not); BEAM was never in its training set, so BEAM is the
honest out-of-sample test.

Metric: answer recall (did the reference answer's words reach the context),
the measure every REPORT.md table uses. Strict evidence recall is reported
alongside, but under `app` a turn counts as present when any one of its
pieces is, so treat `app` strict recall as optimistic.

Usage:
    python3 stress_questions.py                         # all three corpora
    python3 stress_questions.py --corpora locomo --store-budget 4096
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from memgate.data import load_locomo, load_realtalk, load_beam
from memgate.harness import run_policy
from memgate.ingest import GuardedScorer, chunk, focus
from memgate.policies import MemGateSelectPolicy, RAGPolicy
from memgate.pretrained import load_learned
from memgate.scoring import HeuristicScorer
from memgate.types import Turn
from memgate.utils import backend_info, warm_cache, count_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "stress")
CONTEXT_BUDGET = 2048
LOADERS = {
    "locomo": lambda: load_locomo(),
    "realtalk": lambda: load_realtalk(),
    "beam": lambda: load_beam(scales=("100K", "500K")),
}
# Store budgets: LoCoMo/REALTALK at the paper's headline S; BEAM chats are
# ~5-25x longer, so the BEAM_RESULTS.md budget.
DEFAULT_S = {"locomo": 4096, "realtalk": 4096, "beam": 26000}
SETUPS = ("rag", "heuristic", "learned", "app")
LONG_SETUPS = ("app", "app-nofocus")


def wrap_long(question, padding, rnd):
    """The question as a person might ask it in a long message: a few
    paragraphs about something else first, sometimes more after it."""
    k = rnd.randint(6, 14)
    before = " ".join(rnd.sample(padding, k))
    after = " ".join(rnd.sample(padding, rnd.randint(0, 4)))
    return (f"{before}\n\nAnyway, something I wanted to check: {question}"
            + (f"\n\nAlso, for what it's worth: {after}" if after else ""))


class AppPath:
    """A policy fed the way app.py feeds one: each turn cut into turn-sized
    pieces (every piece keeps the turn's fact_id, so evidence is still
    traceable), each question reduced to what ingest.focus() retrieves with."""

    def __init__(self, inner, use_focus=True):
        self.inner = inner
        self.use_focus = use_focus
        self.name = "app path"

    def observe(self, turn):
        for k, piece in enumerate(chunk(turn.text) or [turn.text]):
            self.inner.observe(Turn(session=turn.session, index=turn.index,
                                    speaker=turn.speaker, text=piece,
                                    fact_id=turn.fact_id))

    def build_context(self, query, budget):
        q = (focus(query) or query) if self.use_focus else query
        return self.inner.build_context(q, budget)

    def stored_tokens(self):
        return self.inner.stored_tokens()

    def stats(self):
        return self.inner.stats()


def make(setup, S, learned):
    kw = dict(budget=CONTEXT_BUDGET, store_budget=S)
    if setup == "rag":
        return RAGPolicy(**kw)
    if setup == "heuristic":
        return MemGateSelectPolicy(scorer=HeuristicScorer(), **kw)
    if setup == "learned":
        return MemGateSelectPolicy(scorer=learned, **kw)
    return AppPath(MemGateSelectPolicy(scorer=GuardedScorer(learned), **kw),
                   use_focus=setup != "app-nofocus")


def buckets(q, turns):
    """The difficulty tags of one question."""
    pos = {t.fact_id: i for i, t in enumerate(turns)}
    ev = [pos[e] for e in q.evidence_ids if e in pos]
    hops = len(set(q.evidence_ids))
    oldest = min(ev) / max(1, len(turns) - 1) if ev else 1.0
    qw = len(q.text.split())
    return {
        "category": str(q.kind),
        "hops": "1" if hops == 1 else ("2" if hops == 2 else "3+"),
        "age": ("first third" if oldest < 1 / 3 else
                "middle third" if oldest < 2 / 3 else "last third"),
        "q length": ("<=10 words" if qw <= 10 else
                     "11-25 words" if qw <= 25 else ">25 words"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="locomo,realtalk,beam")
    ap.add_argument("--store-budget", type=int, default=0,
                    help="one S for every corpus (default: per corpus)")
    ap.add_argument("--setups", default="")
    ap.add_argument("--long-questions", action="store_true",
                    help="ask every question inside a long message (150-450 "
                         "words of other chat around it), with and without "
                         "ingest.focus()")
    args = ap.parse_args()
    names = [c for c in args.corpora.split(",") if c]
    setups = [s for s in args.setups.split(",") if s] or list(
        LONG_SETUPS if args.long_questions else SETUPS)
    learned = load_learned()
    info = backend_info()
    print(f"embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    os.makedirs(RESULTS, exist_ok=True)

    rows = []
    for c in names:
        data = LOADERS[c]()
        S = args.store_budget or DEFAULT_S[c]
        if args.long_questions:
            import copy
            import random
            rnd = random.Random(20261021)
            new = []
            for sid, ts, qs in data:
                # padding from the OTHER conversations: chat about other people
                pad = [t.text for s2, ts2, _ in data if s2 != sid for t in ts2
                       if len(t.text.split()) >= 12]
                qs2 = []
                for q in qs:
                    q2 = copy.copy(q)
                    q2.text = wrap_long(q.text, pad, rnd)
                    qs2.append(q2)
                new.append((sid, ts, qs2))
            data = new
        texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
        if "app" in setups:
            texts += [f"[{t.speaker}] {p}" for _s, ts, _q in data for t in ts
                      for p in chunk(t.text)]
            texts += [focus(q.text) for _s, _t, qs in data for q in qs]
        texts += [q.text for _s, _t, qs in data for q in qs]
        t0 = time.time()
        warm_cache(texts)
        print(f"\n{c}: {len(data)} conversations, S={S}, "
              f"{sum(len(ts) for _s, ts, _q in data):,} turns "
              f"(embedded {len(set(texts)):,} strings in {time.time() - t0:.0f}s)")
        for setup in setups:
            t0 = time.time()
            for sid, ts, qs in data:
                if not qs:
                    continue
                r = run_policy(make(setup, S, learned), ts, qs, CONTEXT_BUDGET)
                for q, d in zip(qs, r["detail"]):
                    rows.append(dict(corpus=c, setup=setup, conv=sid, S=S,
                                     strict=int(d["strict"]),
                                     answer=None if d["answer"] is None else int(d["answer"]),
                                     ctx_tokens=d["tokens"], **buckets(q, ts)))
            sub = [r for r in rows if r["corpus"] == c and r["setup"] == setup]
            a = [r["answer"] for r in sub if r["answer"] is not None]
            print(f"  {setup:<10} answer {np.mean(a) * 100:5.1f}%  "
                  f"strict {np.mean([r['strict'] for r in sub]) * 100:5.1f}%  "
                  f"({len(sub)} q, {time.time() - t0:.0f}s)")

    tag = "_".join(names) + (f"_S{args.store_budget}" if args.store_budget else "") \
        + ("_longq" if args.long_questions else "")
    path = os.path.join(RESULTS, f"questions_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-question rows -> {os.path.relpath(path, HERE)}")
    summary = summarise(rows, setups)
    with open(os.path.join(RESULTS, f"questions_{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print_summary(summary, setups)


def summarise(rows, setups):
    out = {}
    for c in dict.fromkeys(r["corpus"] for r in rows):
        out[c] = {}
        for dim in ("category", "hops", "age", "q length"):
            out[c][dim] = {}
            for val in sorted({r[dim] for r in rows if r["corpus"] == c}):
                cell = {}
                for s in setups:
                    sub = [r for r in rows if r["corpus"] == c and r["setup"] == s
                           and r[dim] == val]
                    a = [r["answer"] for r in sub if r["answer"] is not None]
                    cell[s] = {"n": len(sub), "n_answer": len(a),
                               "answer": float(np.mean(a)) * 100 if a else None,
                               "strict": float(np.mean([r["strict"] for r in sub])) * 100
                               if sub else None}
                out[c][dim][val] = cell
    return out


def print_summary(summary, setups):
    for c, dims in summary.items():
        print(f"\n=== {c}: answer recall % (n answerable) ===")
        for dim, vals in dims.items():
            print(f"  by {dim}")
            print("    " + f"{'':<30}" + "".join(f"{s:>11}" for s in setups))
            for val, cell in vals.items():
                n = cell[setups[0]]["n_answer"]
                line = "".join(
                    f"{cell[s]['answer']:>10.1f}%" if cell[s]["answer"] is not None
                    else f"{'--':>11}" for s in setups)
                print(f"    {(val + f' (n={n})')[:30]:<30}{line}")


if __name__ == "__main__":
    main()
