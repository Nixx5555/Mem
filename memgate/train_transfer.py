#!/usr/bin/env python3
"""P3-learned on more than one corpus — does a bigger training set transfer?

§30 found the hand-written scorer's evidence separation flips sign between
LoCoMo and LongMemEval, and §39 left "a scorer that transfers" as the one open
track. `train_scorer.py` fits on nine LoCoMo conversations and tests on the
tenth, which says nothing about transfer: train and test share a generator,
a style and an annotation team. This script adds corpora and asks three
questions of every held-out conversation, each with its own training set:

  in-corpus   fitted on the OTHER conversations of the same corpus
              (train_scorer.py's protocol; the number we already report)
  cross       fitted on the other corpora ONLY -- has it learned what makes a
              turn worth keeping, or what LoCoMo looks like?
  pooled      fitted on everything except this conversation -- the "larger
              dataset" scorer

The fold discipline of train_scorer.py is unchanged and asserted: the
conversation being scored never contributes a turn or a label to the model
that scores it. At inference the scorer sees text and speaker only.

Corpora (any subset, via --corpora):
  locomo       10 conversations, LLM-generated (data/locomo/locomo10.json)
  realtalk     10 conversations, typed by real people over 21 days
               (data/realtalk/Chat_*.json, from github.com/danny911kr/REALTALK)
  longmemeval  one haystack per question (data/longmemeval/...s_cleaned.json);
               --lme-limit caps the items, since each is ~500 turns

Usage:
    python3 train_transfer.py --corpora locomo,realtalk
    python3 train_transfer.py --corpora locomo,realtalk,longmemeval --lme-limit 100
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from memgate.data import load_locomo, load_realtalk, load_longmemeval, load_beam
from memgate.policies import MemGateSelectPolicy, RAGPolicy
from memgate.harness import run_policy
from memgate.scoring import HeuristicScorer, LearnedScorer
from memgate.utils import backend_info, warm_cache, reset_cache, embed, count_tokens

from train_scorer import build_xy

MODELS = ("lr", "hgb", "mlp")


def make_model(name):
    """The scorer's classifier. "lr" is train_scorer.fit's pipeline, unchanged."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0))
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=0)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
            max_iter=200, random_state=0))
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONTEXT_BUDGET = 2048
B = 10000
RNG = np.random.default_rng(20261013)
LOADERS = {
    "locomo": lambda a: load_locomo(),
    "realtalk": lambda a: load_realtalk(),
    "longmemeval": lambda a: load_longmemeval(limit=a.lme_limit),
    "beam": lambda a: load_beam(scales=("100K",)),
    "beam500": lambda a: load_beam(scales=("500K",)),
}
# Corpora that are one source at two sizes. "in-corpus" and "cross" are
# decided by family, so BEAM-500K chats never count as cross-corpus data for a
# BEAM-100K test chat.
FAMILY = {"beam500": "beam"}
PREV = ("locomo", "realtalk")


def cluster_bootstrap(a, b, conv, n=B):
    """95% CI for mean(a) - mean(b), resampling whole conversations.

    Clustered because questions within a conversation share a memory state;
    resampling questions independently would overstate certainty (§24.4).
    """
    ok = ~(np.isnan(a) | np.isnan(b))
    d, conv = (a - b)[ok], conv[ok]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    groups = [d[conv == c] for c in np.unique(conv)]
    k = len(groups)
    means = np.empty(n)
    for i in range(n):
        means[i] = np.concatenate([groups[j] for j in RNG.integers(0, k, k)]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100


_TAG = re.compile(r"^\[[^\]]*\] ")


class MessageOnly:
    """Score the message without the "[Speaker] " tag the policy prepends.

    Written before scoring.py stripped the tag itself (message_only, from the
    speaker-tag fix); on top of that fix it changes only what the learned
    scorer EMBEDS, since LearnedScorer passes its input to MiniLM unstripped.
    With --scorer-fix the learned scorer is trained on the bare message too,
    so train and inference see the same string.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "scorer")

    def score(self, text, speaker="user"):
        return self.inner.score(_TAG.sub("", text, count=1), speaker)


def apply_month_fix():
    """Whole-word months, swapped into scoring.py at runtime.

    The committed MONTH_PAT has no word end, so "dec" matches "decided" and
    "mar" matches "market" and "Maria" (371 of the 378 LoCoMo turns it fires
    on contain no month). Patched here rather than in scoring.py so this
    script can measure both versions from one tree. "May" stays a month only
    when capitalised, so the modal verb does not fire.
    """
    from memgate import scoring
    scoring.MONTH_PAT = re.compile(
        r"(?i:\b(?:january|february|march|april|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|"
        r"nov|dec)\b)|\bMay\b")


def outcomes(r):
    s = np.array([1.0 if d["strict"] else 0.0 for d in r["detail"]])
    a = np.array([np.nan if d["answer"] is None else (1.0 if d["answer"] else 0.0)
                  for d in r["detail"]])
    return s, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="locomo,realtalk")
    ap.add_argument("--store-budget", type=int, default=2048)
    ap.add_argument("--test-corpora", default="",
                    help="corpora to test on (default: all of --corpora); the "
                         "rest are used for training only")
    ap.add_argument("--lme-limit", type=int, default=100)
    ap.add_argument("--features", default="hand,hand+MiniLM")
    ap.add_argument("--model", default="lr", choices=MODELS)
    ap.add_argument("--balance", action="store_true",
                    help="weight each corpus family equally in training")
    ap.add_argument("--scorer-fix", action="store_true",
                    help="apply both SCORER_CHECK.md fixes, for P1 and P3: "
                         "score the bare message, not '[speaker] message', "
                         "and match months as whole words")
    args = ap.parse_args()
    args.message_only = args.scorer_fix
    if args.scorer_fix:
        apply_month_fix()
    names = [c.strip() for c in args.corpora.split(",") if c.strip()]
    feat_sets = [f.strip() for f in args.features.split(",")]
    tests = [c.strip() for c in args.test_corpora.split(",") if c.strip()] or names
    fam = lambda c: FAMILY.get(c, c)

    print("\nMemGate — P3-learned across corpora (in-corpus / cross / pooled)")
    print("=" * 80)
    corpora = {c: LOADERS[c](args) for c in names}
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    if "hand+MiniLM" in feat_sets:
        warm_cache([f"[{t.speaker}] {t.text}" for d in corpora.values()
                    for _s, ts, _q in d for t in ts]
                   + [q.text for d in corpora.values() for _s, _t, qs in d for q in qs]
                   + ([t.text for d in corpora.values() for _s, ts, _q in d for t in ts]
                      if args.message_only else []))
    S = args.store_budget
    print(f"  context budget {CONTEXT_BUDGET}   store budget {S}   scorer input: "
          f"{'FIXED (message only, whole-word months)' if args.scorer_fix else 'scoring.py as is'}")
    for c, d in corpora.items():
        tok = sum(count_tokens(t.text) for _s, ts, _q in d for t in ts)
        print(f"  {c:<12} {len(d):>4} units  {tok/len(d):>9,.0f} tok/unit  "
              f"store/unit {S/(tok/len(d))*100:5.1f}%")
    print()

    # every conversation, tagged with its corpus
    units = [(c, sid, ts, qs) for c, d in corpora.items() for sid, ts, qs in d]
    rows, sig_rows = [], []
    for fs in feat_sets:
        use_emb = fs == "hand+MiniLM"
        feats = [build_xy(ts, qs, use_emb, message_only=args.message_only)
                 for _c, _s, ts, qs in units]
        total = sum(len(f[0]) for f in feats)

        def train_on(keep):
            idx = [j for j in range(len(units)) if keep(j)]
            X = np.concatenate([feats[j][0] for j in idx])
            y = np.concatenate([feats[j][1] for j in idx])
            if not args.balance:
                return make_model(args.model).fit(X, y)
            # every corpus family carries the same total weight, so BEAM's
            # 43,800 turns cannot outvote LoCoMo's 5,900
            n_fam = {}
            for j in idx:
                f = fam(units[j][0]); n_fam[f] = n_fam.get(f, 0) + len(feats[j][1])
            w = np.concatenate([np.full(len(feats[j][1]),
                                        len(y) / (len(n_fam) * n_fam[fam(units[j][0])]))
                                for j in idx])
            m = make_model(args.model)
            step = m.steps[-1][0] if hasattr(m, "steps") else None
            kw = {f"{step}__sample_weight": w} if step else {"sample_weight": w}
            return m.fit(X, y, **kw)

        for test_c in tests:
            others = [o for o in names if fam(o) != fam(test_c)]
            protocols = ["in-corpus"] + (["cross", "pooled"] if others else [])
            # "pooled-prev": pooled, but on only the data PR #4 trained on, so
            # pooled - pooled-prev is what the added corpora are worth
            if any(fam(c) not in PREV for c in names) and any(fam(c) in PREV for c in names):
                protocols.append("pooled-prev")
            cross_model = None    # one training set for every test unit
            acc = {p: {"auc": [], "ap": [], "s": [], "a": [], "conv": []}
                   for p in ["heuristic", "rag"] + protocols}
            for i, (c, sid, ts, qs) in enumerate(units):
                if c != test_c or not qs:
                    continue
                Xte, yte = feats[i]
                rule = {"in-corpus": lambda j: j != i and fam(units[j][0]) == fam(test_c),
                        "cross": lambda j: fam(units[j][0]) != fam(test_c),
                        "pooled": lambda j: j != i,
                        "pooled-prev": lambda j: j != i and fam(units[j][0]) in PREV}
                scorers = {"heuristic": HeuristicScorer()}
                for p in protocols:
                    # Fold integrity, asserted rather than assumed: the unit
                    # being scored must not be in its own training set.
                    assert not rule[p](i), f"{p}: test unit in its training set"
                    n_tr = sum(len(feats[j][0]) for j in range(len(units)) if rule[p](j))
                    assert n_tr + len(Xte) <= total
                    if p == "cross":
                        cross_model = cross_model or train_on(rule[p])
                        m = cross_model
                    else:
                        m = train_on(rule[p])
                    if len(set(yte)) > 1:
                        pr = m.predict_proba(Xte)[:, 1]
                        acc[p]["auc"].append(roc_auc_score(yte, pr))
                        acc[p]["ap"].append(average_precision_score(yte, pr))
                    scorers[p] = LearnedScorer(model=m, use_embedding=use_emb,
                                               embedder=embed)
                if len(set(yte)) > 1:
                    h = np.array([HeuristicScorer().score(
                        t.text if args.message_only else f"[{t.speaker}] {t.text}",
                        t.speaker)[0]
                                  for t in ts])
                    acc["heuristic"]["auc"].append(roc_auc_score(yte, h))
                    acc["heuristic"]["ap"].append(average_precision_score(yte, h))
                for p, sc in scorers.items():
                    if args.message_only:
                        sc = MessageOnly(sc)
                    r = run_policy(MemGateSelectPolicy(budget=CONTEXT_BUDGET,
                                                       store_budget=S, scorer=sc),
                                   ts, qs, CONTEXT_BUDGET)
                    s, a = outcomes(r)
                    acc[p]["s"].append(s); acc[p]["a"].append(a)
                    acc[p]["conv"].append(np.full(len(s), i))
                if fs == feat_sets[0]:
                    r = run_policy(RAGPolicy(budget=CONTEXT_BUDGET, store_budget=S),
                                   ts, qs, CONTEXT_BUDGET)
                    s, a = outcomes(r)
                    acc["rag"]["s"].append(s); acc["rag"]["a"].append(a)
                    acc["rag"]["conv"].append(np.full(len(s), i))

            print(f"  test corpus: {test_c}   features: {fs}")
            print(f"    {'scorer':<22}{'AUC':>7}{'AP':>7}{'strict':>9}{'answer':>9}")
            cat = {p: {k: np.concatenate(v[k]) for k in ("s", "a", "conv")}
                   for p, v in acc.items() if v["s"]}
            for p, v in acc.items():
                if p not in cat:
                    continue
                st = cat[p]["s"].mean() * 100
                an = np.nanmean(cat[p]["a"]) * 100
                auc = np.mean(v["auc"]) if v["auc"] else float("nan")
                apv = np.mean(v["ap"]) if v["ap"] else float("nan")
                label = {"heuristic": "P1 heuristic", "rag": "RAG (FIFO, same S)"}.get(
                    p, f"P3 {p}")
                print(f"    {label:<22}{auc:7.3f}{apv:7.3f}{st:8.1f}%{an:8.1f}%")
                rows.append({"test_corpus": test_c, "features": fs, "scorer": p,
                             "auc": auc, "ap": apv, "strict": st, "answer": an,
                             "n_questions": len(cat[p]["s"]),
                             "n_answer": int((~np.isnan(cat[p]["a"])).sum()),
                             "store_budget": S, "scorer_fix": args.scorer_fix,
                             "model": args.model, "balance": args.balance,
                             "embedder": info["embedder"]})
            print("    paired differences, answer recall (95% CI, clustered by unit):")
            pairs = [(p, "heuristic") for p in protocols]
            if others:
                pairs += [("pooled", "in-corpus"), ("cross", "in-corpus")]
            if "pooled-prev" in protocols:
                pairs += [("pooled", "pooled-prev")]
            for x, y in pairs:
                for metric, key in (("answer", "a"), ("strict", "s")):
                    o, lo, hi = cluster_bootstrap(cat[x][key], cat[y][key], cat[x]["conv"])
                    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
                    if metric == "answer":
                        print(f"      {x:>9} - {y:<10} {o:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  {sig}")
                    sig_rows.append({"test_corpus": test_c, "features": fs,
                                     "a": x, "b": y, "metric": metric,
                                     "delta": o, "lo": lo, "hi": hi,
                                     "significant": sig == "SIG",
                                     "store_budget": S,
                                     "scorer_fix": args.scorer_fix})
            print()

    os.makedirs(RESULTS, exist_ok=True)
    tag = "_".join(names) + ("" if tests == names else "_test-" + "-".join(tests)) \
        + f"_S{S}" + ("_tagmonthfix" if args.scorer_fix else "") \
        + ("" if args.model == "lr" else f"_{args.model}") \
        + ("_balanced" if args.balance else "")
    for fname, data in ((f"transfer_{tag}.csv", rows),
                        (f"transfer_sig_{tag}.csv", sig_rows)):
        out = os.path.join(RESULTS, fname)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
