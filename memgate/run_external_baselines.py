#!/usr/bin/env python3
"""MemGate against MemGPT-style and Mem0-style memory, at the same budgets.

The question a jury asks first is "how does this compare to MemGPT or Mem0?".
Both systems' memory-management patterns are rebuilt in
memgate/external_baselines.py (what is and is not copied is written there),
and this script runs them through the same LoCoMo harness as every other
result: context budget 2048, store budget S, the same embedder, the same
1527 questions, answer-presence over the 715 answerable ones.

Fairness, in the baselines' favour:
  * --tune picks each baseline's knobs by grid search ON LOCOMO ITSELF at
    S=4096 (best answer recall). MemGate's P1-S gets no such tuning: its
    thresholds are the ones every earlier result used.
  * MemGate is shown twice. "P1-S" uses the P1 heuristic scorer, which was
    never trained on anything. "P1-S learned" uses the learned scorer under
    the paper's pooled protocol: for each LoCoMo conversation a fresh model is
    fitted on the other nine plus all of REALTALK, so the conversation being
    scored is never in its own training set (asserted). No MemGate row has
    seen the answers it is scored on; the tuned baselines have.

Significance is the paired test from run_significance.py: question-level and
conversation-level bootstrap intervals and McNemar's exact test.

Usage:
    python3 run_external_baselines.py --tune      # grid search, then the sweep
    python3 run_external_baselines.py             # the sweep with the saved knobs
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

from memgate.data import load_locomo
from memgate.policies import RAGPolicy, MemGateSelectPolicy
from memgate.external_baselines import MemGPTStylePolicy, Mem0StylePolicy
from memgate.harness import run_policy
from memgate.scoring import LearnedScorer
from memgate.utils import backend_info, warm_cache, reset_cache, embed
from run_significance import collect, paired_bootstrap, mcnemar, CONTEXT_BUDGET

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KNOBS = os.path.join(RESULTS, "external_baselines_knobs.json")
BUDGETS = [2048, 4096, 8192, 16384]
TUNE_AT = 4096

GRID = {
    "MemGPT-style": (MemGPTStylePolicy, {"summary_share": [0.1, 0.25, 0.4, 0.5],
                                         "gist_tokens": [8, 12, 16, 24, 32]}),
    "Mem0-style": (Mem0StylePolicy, {"fact_tokens": [16, 24, 32, 48, 999],
                                     "update_threshold": [0.6, 0.65, 0.7, 0.75, 0.8,
                                                          0.85, 0.9, 0.95]}),
}


def tune(data):
    print(f"\nTuning each baseline on LoCoMo at S={TUNE_AT} (answer recall)")
    print("-" * 70)
    best = {}
    for name, (F, grid) in GRID.items():
        keys = list(grid)
        top = None
        for vals in itertools.product(*(grid[k] for k in keys)):
            kw = dict(zip(keys, vals))
            _s, a, _c = collect(F, data, TUNE_AT, **kw)
            m = float(np.nanmean(a))
            print(f"  {name:<13} {kw}  answer {m*100:5.1f}%")
            if top is None or m > top[0]:
                top = (m, kw)
        best[name] = top[1]
        print(f"  -> {name} uses {top[1]} ({top[0]*100:.1f}%)\n")
    with open(KNOBS, "w") as f:
        json.dump(best, f, indent=2)
    return best


def learned_scorers(data):
    """One held-out learned scorer per LoCoMo conversation (pooled protocol).

    Needs data/realtalk (git clone github.com/danny911kr/REALTALK). Without
    it the learned row is skipped rather than silently trained on less.
    """
    from memgate.data import load_realtalk
    from train_transfer import make_model
    from train_scorer import build_xy
    try:
        realtalk = load_realtalk()
    except (OSError, ValueError) as e:
        print(f"  (no REALTALK, skipping the learned row: {e})")
        return None
    warm_cache([f"[{t.speaker}] {t.text}" for _s, ts, _q in realtalk for t in ts])
    loc = [build_xy(ts, qs, True) for _s, ts, qs in data]
    rt = [build_xy(ts, qs, True) for _s, ts, qs in realtalk]
    out = []
    for i in range(len(data)):
        keep = [f for j, f in enumerate(loc) if j != i] + rt
        assert all(f is not loc[i] for f in keep), "test conversation in training"
        X = np.concatenate([f[0] for f in keep])
        y = np.concatenate([f[1] for f in keep])
        out.append(LearnedScorer(model=make_model("lr").fit(X, y),
                                 use_embedding=True, embedder=embed))
    return out


def collect_learned(scorers, data, S):
    """collect() for P1-S with conversation i scored by scorers[i]."""
    strict, answer, conv = [], [], []
    for ci, (_sid, turns, questions) in enumerate(data):
        if not questions:
            continue
        r = run_policy(MemGateSelectPolicy(budget=CONTEXT_BUDGET, store_budget=S,
                                           scorer=scorers[ci]),
                       turns, questions, CONTEXT_BUDGET)
        for d in r["detail"]:
            strict.append(1.0 if d["strict"] else 0.0)
            answer.append(np.nan if d["answer"] is None
                          else (1.0 if d["answer"] else 0.0))
            conv.append(ci)
    return np.array(strict), np.array(answer), np.array(conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    print("\nMemGate vs MemGPT-style and Mem0-style memory on LoCoMo")
    print("=" * 86)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)

    os.makedirs(RESULTS, exist_ok=True)
    if args.tune or not os.path.exists(KNOBS):
        knobs = tune(data)
    else:
        with open(KNOBS) as f:
            knobs = json.load(f)
    print(f"  baseline knobs: {knobs}\n")

    policies = [("RAG store-all", RAGPolicy, {}),
                ("MemGPT-style", MemGPTStylePolicy, knobs["MemGPT-style"]),
                ("Mem0-style", Mem0StylePolicy, knobs["Mem0-style"]),
                ("MemGate P1-S", MemGateSelectPolicy, {})]

    scorers = learned_scorers(data)
    if scorers is not None:
        policies.append(("MemGate P1-S learned", None, {}))

    rows, sig = [], []
    short = {"RAG store-all": "RAG", "MemGate P1-S": "MemGate P1-S",
             "MemGate P1-S learned": "P1-S learned"}
    hdr = f"{'S':>6}  " + "".join(f"{short.get(n, n):>16}" for n, _F, _k in policies)
    print("Answer recall (strict recall)")
    print(hdr)
    print("-" * len(hdr))
    for S in BUDGETS:
        out = {}
        for name, F, kw in policies:
            out[name] = (collect(F, data, S, **kw) if F is not None
                         else collect_learned(scorers, data, S))
            st, an, _c = out[name]
            rows.append({"policy": name, "store_budget": S,
                         "strict": st.mean() * 100, "answer": np.nanmean(an) * 100,
                         "n_questions": len(st), "n_answer": int((~np.isnan(an)).sum())})
        print(f"{S:>6}  " + "".join(
            f"{np.nanmean(out[n][1])*100:>9.1f} ({out[n][0].mean()*100:4.1f})"
            for n, _F, _k in policies))
        pairs = [("MemGate P1-S", o) for o in ("MemGPT-style", "Mem0-style",
                                               "RAG store-all")]
        if scorers is not None:
            pairs += [("MemGate P1-S learned", o) for o in ("MemGPT-style",
                                                           "Mem0-style")]
        for mine, other in pairs:
            mg, b = out[mine], out[other]
            for metric, i in (("answer", 1), ("strict", 0)):
                d, lo, hi = paired_bootstrap(mg[i], b[i], mg[2], cluster=False)
                _d, clo, chi = paired_bootstrap(mg[i], b[i], mg[2], cluster=True)
                n01, n10, p = mcnemar(mg[i], b[i])
                sig.append({"store_budget": S, "a": mine, "b": other,
                            "metric": metric, "delta": d, "lo": lo, "hi": hi,
                            "conv_lo": clo, "conv_hi": chi, "mcnemar_p": p,
                            "a_only": n01, "b_only": n10,
                            "significant": lo > 0 or hi < 0})

    print("\nMemGate minus each baseline, answer recall (95% CI, question bootstrap)")
    print("-" * 86)
    for r in sig:
        if r["metric"] != "answer":
            continue
        star = "significant" if r["significant"] else "n.s."
        print(f"  S={r['store_budget']:>5}  {r['a']:<21} vs {r['b']:<14} {r['delta']:+6.2f}"
              f"  [{r['lo']:+.2f}, {r['hi']:+.2f}]  McNemar p={r['mcnemar_p']:.4f}  {star}")

    out_csv = os.path.join(RESULTS, "external_baselines.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "store_budget", "context_budget", "strict_recall",
                    "answer_recall", "n_questions", "n_answer", "knobs",
                    "embedder", "tokenizer"])
        for r in rows:
            kn = json.dumps(knobs.get(r["policy"], {}), sort_keys=True)
            w.writerow([r["policy"], r["store_budget"], CONTEXT_BUDGET,
                        f"{r['strict']:.2f}", f"{r['answer']:.2f}",
                        r["n_questions"], r["n_answer"], kn,
                        info["embedder"], info["tokenizer"]])
    sig_csv = os.path.join(RESULTS, "external_baselines_sig.csv")
    with open(sig_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig[0]))
        w.writeheader()
        for r in sig:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {out_csv}\nwrote {sig_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
