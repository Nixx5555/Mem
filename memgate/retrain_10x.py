#!/usr/bin/env python3
"""Retrain the scorer on ten times the data of PR #12, and test it held-out.

PR #12's scorer (retrain_scorer.py, shipped) was fitted on 58,600 turns:
LoCoMo, REALTALK, BEAM 100K and BEAM 500K. This script adds what the cloud
can reach, about 593,000 turns in all:

  beam1m     BEAM 1M, 35 chats, 74,630 turns (ids de-duplicated, data.py)
  beam10m    BEAM 10M, 10 chats, 208,696 turns (plans pinned, data.py)
  mb3        MemBench ThirdAgent, every factual trajectory, ~49,500 turns
  mb1        MemBench FirstAgent, the first 125 trajectories of each of the
             seven factual kinds (875 x ~231 turns, ~202,000 turns)

LongMemEval would have been the natural addition, but huggingface.co is not
reachable from the cloud, and MemBench FirstAgent has 3,500 trajectories; the
subset keeps the total near 10x rather than 20x.

Corpus families are weighted equally, as in PR #12: LoCoMo, REALTALK, BEAM
(all four scales) and MemBench (both). Small-talk negatives are unchanged.

Candidates, StandardScaler + LogisticRegression on 15 hand features + MiniLM:

  pr12   PR #12's recipe: LoCoMo + REALTALK + BEAM 100K/500K + small talk
  10x    every corpus above + small talk
  beam   PR #12's recipe + BEAM 1M/10M, no MemBench (which addition helps?)
  10x-eq the 10x set, but every corpus weighs the same within its family, so
         BEAM 10M's 208,696 turns do not drown BEAM's other three scales

Protocol: 5 folds by conversation, stratified by corpus. Each conversation is
scored by models that never saw it, replayed through P1-S, and answer recall
is compared with a 95% CI clustered by conversation. Seeds 1+ deal the folds
from a shuffled order. MemBench ThirdAgent trajectories are ~200 tokens, so
nothing is ever forgotten at any budget and they are training data only.

Every replay result is appended to results/scale/replays.jsonl as it
finishes and every fold model is pickled, so a restarted run picks up where
it stopped (`--resume` is the default).

    python3 retrain_10x.py --embed-only          # hours: MiniLM, checkpointed
    python3 retrain_10x.py                       # folds, main budgets
    python3 retrain_10x.py --sweep               # + more budgets
    python3 retrain_10x.py --seeds 0,1,2         # + reshuffled folds
    python3 retrain_10x.py --report              # tables from replays.jsonl
    python3 retrain_10x.py --export              # ship 10x if it wins
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from memgate.data import load_locomo, load_realtalk, load_beam, load_membench
from memgate.harness import run_policy
from memgate.ingest import GuardedScorer, chunk, queries
from memgate.policies import MemGateSelectPolicy, RAGPolicy
from memgate.scoring import HeuristicScorer, LearnedScorer, features
from memgate.utils import backend_info, embed
from train_scorer import build_xy
from train_transfer import cluster_bootstrap
from stress_questions import AppPath
import retrain_scorer as R
import scale_cache

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "scale")
MODELS = os.path.join(HERE, "data", "cache", "models_10x")
CONTEXT = 2048
FOLDS = 5
LOADERS = {
    "locomo": load_locomo,
    "realtalk": load_realtalk,
    "beam": lambda: load_beam(scales=("100K",)),
    "beam500": lambda: load_beam(scales=("500K",)),
    "beam1m": lambda: load_beam(scales=("1M",)),
    "beam10m": lambda: load_beam(scales=("10M",)),
    "mb3": lambda: load_membench(agents=("ThirdAgent",)),
    "mb1": lambda: load_membench(agents=("FirstAgent",), per_kind=125),
}
PR12 = ("locomo", "realtalk", "beam", "beam500")
BEAM = PR12 + ("beam1m", "beam10m")
FAMILY = {"beam500": "beam", "beam1m": "beam", "beam10m": "beam",
          "mb1": "membench", "mb3": "membench"}
# The paper's budgets (S main), and the sweep. MemBench FirstAgent chats are
# ~3,600 tokens, so 2048 makes it forget about half.
S_MAIN = {"locomo": 4096, "realtalk": 4096, "beam": 26000, "beam500": 26000,
          "beam1m": 26000, "beam10m": 26000, "mb1": 2048}
S_SWEEP = {"locomo": (1024, 2048, 8192, 16384), "realtalk": (1024, 2048, 8192, 16384),
           "mb1": (512, 1024), "beam": (8000, 64000), "beam500": (8000, 64000),
           "beam1m": (8000, 64000), "beam10m": (8000, 64000)}
APP = ("locomo", "realtalk", "beam")          # chunked app path; see PR #12
EVAL = ("locomo", "realtalk", "beam", "beam500", "beam1m", "beam10m", "mb1")


def log(*a):
    print(*a, flush=True)


def load_all(names):
    return [(c, sid, ts, qs) for c in names for sid, ts, qs in LOADERS[c]()]


def all_texts(units, talk_text, app=APP):
    texts = [f"[{t.speaker}] {t.text}" for _c, _s, ts, _q in units for t in ts]
    texts += [f"[{spk}] {x}" for x in talk_text for spk in ("user", "assistant")]
    texts += talk_text
    texts += [q.text for *_x, qs in units for q in qs]
    for c, _s, ts, qs in units:
        if c in app:
            texts += [f"[{t.speaker}] {p}" for t in ts for p in chunk(t.text)]
            texts += [x for q in qs for x in queries(q.text)]
    return texts


def fold_of(units, seed):
    """Fold per unit, each corpus dealt round-robin; seed > 0 shuffles the
    order within each corpus first."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(units), int)
    for c in dict.fromkeys(u[0] for u in units):
        idx = [i for i, u in enumerate(units) if u[0] == c]
        if seed:
            idx = list(rng.permutation(idx))
        for k, i in enumerate(idx):
            out[i] = k % FOLDS
    return out


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:                # a line cut off mid-write
                    continue
                keys.add((r["seed"], r["unit"], r["setup"], r["S"]))
    return keys


def replay(pol, ts, qs):
    """Per question: answer recall (None where the answer is not in the
    evidence), and soft evidence recall (share of cited turns in the context),
    which every question with evidence has, so BEAM's n is ten times larger."""
    det = run_policy(pol, ts, qs, CONTEXT)["detail"]
    return ([None if d["answer"] is None else bool(d["answer"]) for d in det],
            [round(d["soft"], 4) for d in det])


def fit_eq(feats, units, idx, talk):
    """As R.fit, but every corpus weighs the same within its family, so BEAM
    10M's 208,696 turns (and 128 questions) do not swamp BEAM's other scales."""
    X = np.concatenate([feats[j][0] for j in idx])
    y = np.concatenate([feats[j][1] for j in idx])
    corp = np.concatenate([np.full(len(feats[j][1]), units[j][0]) for j in idx])
    fam_of = {c: FAMILY.get(c, c) for c in set(corp)}
    fams = set(fam_of.values())
    per_fam = {f: sum(1 for c in fam_of if fam_of[c] == f) for f in fams}
    n = {c: (corp == c).sum() for c in fam_of}
    w = np.array([len(y) / (len(fams) * per_fam[fam_of[c]] * n[c]) for c in corp])
    Xt, yt = talk
    X = np.concatenate([X, Xt])
    y = np.concatenate([y, yt])
    w = np.concatenate([w, np.full(len(yt), 0.05 * len(w) / len(yt))])
    m = R.make_model("lr")
    return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})


def fold_models(feats, units, tr, talk, seed, k):
    """The candidates fitted on fold k's training units, pickled."""
    os.makedirs(MODELS, exist_ok=True)
    out = {}
    for name, keep in (("pr12", PR12), ("beam", BEAM), ("10x", None), ("10x-eq", None)):
        fn = os.path.join(MODELS, f"seed{seed}_fold{k}_{name}.pkl")
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                out[name] = pickle.load(f)
            continue
        idx = [j for j in tr if keep is None or units[j][0] in keep]
        t0 = time.time()
        m = (fit_eq if name == "10x-eq" else R.fit)(feats, units, idx, talk)
        with open(fn + ".tmp", "wb") as f:
            pickle.dump(m, f)
        os.replace(fn + ".tmp", fn)
        log(f"    fitted {name} on {sum(len(feats[j][1]) for j in idx):,} turns "
            f"({time.time() - t0:.0f}s)")
        out[name] = m
    return out


def run(args, units, feats, talk):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "replays.jsonl")
    done = done_keys(path)
    t0 = time.time()
    for seed in [int(s) for s in args.seeds.split(",")]:
        fold = fold_of(units, seed)
        for k in range(FOLDS):
            tr = [j for j in range(len(units)) if fold[j] != k]
            te = [j for j in range(len(units)) if fold[j] == k
                  and units[j][0] in EVAL and units[j][3]]
            models = fold_models(feats, units, tr, talk, seed, k)
            for j in te:
                c, sid, ts, qs = units[j]
                budgets = [S_MAIN[c]] + (list(S_SWEEP[c]) if args.sweep and seed == 0 else [])
                for S in budgets:
                    runs = {"rag": lambda: RAGPolicy(budget=CONTEXT, store_budget=S),
                            "heuristic": lambda: MemGateSelectPolicy(
                                budget=CONTEXT, store_budget=S, scorer=HeuristicScorer())}
                    for name, m in models.items():
                        sc = LearnedScorer(model=m, use_embedding=True, embedder=embed)
                        runs[name] = (lambda sc=sc: MemGateSelectPolicy(
                            budget=CONTEXT, store_budget=S, scorer=sc))
                        if c in APP and S == S_MAIN[c] and name in ("pr12", "10x", "10x-eq") \
                                or c in args.app_extra and S == S_MAIN[c] and name in ("pr12", "10x-eq"):
                            runs[f"app:{name}"] = (lambda sc=sc: AppPath(MemGateSelectPolicy(
                                budget=CONTEXT, store_budget=S, scorer=GuardedScorer(sc))))
                    for setup, make in runs.items():
                        if seed and setup in ("rag", "heuristic") and \
                                (0, sid, setup, S) in done:
                            continue        # the same replay at every seed
                        if (seed, sid, setup, S) in done:
                            continue
                        a, soft = replay(make(), ts, qs)
                        with open(path, "a") as f:
                            f.write(json.dumps({"seed": seed, "fold": k, "corpus": c,
                                                "unit": sid, "setup": setup, "S": S,
                                                "answer": a, "soft": soft}) + "\n")
                        done.add((seed, sid, setup, S))
            log(f"  seed {seed} fold {k + 1}/{FOLDS} done ({(time.time() - t0) / 60:.0f} min)")


SETUPS = ("rag", "heuristic", "pr12", "beam", "10x", "10x-eq",
          "app:pr12", "app:10x", "app:10x-eq")


def report():
    """Tables with 95% CIs clustered by conversation, from replays.jsonl, for
    answer recall and soft evidence recall."""
    rows = [json.loads(l) for l in open(os.path.join(OUT, "replays.jsonl"))
            if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["corpus"], r["S"], r["setup"]), {})[r["unit"]] = r
    seeds = sorted({r["seed"] for r in rows})
    out = []

    def arr(seed, c, S, setup, metric):
        d = by.get((seed, c, S, setup)) or (by.get((0, c, S, setup))
                                           if setup in ("rag", "heuristic") else None)
        if not d:
            return None
        units = sorted(d)
        a = np.array([np.nan if x is None else float(x)
                      for u in units for x in d[u][metric]])
        conv = np.array([i for i, u in enumerate(units) for _ in d[u][metric]])
        return a, conv, units

    for metric in ("answer", "soft"):
        for seed in seeds:
            for c in EVAL:
                for S in sorted({r["S"] for r in rows if r["corpus"] == c}):
                    res = {}
                    for setup in SETUPS:
                        got = arr(seed, c, S, setup, metric)
                        if got is not None:
                            res[setup] = got
                    if "10x" not in res:
                        continue
                    for setup, (a, conv, units) in res.items():
                        row = {"metric": metric, "seed": seed, "corpus": c, "S": S,
                               "setup": setup, "value": float(np.nanmean(a) * 100),
                               "n": int((~np.isnan(a)).sum()), "convs": len(units)}
                        base = "app:pr12" if setup.startswith("app:") else "pr12"
                        if setup not in (base, "rag", "heuristic") and base in res:
                            b, _, bu = res[base]
                            if bu == units:
                                o, lo, hi = cluster_bootstrap(a, b, conv)
                                row.update(vs=base, diff=o, lo=lo, hi=hi,
                                           sig=bool(lo > 0 or hi < 0))
                        out.append(row)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in out:
        line = (f"{r['metric']:<6} seed {r['seed']}  {r['corpus']:<9} S={r['S']:<6} "
                f"{r['setup']:<11}{r['value']:6.1f}%  n={r['n']:<5} convs={r['convs']}")
        if "diff" in r:
            line += (f"   vs {r['vs']}: {r['diff']:+5.1f} [{r['lo']:+5.1f}, "
                     f"{r['hi']:+5.1f}] {'SIG' if r['sig'] else 'n.s.'}")
        log(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--app-extra", default="",
                    help="also run the app path on these corpora (e.g. beam1m); "
                         "its chunks are embedded as they come, so this is slow")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    args.app_extra = tuple(c for c in args.app_extra.split(",") if c)
    if args.report:
        report()
        return
    if not backend_info()["embedder"].startswith("all-MiniLM-L6-v2"):
        sys.exit("needs MiniLM")
    units = load_all(list(LOADERS))
    n = {c: sum(len(u[2]) for u in units if u[0] == c) for c in LOADERS}
    log("turns per corpus: " + ", ".join(f"{c} {v:,}" for c, v in n.items()))
    log(f"  PR #12 set: {sum(n[c] for c in PR12):,} turns; "
        f"10x set: {sum(n.values()):,} ({sum(n.values()) / sum(n[c] for c in PR12):.1f}x)")
    talk_text = R.small_talk()
    t0 = time.time()
    k = scale_cache.warm(all_texts(units, talk_text, APP + args.app_extra), log=log)
    log(f"embedded {k:,} new strings in {(time.time() - t0) / 60:.0f} min")
    if args.embed_only:
        return
    R.FAMILY.update(FAMILY)
    feats = [build_xy(ts, qs, True) for _c, _s, ts, qs in units]
    Xt = np.asarray([list(features(t, spk)) + list(embed(t))
                     for x in talk_text for spk in ("user", "assistant")
                     for t in (f"[{spk}] {x}", x)], np.float32)
    talk = (Xt, np.zeros(len(Xt), np.int8))
    if args.export:
        export(units, feats, talk)
        return
    run(args, units, feats, talk)
    report()


def export(units, feats, talk):
    """Fit 10x on everything, calibrate as PR #12 did, print the app's sanity
    lines, and write the shipped model."""
    full = R.fit(feats, units, list(range(len(units))), talk)
    prev = [i for i, u in enumerate(units) if u[0] in ("locomo", "realtalk")]
    R.calibrate(full, feats, prev)
    sc = LearnedScorer(model=full, use_embedding=True, embedder=embed)
    for p in ["Hi!", "ok thanks", "lol yes", "You're welcome!", "cool", "sure",
              "My flight to Delhi is on 14 December at 6:40 am.",
              "My budget for the laptop is 85,000 rupees.",
              "Actually the review moved to 8 November.",
              "My sister Priya lives in Pune and works at Infosys."]:
        log(f"    {sc.score(f'[user] {p}', 'user')[0]:.2f}  {p}")
    from memgate.pretrained import MODEL_PATH, load_learned
    s, lr = full[0], full[-1]
    y = np.concatenate([f[1] for f in feats])
    n = {c: sum(len(u[2]) for u in units if u[0] == c) for c in LOADERS}
    d = {"model": "StandardScaler + LogisticRegression(class_weight=balanced)",
         "trained_on": "LoCoMo (10) + REALTALK (10) + BEAM 100K (20) + 500K (35) "
                       "+ 1M (35) + 10M (10) + MemBench ThirdAgent (3,455) + "
                       "FirstAgent (875), every turn; label = cited as evidence "
                       "by any question; corpus families (LoCoMo, REALTALK, BEAM, "
                       "MemBench) weighted equally; plus "
                       f"{len(talk[1])} small-talk lines as negatives at 5% weight "
                       "(retrain_10x.py); logit rescaled so P1's thresholds "
                       "split LoCoMo + REALTALK as the PR #4 model did",
         "n_turns": int(len(y)), "turns_per_corpus": n,
         "positive_rate": float(y.mean()),
         "embedder": backend_info()["embedder"], "use_embedding": True,
         "feature_names": list(R.FEATURE_NAMES),
         "mean": s.mean_.tolist(), "scale": s.scale_.tolist(),
         "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    with open(MODEL_PATH, "w") as f:
        json.dump(d, f)
    X = feats[0][0][:500]
    ref = full.predict_proba(X)[:, 1]
    got = np.array([p for _, p in load_learned().model.predict_proba(X)])
    assert np.allclose(ref, got, atol=1e-6), "exported model disagrees with sklearn"
    log(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
