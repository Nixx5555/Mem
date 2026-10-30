#!/usr/bin/env python3
"""Retrain the shipped scorer on more, and more varied, data, and test it
on conversations it never saw.

stress_questions.py found the shipped scorer (LoCoMo + REALTALK, PR #4) does
WORSE than plain retrieval on BEAM 500K, which it was never trained on:
21.8% answer recall against RAG's 32.7%. PR #8 showed that adding BEAM to the
training set fixes BEAM without costing LoCoMo, but that model was not shipped
because it scored "Hi!" (0.68) above a flight date (0.12).

Candidates, all StandardScaler + LogisticRegression on the 15 hand features +
MiniLM (train_transfer.py's "lr"), corpus families weighted equally:

  shipped    LoCoMo + REALTALK                         (what the app runs now)
  beam       + BEAM 100K + 500K                        (PR #8's model)
  beam+talk  + BEAM, + small talk as labelled negatives (small_talk() below)

Small talk. BEAM has almost none ("Hi!" never occurs), so nothing teaches the
model that a greeting is worth nothing; LoCoMo and REALTALK teach it only
through the turns around it. small_talk() is 923 greetings, thanks and
reactions written from templates, every one labelled "not evidence", which is
what LoCoMo's and REALTALK's own small talk is labelled too. It is generated,
not taken from any test conversation, so it cannot leak a label.

Protocol: 5 folds by conversation, stratified by corpus (every fold holds out
a fifth of each corpus). A conversation is scored only by a model fitted on
the other four folds, then replayed through P1-S at the paper's budgets
(context 2048; S = 4096 for LoCoMo/REALTALK, 26000 for BEAM), and answer
recall is compared with RAG and P1 on the same questions, with a 95% CI
clustered by conversation. The app path (GuardedScorer + chunk + queries) is
run too, on every corpus but BEAM 500K (see stress_questions.py for why).

    python3 retrain_scorer.py              # evaluate the three candidates
    python3 retrain_scorer.py --export     # ... and ship beam+talk if it wins
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from memgate.data import load_locomo, load_realtalk, load_beam
from memgate.harness import run_policy
from memgate.ingest import GuardedScorer, chunk, focus, queries
from memgate.policies import MemGateSelectPolicy, RAGPolicy
from memgate.scoring import HeuristicScorer, LearnedScorer, FEATURE_NAMES
from memgate.utils import backend_info, embed, warm_cache
from train_scorer import build_xy
from train_transfer import make_model, cluster_bootstrap
from stress_questions import AppPath

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "stress")
CONTEXT = 2048
S = {"locomo": 4096, "realtalk": 4096, "beam": 26000, "beam500": 26000}
FAMILY = {"beam500": "beam"}
FOLDS = 5


def small_talk():
    """Greetings, thanks and reactions: nothing a later question needs."""
    hello = ["hi", "hey", "hello", "hiya", "yo", "good morning", "good evening",
             "good night", "hey there", "hi there", "morning", "sup"]
    thanks = ["thanks", "thank you", "thanks a lot", "thank you so much", "ty",
              "thx", "cheers", "much appreciated", "thanks!", "you're welcome",
              "no problem", "anytime", "my pleasure", "no worries"]
    react = ["ok", "okay", "cool", "nice", "great", "awesome", "lol", "haha",
             "wow", "sure", "yes", "yeah", "yep", "no", "nope", "got it",
             "makes sense", "sounds good", "perfect", "alright", "I see",
             "oh nice", "that's great", "fair enough", "true", "right",
             "interesting", "oh wow", "same here", "me too", "hmm", "ok cool",
             "great, thanks", "love it", "agreed", "bye", "see you", "talk later",
             "take care", "have a good day", "how are you?", "what's up?",
             "how's it going?", "all good?", "you there?"]
    tails = ["", "!", "!!", ".", " :)", " 😊", "?"]
    out = set()
    for w, t in itertools.product(hello + thanks + react, tails):
        if not (w.endswith("?") and t):
            out.add(w + t)
            out.add((w + t).capitalize())
    return sorted(out)


def load(names):
    loaders = {"locomo": load_locomo, "realtalk": load_realtalk,
               "beam": lambda: load_beam(scales=("100K",)),
               "beam500": lambda: load_beam(scales=("500K",))}
    return [(c, sid, ts, qs) for c in names for sid, ts, qs in loaders[c]()]


def folds_of(units):
    """Fold index per unit: each corpus dealt round-robin into FOLDS folds."""
    out, seen = [], {}
    for c, *_ in units:
        out.append(seen.get(c, 0) % FOLDS)
        seen[c] = seen.get(c, 0) + 1
    return np.array(out)


def fit(feats, units, idx, talk=None):
    """LR on units `idx`, corpus families weighted equally; `talk` adds the
    small-talk negatives as a family of its own at a twentieth the weight."""
    X = np.concatenate([feats[j][0] for j in idx])
    y = np.concatenate([feats[j][1] for j in idx])
    fam = np.concatenate([np.full(len(feats[j][1]), FAMILY.get(units[j][0], units[j][0]))
                          for j in idx])
    n = {f: (fam == f).sum() for f in set(fam)}
    w = np.array([len(y) / (len(n) * n[f]) for f in fam])
    if talk is not None:
        Xt, yt = talk
        X = np.concatenate([X, Xt])
        y = np.concatenate([y, yt])
        w = np.concatenate([w, np.full(len(yt), 0.05 * len(w) / len(yt))])
    m = make_model("lr")
    return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})


def outcomes(r):
    return np.array([np.nan if d["answer"] is None else float(d["answer"])
                     for d in r["detail"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="locomo,realtalk,beam,beam500")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--skip-eval", action="store_true",
                    help="with --export: fit on everything and ship, no folds")
    args = ap.parse_args()
    if not backend_info()["embedder"].startswith("all-MiniLM-L6-v2"):
        sys.exit("needs MiniLM")
    names = args.corpora.split(",")
    units = load(names)
    talk_text = small_talk()
    t0 = time.time()
    texts = [f"[{t.speaker}] {t.text}" for _c, _s, ts, _q in units for t in ts]
    texts += [f"[{spk}] {x}" for x in talk_text for spk in ("user", "assistant")]
    texts += talk_text
    texts += [q.text for *_x, qs in units for q in qs]
    app_units = [i for i, u in enumerate(units) if u[0] != "beam500"] \
        if not (args.no_app or args.skip_eval) else []
    for i in app_units:
        _c, _s, ts, qs = units[i]
        texts += [f"[{t.speaker}] {p}" for t in ts for p in chunk(t.text)]
        texts += [x for q in qs for x in queries(q.text)]
    cache = os.path.join(HERE, "data", "cache", "minilm_retrain.pkl")
    from memgate import utils as U
    if os.path.exists(cache):                 # 40 minutes of MiniLM, kept
        import pickle
        with open(cache, "rb") as f:
            U._cache.update(pickle.load(f))
    warm_cache(texts)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    import pickle
    with open(cache, "wb") as f:
        pickle.dump(dict(U._cache), f)
    print(f"embedded {len(set(texts)):,} strings in {time.time() - t0:.0f}s")

    feats = [build_xy(ts, qs, True) for _c, _s, ts, qs in units]
    from memgate.scoring import features
    # tagged, as the policy scores a turn, and bare, as a caller might
    Xt = np.asarray([list(features(t, spk)) + list(embed(t))
                     for x in talk_text for spk in ("user", "assistant")
                     for t in (f"[{spk}] {x}", x)], np.float32)
    talk = (Xt, np.zeros(len(Xt), np.int8))
    fold = folds_of(units)
    prev = [i for i, u in enumerate(units) if u[0] in ("locomo", "realtalk")]
    cands = {
        "shipped": lambda tr: fit(feats, units, [j for j in tr if j in prev]),
        "beam": lambda tr: fit(feats, units, tr),
        "beam+talk": lambda tr: fit(feats, units, tr, talk),
    }
    res = {}          # (setup, corpus) -> list of arrays; conv ids alongside
    for k in range(0 if args.skip_eval else FOLDS):
        tr = [j for j in range(len(units)) if fold[j] != k]
        te = [j for j in range(len(units)) if fold[j] == k]
        models = {name: f(tr) for name, f in cands.items()}
        for j in te:
            c, sid, ts, qs = units[j]
            if not qs:
                continue
            runs = {"rag": RAGPolicy(budget=CONTEXT, store_budget=S[c]),
                    "heuristic": MemGateSelectPolicy(budget=CONTEXT, store_budget=S[c],
                                                     scorer=HeuristicScorer())}
            for name, m in models.items():
                sc = LearnedScorer(model=m, use_embedding=True, embedder=embed)
                runs[name] = MemGateSelectPolicy(budget=CONTEXT, store_budget=S[c],
                                                 scorer=sc)
                if j in app_units:
                    runs[f"app:{name}"] = AppPath(MemGateSelectPolicy(
                        budget=CONTEXT, store_budget=S[c], scorer=GuardedScorer(sc)))
            for setup, pol in runs.items():
                a = outcomes(run_policy(pol, ts, qs, CONTEXT))
                d = res.setdefault((setup, c), {"a": [], "conv": []})
                d["a"].append(a)
                d["conv"].append(np.full(len(a), j))
        print(f"  fold {k + 1}/{FOLDS} done ({time.time() - t0:.0f}s)", flush=True)

    rows, table = [], {}
    for (setup, c), d in sorted(res.items()):
        a, conv = np.concatenate(d["a"]), np.concatenate(d["conv"])
        table[(setup, c)] = (a, conv)
    print("\nanswer recall, held-out conversations (5-fold by conversation)")
    setups = sorted({s for s, _ in table}, key=lambda s: (s.startswith("app"), s))
    for c in names:
        print(f"\n  {c}")
        for s in setups:
            if (s, c) not in table:
                continue
            a, conv = table[(s, c)]
            line = f"    {s:<16}{np.nanmean(a) * 100:6.1f}%  (n={int((~np.isnan(a)).sum())})"
            base = "app:shipped" if s.startswith("app:") else "shipped"
            if s not in ("rag", "heuristic", base) and (base, c) in table:
                o, lo, hi = cluster_bootstrap(a, table[(base, c)][0], conv)
                sig = "SIG" if lo > 0 or hi < 0 else "n.s."
                line += f"   vs {base}: {o:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] {sig}"
            print(line)
            rows.append({"corpus": c, "setup": s, "answer": float(np.nanmean(a) * 100),
                         "n": int((~np.isnan(a)).sum())})
    os.makedirs(RESULTS, exist_ok=True)
    if rows:
        with open(os.path.join(RESULTS, "retrain_5fold.json"), "w") as f:
            json.dump(rows, f, indent=1)

    # the model fitted on everything, checked on the sanity lines the app tests
    full = cands["beam+talk"](list(range(len(units))))
    calibrate(full, feats, prev)
    sc = LearnedScorer(model=full, use_embedding=True, embedder=embed)
    probe = ["Hi!", "ok thanks", "lol yes", "You're welcome!", "cool", "sure",
             "My flight to Delhi is on 14 December at 6:40 am.",
             "My budget for the laptop is 85,000 rupees.",
             "Actually the review moved to 8 November.",
             "My sister Priya lives in Pune and works at Infosys."]
    print("\nbeam+talk fitted on everything:")
    for p in probe:
        print(f"    {sc.score(f'[user] {p}', 'user')[0]:.2f}  {p}")
    if args.export:
        export(full, units, feats, talk)


# Shares of the LoCoMo + REALTALK turns the PR #4 model (the shipped scorer
# before this script) put at or above P1's thresholds, measured once on that
# model so calibration does not depend on whichever model is shipped now.
PR4_SHARE = {0.45: 0.3648, 0.15: 0.7185}


def calibrate(m, feats, prev):
    """Put the new model's scores on the old shipped model's scale.

    P1-S only ORDERS by score, but P1's tau_fact = 0.45 / tau_low = 0.15 and
    GuardedScorer's 0.5 "neutral" read scores as absolute. An affine map of
    the logit is fitted so that the same shares of LoCoMo + REALTALK turns
    land above each threshold as with the PR #4 model (PR4_SHARE). It keeps
    every ranking, so every P1-S number above, and it is still a logistic
    regression: coef * a, intercept * a + b.
    """
    X = np.concatenate([feats[j][0] for j in prev])
    lr = m[-1]
    z = m[:-1].transform(X) @ lr.coef_[0] + lr.intercept_[0]
    (t1, s1), (t2, s2) = PR4_SHARE.items()
    q1, q2 = np.quantile(z, 1 - s1), np.quantile(z, 1 - s2)
    logit = lambda p: np.log(p / (1 - p))
    a = (logit(t1) - logit(t2)) / (q1 - q2)
    b = logit(t1) - a * q1
    lr.coef_ = lr.coef_ * a
    lr.intercept_ = lr.intercept_ * a + b
    print(f"calibrated: logit -> {a:.3f} * logit + {b:+.3f}")


def export(m, units, feats, talk):
    from memgate.pretrained import MODEL_PATH, load_learned
    sc, lr = m[0], m[-1]
    y = np.concatenate([f[1] for f in feats])
    d = {"model": "StandardScaler + LogisticRegression(class_weight=balanced)",
         "trained_on": "LoCoMo (10) + REALTALK (10) + BEAM 100K (20) + 500K (35), "
                       "every turn; label = cited as evidence by any question; "
                       "corpus families weighted equally; plus "
                       f"{len(talk[1])} small-talk lines as negatives at 5% weight "
                       "(retrain_scorer.py); logit rescaled so P1's thresholds "
                       "split LoCoMo + REALTALK as the PR #4 model did",
         "n_turns": int(len(y)), "positive_rate": float(y.mean()),
         "embedder": backend_info()["embedder"], "use_embedding": True,
         "feature_names": list(FEATURE_NAMES),
         "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
         "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    with open(MODEL_PATH, "w") as f:
        json.dump(d, f)
    X = feats[0][0][:500]
    ref = m.predict_proba(X)[:, 1]
    got = np.array([p for _, p in load_learned().model.predict_proba(X)])
    assert np.allclose(ref, got, atol=1e-6), "exported model disagrees with sklearn"
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
