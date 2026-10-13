#!/usr/bin/env python3
"""Which model should the learned scorer be? Same folds, same features, same labels.

`train_transfer.py` fixes the model (logistic regression, train_scorer.fit) and
varies the training data. This holds the data fixed and varies the model, on
the protocols that matter for transfer (in-corpus, cross, pooled; see
train_transfer.py), and also records what each costs to run, since the scorer
is called once per turn on the write path of a live agent.

Intrinsic only (AUC / AP of "is this turn ever cited as evidence"), because it
is cheap enough to run every candidate on every fold. The winner is then run
end to end by `train_transfer.py --model <name>`.

Label-blindness is unaffected: every candidate sees the same text-and-speaker
features, fitted on folds that exclude the conversation being scored.

Usage:
    python3 compare_scorers.py --scorer-fix
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from memgate.utils import backend_info, warm_cache, reset_cache
from train_scorer import build_xy
from train_transfer import LOADERS, apply_month_fix, MODELS, make_model

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="locomo,realtalk")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--scorer-fix", action="store_true")
    ap.add_argument("--lme-limit", type=int, default=100)
    args = ap.parse_args()
    if args.scorer_fix:
        apply_month_fix()
    names = [c.strip() for c in args.corpora.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    print("\nMemGate — scorer model comparison (intrinsic, held-out conversations)")
    print("=" * 80)
    corpora = {c: LOADERS[c](args) for c in names}
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  scorer fix: {args.scorer_fix}")
    units = [(c, ts, qs) for c, d in corpora.items() for _s, ts, qs in d]
    reset_cache()
    warm_cache([t.text if args.scorer_fix else f"[{t.speaker}] {t.text}"
                for _c, ts, _q in units for t in ts])

    out = os.path.join(RESULTS, "scorer_models_" + "_".join(names)
                       + ("_tagmonthfix" if args.scorer_fix else "") + ".csv")
    rows = []
    for use_emb in (False, True):
        fs = "hand+MiniLM" if use_emb else "hand"
        feats = [build_xy(ts, qs, use_emb, message_only=args.scorer_fix)
                 for _c, ts, qs in units]
        for test_c in names:
            others = [o for o in names if o != test_c]
            protocols = ["in-corpus"] + (["cross", "pooled"] if others else [])
            for p in protocols:
                print(f"  {fs:<12} test {test_c:<9} {p:<10}", end="", flush=True)
                for mname in models:
                    aucs, aps, fit_s, inf_us = [], [], 0.0, []
                    cross_model = None
                    for i, (c, _ts, _qs) in enumerate(units):
                        if c != test_c:
                            continue
                        Xte, yte = feats[i]
                        if len(set(yte)) < 2:
                            continue
                        keep = {"in-corpus": lambda j: j != i and units[j][0] == test_c,
                                "cross": lambda j: units[j][0] != test_c,
                                "pooled": lambda j: j != i}[p]
                        assert not keep(i)
                        if p == "cross" and cross_model is not None:
                            m = cross_model    # same training set for every test unit
                        else:
                            idx = [j for j in range(len(units)) if keep(j)]
                            t0 = time.perf_counter()
                            m = make_model(mname).fit(
                                np.concatenate([feats[j][0] for j in idx]),
                                np.concatenate([feats[j][1] for j in idx]))
                            fit_s += time.perf_counter() - t0
                            if p == "cross":
                                cross_model = m
                        # per-turn latency: the write path scores one turn at a time
                        t0 = time.perf_counter()
                        for x in Xte[:200]:
                            m.predict_proba(x[None, :])
                        inf_us.append((time.perf_counter() - t0) / min(200, len(Xte)) * 1e6)
                        pr = m.predict_proba(Xte)[:, 1]
                        aucs.append(roc_auc_score(yte, pr))
                        aps.append(average_precision_score(yte, pr))
                    rows.append({"features": fs, "test_corpus": test_c, "protocol": p,
                                 "model": mname, "auc": np.mean(aucs),
                                 "auc_sd": np.std(aucs), "ap": np.mean(aps),
                                 "fit_s_total": fit_s,
                                 "us_per_turn": float(np.median(inf_us)),
                                 "scorer_fix": args.scorer_fix,
                                 "embedder": info["embedder"]})
                    print(f"  {mname} {np.mean(aucs):.3f}/{np.mean(aps):.3f}",
                          end="", flush=True)
                print()
                # written after every block, so a killed run keeps what it did
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
