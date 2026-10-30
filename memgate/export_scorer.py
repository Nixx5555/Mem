#!/usr/bin/env python3
"""Fit the shipped learned scorer on every LoCoMo + REALTALK conversation.

Same model and features as train_transfer.py's "pooled" row (logistic
regression on the 15 hand features + MiniLM), fitted once on all 20
conversations and written as plain numbers to memgate/scorer_pooled_lr.json.
See memgate/pretrained.py for how it is loaded and selected.

Usage:
    python3 export_scorer.py                       # LoCoMo + REALTALK (PR #4)
    python3 export_scorer.py --with-beam --balance # + BEAM 100K/500K, each
                                                   # corpus family weighted equally
"""
import argparse
import json
import os
import sys

import numpy as np

from memgate.data import load_locomo, load_realtalk, load_beam
from memgate.pretrained import MODEL_PATH, load_learned
from memgate.scoring import FEATURE_NAMES
from memgate.utils import backend_info, warm_cache
from train_scorer import build_xy
from train_transfer import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-beam", action="store_true")
    ap.add_argument("--balance", action="store_true")
    args = ap.parse_args()
    info = backend_info()
    if not info["embedder"].startswith("all-MiniLM-L6-v2"):
        sys.exit(f"needs MiniLM, embedder is {info['embedder']}")
    parts = [("locomo", load_locomo()), ("realtalk", load_realtalk())]
    if args.with_beam:
        parts.append(("beam", load_beam(scales=("100K", "500K"))))
    data = [(fam, d) for fam, ds in parts for d in ds]
    warm_cache([f"[{t.speaker}] {t.text}" for _f, (_s, ts, _q) in data for t in ts])
    xy = [build_xy(ts, qs, True) for _f, (_s, ts, qs) in data]
    X = np.concatenate([a for a, _ in xy]); y = np.concatenate([b for _, b in xy])
    m = make_model("lr")
    if args.balance:
        # same weighting as train_transfer.py --balance
        fams = np.concatenate([np.full(len(b), f) for (f, _d), (_a, b) in zip(data, xy)])
        n = {f: (fams == f).sum() for f in set(fams)}
        w = np.array([len(y) / (len(n) * n[f]) for f in fams])
        m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    else:
        m.fit(X, y)
    names = " + ".join({"locomo": "LoCoMo (10)", "realtalk": "REALTALK (10)",
                        "beam": "BEAM 100K (20) + 500K (35)"}[f] for f, _ in parts)
    sc, lr = m[0], m[-1]
    d = {"model": "StandardScaler + LogisticRegression(class_weight=balanced)",
         "trained_on": names + ", every turn; label = cited as evidence by "
                       "any question" + ("; corpus families weighted equally"
                                         if args.balance else ""),
         "n_turns": int(len(y)), "positive_rate": float(y.mean()),
         "embedder": info["embedder"], "use_embedding": True,
         "feature_names": list(FEATURE_NAMES),
         "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
         "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump(d, f)
    # the JSON copy must score exactly what the sklearn model scores
    shipped = load_learned()
    ref = m.predict_proba(X[:500])[:, 1]
    got = np.array([p for _, p in shipped.model.predict_proba(X[:500])])
    assert np.allclose(ref, got, atol=1e-6), "exported model disagrees with sklearn"
    print(f"wrote {MODEL_PATH}  ({len(y)} turns, {y.mean():.3f} positive; "
          f"max |diff| vs sklearn {np.abs(ref-got).max():.1e})")


if __name__ == "__main__":
    sys.exit(main())
