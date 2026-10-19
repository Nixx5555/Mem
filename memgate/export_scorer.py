#!/usr/bin/env python3
"""Fit the shipped learned scorer on every LoCoMo + REALTALK conversation.

Same model and features as train_transfer.py's "pooled" row (logistic
regression on the 15 hand features + MiniLM), fitted once on all 20
conversations and written as plain numbers to memgate/scorer_pooled_lr.json.
See memgate/pretrained.py for how it is loaded and selected.

Usage:
    python3 export_scorer.py
"""
import json
import os
import sys

import numpy as np

from memgate.data import load_locomo, load_realtalk
from memgate.pretrained import MODEL_PATH, load_learned
from memgate.scoring import FEATURE_NAMES
from memgate.utils import backend_info, warm_cache
from train_scorer import build_xy
from train_transfer import make_model


def main():
    info = backend_info()
    if not info["embedder"].startswith("all-MiniLM-L6-v2"):
        sys.exit(f"needs MiniLM, embedder is {info['embedder']}")
    data = load_locomo() + load_realtalk()
    warm_cache([f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts])
    xy = [build_xy(ts, qs, True) for _s, ts, qs in data]
    X = np.concatenate([a for a, _ in xy]); y = np.concatenate([b for _, b in xy])
    m = make_model("lr").fit(X, y)
    sc, lr = m[0], m[-1]
    d = {"model": "StandardScaler + LogisticRegression(class_weight=balanced)",
         "trained_on": "LoCoMo (10) + REALTALK (10), every turn; label = cited "
                       "as evidence by any question",
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
