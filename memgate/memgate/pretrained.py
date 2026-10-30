"""The shipped learned scorer: logistic regression fitted on LoCoMo, REALTALK
and BEAM 100K/500K (families weighted equally) plus generated small talk as
negatives; written by retrain_scorer.py (docs/STRESS_TEST.md §6). Until
21 October 2026 it was the LoCoMo + REALTALK model of export_scorer.py.

train_transfer.py showed a scorer fitted on OTHER conversations beats the P1
heuristic on a conversation it has never seen (the "pooled" protocol), so a
model fitted once on every conversation is usable on a new chat with no
labels at run time. `export_scorer.py` writes it to memgate/scorer_pooled_lr.json
as plain numbers -- scaler mean/scale, coefficients, intercept -- rather than a
pickle, so loading it runs no code and the file diffs readably.

Selection is by MEMGATE_SCORER:
    heuristic  P1, the rule scorer (the library default, so every experiment
               script keeps meaning what REPORT.md says it means)
    learned    this model; falls back to P1, loudly, when the model file is
               missing or MiniLM is not the embedder (its 384 inputs are MiniLM
               dimensions, so hashing vectors would be silently meaningless)
"""
import json
import math
import os
import sys

from .scoring import HeuristicScorer, LearnedScorer, FEATURE_NAMES

# Beside this file, not in models/: models/ holds downloads and is gitignored,
# this is 26 KB of the project's own output and is committed.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scorer_pooled_lr.json")


class LinearModel:
    """predict_proba for a StandardScaler + LogisticRegression, from numbers."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = mean, scale
        self.coef, self.intercept = coef, intercept

    def predict_proba(self, X):
        out = []
        for x in X:
            z = self.intercept + sum(
                c * (v - m) / s for c, v, m, s in
                zip(self.coef, x, self.mean, self.scale))
            p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
            out.append([1.0 - p, p])
        return out


def load_learned(path: str = MODEL_PATH):
    """The shipped LearnedScorer, or None (with the reason on stderr)."""
    from .utils import backend_info, embed
    try:
        with open(path) as f:
            d = json.load(f)
    except OSError as exc:
        print(f"  [MemGate] learned scorer unavailable ({exc}); using P1.",
              file=sys.stderr)
        return None
    if d.get("feature_names") != list(FEATURE_NAMES):
        print("  [MemGate] learned scorer was fitted on different hand features; "
              "using P1.", file=sys.stderr)
        return None
    if d["use_embedding"] and not backend_info()["embedder"].startswith(
            "all-MiniLM-L6-v2"):
        print("  [MemGate] learned scorer needs MiniLM, embedder is "
              f"{backend_info()['embedder']}; using P1.", file=sys.stderr)
        return None
    model = LinearModel(d["mean"], d["scale"], d["coef"], d["intercept"])
    return LearnedScorer(model=model, use_embedding=d["use_embedding"],
                         embedder=embed)


def default_scorer():
    """The scorer MEMGATE_SCORER asks for (P1 when unset)."""
    if os.environ.get("MEMGATE_SCORER", "heuristic").lower() == "learned":
        sc = load_learned()
        if sc is not None:
            return sc
    return HeuristicScorer()
