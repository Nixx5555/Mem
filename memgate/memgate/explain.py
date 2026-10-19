"""Why a message got the score it did, broken into the parts that add up to it.

Both shipped scorers are additive, so the breakdown is exact rather than an
approximation like SHAP:

    learned    logit = intercept + sum_i coef_i * (x_i - mean_i) / scale_i,
               then u = sigmoid(logit). The 15 hand features are listed one by
               one; the 384 MiniLM dimensions are summed into one "meaning"
               part, because no single dimension means anything on its own.
    heuristic  u = 0.15 + each rule that fired, clipped to [0, 1].

`explain()` returns the parts and checks that they reproduce `scorer.score()`,
so the page can never show a story that disagrees with the number.
"""
import math

from .pretrained import LinearModel
from .scoring import (HeuristicScorer, LearnedScorer, FEATURE_NAMES, features,
                      message_only, words, FILLER_PAT, DURABLE_HINTS, NUM_PAT,
                      MONEY_PAT, MONTH_PAT, PROPER_PAT, CORRECTION_HINTS,
                      CORRECTION_PHRASES)

# What each hand feature means, for a reader who has not seen scoring.py.
# Yes/no features get both readings: in a standardised model a feature that is
# OFF still moves the score (it sits below the training mean), and "is a
# question: +0.18" on a statement would read as a bug.
FEATURE_LABELS = {
    "n_words": "length: {:.0f} words",
    "n_chars": "length: {:.0f} characters",
    "has_digit": ("contains a number", "no numbers"),
    "has_month": ("mentions a month", "no month"),
    "has_money": ("mentions money", "no money"),
    "has_proper": ("names someone or something", "no names"),
    "is_question": ("is a question", "not a question"),
    "ends_period": ("ends like a statement", "doesn't end with a full stop"),
    "durable_hits": "durable words (name, deadline, prefer…): {:.0f}",
    "correction_hint": ("sounds like a correction", "not a correction"),
    "is_filler": ("is filler (ok, lol, thanks)", "not filler"),
    "speaker_assistant": ("said by the assistant", "said by the user"),
    "frac_stop": "share of short words: {:.0%}",
    "n_caps": "capital letters: {:.0f}",
    "has_time": ("mentions a time of day", "no time of day"),
}


def _label(name, value):
    lab = FEATURE_LABELS.get(name, name)
    if isinstance(lab, tuple):
        return lab[0] if value else lab[1]
    return lab.format(value)


MEANING = "what it is about (MiniLM meaning, 384 dims)"


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def _learned(sc, text, speaker):
    m = sc.model
    x = list(features(text, speaker))
    n_hand = len(x)
    if sc.use_embedding and sc._embed is not None:
        x += list(sc._embed(text))
    contrib = [c * (v - mu) / s for c, v, mu, s in
               zip(m.coef, x, m.mean, m.scale)]
    parts = [{"name": _label(n, x[i]), "feature": n,
              "value": float(x[i]), "effect": contrib[i]}
             for i, n in enumerate(FEATURE_NAMES)]
    if len(contrib) > n_hand:
        parts.append({"name": MEANING, "feature": "embedding",
                      "value": None, "effect": sum(contrib[n_hand:])})
    z = m.intercept + sum(contrib)
    return {"unit": "log-odds", "base": m.intercept, "base_name":
            "starting point (how often a turn is worth keeping)",
            "parts": parts, "total": z, "utility": _sigmoid(z)}


def _heuristic(sc, text, speaker):
    """Replays HeuristicScorer.score rule by rule. test_insights.py checks that
    this agrees with the scorer on every turn of the sample and test chats."""
    on = sc._on
    t = message_only(text).strip()
    parts = []
    if on("filler") and (FILLER_PAT.match(t) or len(words(t)) <= 2):
        return {"unit": "score", "base": 0.02, "base_name": "filler: fixed low score",
                "parts": [], "total": 0.02, "utility": 0.02}
    w = set(words(t))
    if on("durable"):
        hits = len(w & DURABLE_HINTS)
        if hits:
            parts.append({"name": f"durable words ×{hits}", "feature": "durable",
                          "effect": min(0.35, 0.18 * hits)})
    if on("numeric") and NUM_PAT.search(t):
        parts.append({"name": "contains a number", "feature": "numeric", "effect": 0.20})
    if on("temporal") and (MONEY_PAT.search(t) or MONTH_PAT.search(t)):
        parts.append({"name": "mentions a date or money", "feature": "temporal",
                      "effect": 0.15})
    if on("proper") and PROPER_PAT.search(t):
        parts.append({"name": "names someone or something", "feature": "proper",
                      "effect": 0.12})
    if on("question") and t.endswith("?"):
        parts.append({"name": "is a question", "feature": "question", "effect": -0.12})
    if on("speaker") and speaker == "assistant":
        parts.append({"name": "said by the assistant", "feature": "speaker",
                      "effect": -0.10})
    if on("update"):
        tl = t.lower()
        corrected = (bool(w & CORRECTION_HINTS)
                     or any(p in tl for p in CORRECTION_PHRASES))
        if corrected and (NUM_PAT.search(t) or MONTH_PAT.search(t)):
            parts.append({"name": "corrects an earlier fact", "feature": "update",
                          "effect": 0.10})
    total = 0.15 + sum(p["effect"] for p in parts)
    return {"unit": "score", "base": 0.15, "base_name": "base score",
            "parts": parts, "total": total, "utility": max(0.0, min(1.0, total))}


def explain(scorer, text, speaker="user", top=6):
    """The parts of `scorer`'s score for one message, largest effect first.

    Returns None for a scorer that is not additive (none ship today)."""
    u, label = scorer.score(text, speaker)
    if isinstance(scorer, LearnedScorer) and isinstance(scorer.model, LinearModel):
        out = _learned(scorer, text, speaker)
    elif isinstance(scorer, HeuristicScorer):
        out = _heuristic(scorer, text, speaker)
    else:
        return None
    if abs(out["utility"] - u) > 1e-6:      # the story must match the number
        raise AssertionError(f"explanation gives {out['utility']:.6f}, "
                             f"scorer gives {u:.6f} for {text!r}")
    parts = sorted((p for p in out["parts"] if abs(p["effect"]) > 1e-9),
                   key=lambda p: -abs(p["effect"]))
    shown, rest = parts[:top], parts[top:]
    if rest:
        shown.append({"name": f"{len(rest)} smaller effects", "feature": "rest",
                      "effect": sum(p["effect"] for p in rest)})
    for p in shown:
        p["effect"] = round(float(p["effect"]), 4)
    out.update(parts=shown, scorer=scorer.name, label=label, text=text,
               utility=round(float(u), 4), base=round(float(out["base"]), 4),
               total=round(float(out["total"]), 4))
    return out
