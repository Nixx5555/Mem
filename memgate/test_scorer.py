#!/usr/bin/env python3
"""Scorer tests, kept apart from test_memgate.py. Run: python test_scorer.py"""
import sys
from memgate.scoring import HeuristicScorer

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("\n-- scorer --")
s = HeuristicScorer()
check("question scores below the same statement",
      s.score("Is the deadline March 15?")[0] < s.score("The deadline is March 15.")[0])
check("assistant turn scores below the same user turn",
      s.score("The deadline is March 15.", "assistant")[0]
      < s.score("The deadline is March 15.", "user")[0])
check("dated correction labelled update",
      s.score("Actually the deadline moved to April 2.")[1] == "update")
check("utility stays in [0, 1]",
      all(0.0 <= s.score(x)[0] <= 1.0 for x in
          ("", "?", "ok", "My budget is 40000 rupees, due March 15, call Priya, "
           "my email and phone and address never changed.")))

# Every signal must be switchable on its own, or the ablation table cannot
# attribute anything to it.
from memgate.scoring import SIGNALS, RecencyScorer
check("each signal can be ablated alone",
      all(isinstance(HeuristicScorer(signals=set(SIGNALS) - {g}).score(
          "Actually my budget moved to 500 dollars in May, Priya said?")[0], float)
          for g in SIGNALS))
check("filler ablation stops the filler shortcut",
      HeuristicScorer(signals=set(SIGNALS) - {"filler"}).score("ok thanks")[1]
      != "filler")
check("P0 recency scorer is content-blind",
      RecencyScorer().score("The deadline is March 15.") == RecencyScorer().score("hey"))

# The deck and the paper quote the heuristic's evidence/non-evidence separation
# on LoCoMo (+0.0476, SESSION_RECORD §30.3) as the reason it transfers badly.
# Pin it, so a change to the scorer cannot silently move that number.
import os as _os_s
from memgate.data import DEFAULT_LOCOMO_PATH as _LP
if _os_s.path.exists(_LP):
    from memgate.data import load_locomo as _ll
    _ev, _ne = [], []
    for _sid, _ts, _qs in _ll():
        _cited = {e for q in _qs for e in q.evidence_ids}
        for _t in _ts:
            (_ev if _t.fact_id in _cited else _ne).append(s.score(_t.text, _t.speaker)[0])
    _sep = sum(_ev) / len(_ev) - sum(_ne) / len(_ne)
    check("LoCoMo evidence separation is +0.0476 (§30.3)",
          abs(_sep - 0.0476) < 5e-4, f"got {_sep:+.4f}")
else:
    print("  SKIP  LoCoMo separation (data/locomo/locomo10.json not fetched)")

print("\n-- the learned scorer (what the app uses) --")
import json as _json
import tempfile as _tf
import memgate.utils as _U
from memgate import pretrained as _P
_os_s.environ["MEMGATE_SCORER"] = "learned"
L = _P.default_scorer()
check("MEMGATE_SCORER=learned gives P3-learned", L.name == "P3-learned", L.name)
_os_s.environ["MEMGATE_SCORER"] = "heuristic"
check("MEMGATE_SCORER=heuristic gives P1", _P.default_scorer().name == "P1-heuristic")
del _os_s.environ["MEMGATE_SCORER"]
check("unset gives P1, so the experiments keep their meaning",
      _P.default_scorer().name == "P1-heuristic")

ODD = ["", " ", "?", "😀😀😀", "मेरा नाम सिमर है", "我的考试在十一月八日",
       "x" * 20000, "word " * 3000, "null\x00byte \u202e", "```py\nx = 1\n```",
       "<script>alert(1)</script>", "₹40,000.50 on 2026-11-08 at 14:30"]
for name, sc in (("learned", L), ("heuristic", s)):
    out = [sc.score(t, spk) for t in ODD for spk in ("user", "assistant")]
    check(f"{name}: any input gives a utility in [0, 1] and a label",
          all(isinstance(u, float) and 0.0 <= u <= 1.0 and isinstance(lab, str)
              and lab for u, lab in out))
check("learned scorer is deterministic",
      [L.score(t) for t in ODD] == [L.score(t) for t in ODD])
FACT = ["My flight to Delhi is on 14 December at 6:40 am.",
        "My budget for the laptop is 85,000 rupees.",
        "Actually the review moved to 8 November.",
        "My sister Priya lives in Pune and works at Infosys."]
CHAT = ["ok thanks", "You're welcome!", "lol yes", "cool", "Hi!", "sure"]
mean = lambda xs: sum(xs) / len(xs)
check("learned scorer ranks facts above small talk on average",
      mean([L.score(t)[0] for t in FACT]) > mean([L.score(t)[0] for t in CHAT]) + 0.2)
check("every fact outscores every greeting/thanks line for the learned scorer",
      min(L.score(t)[0] for t in FACT) > max(L.score(t)[0] for t in CHAT[:3]))

print("\n-- learned scorer falls back to P1, loudly --")
_d = _tf.mkdtemp()
check("missing model file", _P.load_learned(_os_s.path.join(_d, "nope.json")) is None)
with open(_P.MODEL_PATH) as _fh:
    _m = _json.load(_fh)
_bad = dict(_m, feature_names=["something", "else"])
_bp = _os_s.path.join(_d, "bad.json")
with open(_bp, "w") as _fh:
    _json.dump(_bad, _fh)
check("model fitted on different features", _P.load_learned(_bp) is None)
_real_bi = _U.backend_info
_U.backend_info = lambda: dict(_real_bi(), embedder="hash-bow/384d")
try:
    check("embedder is not MiniLM", _P.load_learned() is None)
finally:
    _U.backend_info = _real_bi
check("the shipped model file is plain numbers, not a pickle",
      {"mean", "scale", "coef", "intercept", "feature_names"} <= set(_m)
      and len(_m["coef"]) == len(_m["mean"]) == len(_m["scale"]))

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
