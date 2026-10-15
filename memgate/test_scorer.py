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

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
