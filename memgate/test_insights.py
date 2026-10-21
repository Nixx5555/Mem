#!/usr/bin/env python3
"""Tests for the app's "why this score" breakdown and the baseline comparison.
Run: python test_insights.py"""
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

os.environ.setdefault("MEMGATE_SCORER", "learned")

import app as A
from memgate.data import build_dataset
from memgate.explain import explain
from memgate.policies import MemGateSelectPolicy
from memgate.pretrained import load_learned
from memgate.reader import ChatModel
from memgate.scoring import HeuristicScorer, SIGNALS
from memgate.types import Turn

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


TEXTS = [(spk, f"[{spk}] {t}") for spk, t in A.SAMPLE]
_turns, _ = build_dataset(n_sessions=3, turns_per_session=30, n_facts=10)
TEXTS += [(t.speaker, f"[{t.speaker}] {t.text}") for t in _turns]
TEXTS += [("user", "[user] Actually the review moved to 8 November."),
          ("user", "[user] my budget is ₹40000 and the meeting is at 5pm?"),
          ("assistant", "[assistant] ok")]

print("-- the breakdown always adds up to the score --")
scorers = [HeuristicScorer()] + [HeuristicScorer(set(SIGNALS) - {s}) for s in SIGNALS]
learned = load_learned()
if learned is not None:
    scorers.append(learned)
else:
    print("  SKIP  learned scorer (MiniLM not available here)")
for sc in scorers:
    bad = []
    for spk, t in TEXTS:
        try:
            e = explain(sc, t, spk)
        except AssertionError as exc:
            bad.append(str(exc))
            continue
        u, lab = sc.score(t, spk)
        parts = e["base"] + sum(p["effect"] for p in e["parts"])
        if abs(parts - e["total"]) > 1e-3 or e["label"] != lab:
            bad.append(f"{t!r}: parts {parts:.4f} vs total {e['total']:.4f}")
    tag = sc.name + ("" if len(getattr(sc, "signals", SIGNALS)) == len(SIGNALS)
                     else f" minus {sorted(set(SIGNALS) - sc.signals)[0]}")
    check(f"{tag}: {len(TEXTS)} messages explained exactly", not bad, bad[:2])

if learned is not None:
    e = explain(learned, "[user] Actually the report deadline moved to 4 November.")
    names = [p["name"] for p in e["parts"]]
    check("learned: the meaning part is reported as one part",
          sum("MiniLM" in n for n in names) == 1, names)
    check("learned: features that are off read as off",
          "not a question" in names or "is a question" not in names, names)
    check("learned: at most 6 parts plus one 'smaller effects' bucket",
          len(e["parts"]) <= 7, len(e["parts"]))
h = explain(HeuristicScorer(), "[user] lol")
check("heuristic: filler is a fixed low score with no parts",
      h["utility"] == 0.02 and h["parts"] == [], h)

from memgate.ingest import GuardedScorer
g = GuardedScorer(learned if learned is not None else HeuristicScorer())
e = explain(g, "[user] 我的考试在十一月八日，地点是北京大学")
check("guarded: a guard that changes the score says so",
      e is not None and e.get("guard") and e["utility"] == 0.5
      and "unguarded" in e, e and {k: e.get(k) for k in ("guard", "utility")})
e = explain(g, "[user] My flight to Delhi is on 14 December at 6:40 am.")
check("guarded: ordinary English is the wrapped scorer's breakdown",
      e is not None and "guard" not in e
      and abs(e["utility"] - g.score("[user] My flight to Delhi is on 14 December at 6:40 am.")[0]) < 1e-4,
      e and {k: e.get(k) for k in ("guard", "utility", "label")})

class Notes(ChatModel):
    """Stands in for the chat model: a short reply saying what its notes held
    (a long one would itself push the compact store over S)."""
    def __init__(self):
        super().__init__("none")

    @property
    def enabled(self):
        return True

    @property
    def name(self):
        return "notes"

    def reply(self, notes, message):
        return f"NOTES {len(notes.splitlines())}, name={'Simar' in notes}"


print("\n-- the baselines hear the same chat under the same budgets --")
folder = tempfile.mkdtemp()
app = A.App(Notes(), A.Chats(folder))
app.new("p1s", 150, "compact")
app.sample()
out = app.say("When is my report due?")
cmp = out["compare"]
rows = cmp["rows"]
check("three memories compared", sorted(rows) == ["memgate", "rag", "window"], sorted(rows))
check("no baseline holds more than S",
      all(rows[k]["stored_tokens"] <= 150 for k in ("window", "rag")),
      {k: r["stored_tokens"] for k, r in rows.items()})
check("every context stays inside the context budget",
      all(r["tokens"] <= cmp["context_budget"] for r in rows.values()),
      {k: r["tokens"] for k, r in rows.items()})
check("the page gets notes, not raw prompt strings",
      all("text" not in r for r in rows.values()))
only = [n for n in rows["memgate"]["notes"] if n["only"]]
theirs = {t for k in ("window", "rag") for n in rows[k]["notes"] for t in n["turns"]}
check("MemGate recalls messages both baselines forgot", len(only) >= 1,
      [n["text"] for n in rows["memgate"]["notes"]])
check("'only MemGate' is exactly the notes no baseline sent",
      all(bool(set(n["turns"]) & theirs) != n["only"] for n in rows["memgate"]["notes"]))
check("the name said in message 2 is one of them",
      any("Simar" in n["text"] for n in only), [n["text"] for n in only])
check("and MemGate's reply was written from it", out["reply"].endswith("name=True"),
      out["reply"])
said = sum(A.count_tokens(f"[{m['speaker']}] {' '.join(m['text'].split())}")
           for m in app.session.transcript)
check("tokens said counts every message", out["said_tokens"] == said,
      (out["said_tokens"], said))

print("\n-- the comparison does not change MemGate itself --")
bare = MemGateSelectPolicy(store_budget=150, short_capacity=4, budget=512)
for i, m in enumerate(app.session.transcript):
    bare.observe(Turn(session=1, index=i, speaker=m["speaker"],
                      text=" ".join(m["text"].split())))
same = ([i.text for i in bare.store.long] ==
        [i.text for i in app.session.policy.store.long])
check("Tier 3 matches a bare policy fed the same messages", same)
q = "When is my report due?"
check("and so does the context it builds",
      bare.build_context(q, 512).text == app.session.policy.build_context(q, 512).text)

print("\n-- a reloaded chat rebuilds the baselines too --")
again = A.Chats(folder).load(app.chat["id"])["session"]
ctx, _ = again.context(q)
c2 = again.compare(q, ctx)
check("same baseline contexts after a reload",
      all(c2["rows"][k]["text"] == app.session.rivals[k].build_context(q, 512).text
          for k in ("window", "rag")))

print("\n-- over HTTP --")
srv = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(app))
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


st = app.full_state()
held = st["t3"][0]
code, w = post("/api/why", {"id": held["id"]})
check("why: a Tier 3 note is explained", code == 200 and w["parts"], (code, w))
check("why: it explains the score the note shows",
      code == 200 and abs(w["utility"] - held["utility"]) < 0.006, (w.get("utility"), held))
code, w = post("/api/why", {"id": "m999999999"})
check("why: an unknown note is a 404 with a reason",
      code == 404 and "no longer" in w["error"], (code, w))
code, w = post("/api/why", {"text": "my exam is on 3 May"})
check("why: any typed message can be scored",
      code == 200 and 0 <= w["utility"] <= 1 and w["text"].endswith("3 May"), (code, w))
code, w = post("/api/why", {"text": "   "})
check("why: an empty message is refused", code == 404, (code, w))

code, r = post("/api/rival", {"rival": "window"})
check("rival: answers from the sliding window's notes",
      code == 200 and r["reply"].startswith("NOTES")
      and r["reply"].endswith("name=False"), (code, r))
n_before = len(app.session.transcript)
code, r = post("/api/rival", {"rival": "rag"})
check("rival: asking a baseline stores nothing",
      code == 200 and len(app.session.transcript) == n_before)
code, r = post("/api/rival", {"rival": "memgate"})
check("rival: only baselines can be asked", code == 404, (code, r))

print("\n-- every kept note can be explained, in every memory mode --")
for mode in ("p1", "p1a"):
    app.new(mode, 150, "compact")
    app.sample()
    for t in ("My laptop is a ThinkPad with 16 GB of RAM.",
              "The viva panel has three professors.", "ok cool",
              "Remind me to email Prof. Rao on Friday at 10am."):
        app.say(t)
    st = app.full_state()
    notes = st["t2"] + st["t3"]
    codes = [post("/api/why", {"id": n["id"]})[0] for n in notes]
    merged = sum(n["label"] == "merged" for n in notes)
    check(f"{mode}: all {len(notes)} Tier 2/3 notes explained "
          f"({merged} merged gists)", notes and set(codes) == {200}, codes)

app.new("p1s", 150, "compact")
code, r = post("/api/rival", {"rival": "window"})
check("rival: a fresh chat says to ask something first",
      code == 404 and "ask" in r["error"], (code, r))
srv.shutdown()

print("\n" + "=" * 46)
print(f"  {passed} passed, {failed} failed")
print("=" * 46)
raise SystemExit(1 if failed else 0)
