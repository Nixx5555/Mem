#!/usr/bin/env python3
"""Benchmark-check tests: the app's Benchmark check button and the numbers it
compares against.
    python test_bench.py          # quick (seconds)
    python test_bench.py --full   # also re-runs the whole LoCoMo check (1-2 min)"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer

import app as A
import run_live_check as L
from memgate.reader import ChatModel
from memgate.data import DEFAULT_LOCOMO_PATH

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("\n-- the committed numbers --")
ref = L.committed(4096)
check("S = 4096 has both arms", set(ref) >= {"RAG store-all", "P1-S MemGate select"},
      str(ref))
check("RAG is 28.1% and P1-S 37.5%, as on the slides",
      round(ref["RAG store-all"] * 100, 1) == 28.1
      and round(ref["P1-S MemGate select"] * 100, 1) == 37.5, str(ref))
check("selection beats FIFO by +9.4 points",
      round((ref["P1-S MemGate select"] - ref["RAG store-all"]) * 100, 1) == 9.4)
check("an unknown budget has no committed numbers", L.committed(12345) == {})
os.environ["MEMGATE_SCORER"] = "learned"
p = L.ARMS[1][1](budget=L.CONTEXT_BUDGET, store_budget=4096)
check("the check pins P1-S to the heuristic even when the app uses learned",
      p.scorer.name == "P1-heuristic", p.scorer.name)

print("\n-- the Benchmark check button --")
fake = tempfile.mkdtemp()
with open(os.path.join(fake, "run_live_check.py"), "w") as fh:
    fh.write("import sys, time\n"
             "print('MemGate live check', sys.argv[1:])\n"
             "print('Warning: noisy library line')\n"
             "time.sleep(1)\n"
             "print('  P1-S MemGate select  live  37.5%   committed  37.5%   MATCH')\n")
real_here = A.HERE
A.HERE = fake
try:
    b = A.Bench()
    check("starts", b.start(2048) is True)
    check("a second press while running is ignored", b.start(2048) is False)
    check("reports running", b.state()["running"] is True)
    for _ in range(100):
        if not b.state()["running"] and b.state()["code"] is not None:
            break
        time.sleep(0.1)
    st = b.state()
    check("finishes with exit code 0", st["code"] == 0 and not st["running"], str(st))
    check("passes the store budget through",
          "['--store', '2048']" in st["lines"][0], st["lines"][0])
    check("library warnings are filtered out",
          not any(l.startswith("Warning") for l in st["lines"]))
    check("the MATCH line reaches the page", any("MATCH" in l for l in st["lines"]))
    check("can run again after finishing", b.start(4096) is True)
    b.proc.wait()

    app = A.App(ChatModel("none"), A.Chats(tempfile.mkdtemp()))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/bench"
    with urllib.request.urlopen(base) as r:
        idle = json.load(r)
    check("GET /api/bench before a run is idle",
          idle == {"running": False, "lines": [], "code": None})
    req = urllib.request.Request(base, data=b'{"store": 4096}', method="POST")
    with urllib.request.urlopen(req) as r:
        check("POST /api/bench starts a run", json.load(r)["running"] is True)
    app.bench.proc.wait()
    srv.shutdown()
finally:
    A.HERE = real_here

if not os.path.exists(DEFAULT_LOCOMO_PATH):
    print("  SKIP  LoCoMo checks (data/locomo/locomo10.json not fetched)")
else:
    print("\n-- one LoCoMo conversation, twice --")
    from memgate.data import load_locomo
    from memgate.harness import run_policy
    sid, turns, qs = next(d for d in load_locomo() if d[2])
    runs = []
    for _ in range(2):
        for name, factory in L.ARMS:
            r = run_policy(factory(budget=L.CONTEXT_BUDGET, store_budget=4096),
                           turns, qs, L.CONTEXT_BUDGET)
            runs.append((name, r["answer_hits"], r["answer_n"]))
    check("the pipeline is deterministic", runs[:2] == runs[2:], str(runs))
    check("every question is scored", all(n == runs[0][2] for _, _, n in runs))

    if "--full" in sys.argv:
        print("\n-- the whole benchmark check (S = 4096) --")
        out = io.StringIO()
        sys.argv = ["run_live_check.py", "--store", "4096"]
        with redirect_stdout(out):
            code = L.main()
        lines = [l for l in out.getvalue().splitlines() if "MATCH" in l or "DIFFERS" in l]
        print("\n".join("        " + l.strip() for l in lines))
        check("both arms MATCH the committed results", code == 0
              and len(lines) == 2 and all("MATCH" in l for l in lines))
    else:
        print("  SKIP  the whole benchmark check (run with --full)")

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
