#!/usr/bin/env python3
"""Run every test suite and print one summary.

    python run_tests.py           # all suites, quick
    python run_tests.py --full    # also re-runs the whole LoCoMo benchmark check
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    ("test_memgate.py", "Library: tiers, compression, budgets, leakage"),
    ("test_scorer.py", "Scorers: P1 rules and the learned scorer"),
    ("test_policy.py", "Memory policy: keep, forget, recall under a budget"),
    ("test_reader.py", "Chat model: backends, fallback, saved key"),
    ("test_app.py", "Chat app: any input, endpoints, saved chats"),
    ("test_ingest.py", "Any text: long pastes, files, other languages, code"),
    ("test_insights.py", "Score breakdown and the baseline comparison"),
    ("test_baselines.py", "MemGPT-style and Mem0-style baselines"),
    ("test_bench.py", "Benchmark check"),
]


def main():
    full = "--full" in sys.argv
    sys.path.insert(0, HERE)
    from memgate.utils import backend_info, embed
    embed("warm up")
    emb = backend_info()["embedder"]
    print(f"embedder: {emb}")
    if not emb.startswith("all-MiniLM-L6-v2"):
        print("  MiniLM did not load, so tests that need it will fail. "
              "Run setup again (python setup_models.py --check lists what's missing).")
    rows, total_p, total_f = [], 0, 0
    for script, area in SUITES:
        t0 = time.time()
        args = [sys.executable, script] + (["--full"] if full and script == "test_bench.py" else [])
        r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
        m = re.findall(r"(\d+) passed, (\d+) failed", r.stdout)
        p, f = map(int, m[-1]) if m else (0, 1)
        if r.returncode and not f:
            f = 1                                   # crashed before its summary
        total_p, total_f = total_p + p, total_f + f
        rows.append((area, script, p, f, time.time() - t0))
        if f:
            fails = [l for l in r.stdout.splitlines() if "FAIL" in l]
            print(f"\n{script}:\n" + "\n".join(fails or r.stderr.splitlines()[-8:]))
    print(f"\n{'Area':<52}{'pass':>6}{'fail':>6}{'secs':>7}")
    for area, script, p, f, t in rows:
        print(f"{area:<52}{p:>6}{f:>6}{t:>7.0f}")
    print(f"{'Total':<52}{total_p:>6}{total_f:>6}")
    print("\nALL PASSED" if not total_f else "\nSOME TESTS FAILED")
    return 1 if total_f else 0


if __name__ == "__main__":
    sys.exit(main())
