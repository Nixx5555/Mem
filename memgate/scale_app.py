#!/usr/bin/env python3
"""The real app on chats of 500,000 to 10,000,000 tokens, and under load.

Part 1, BEAM through the app. Each BEAM chat is typed into app.App one user
message at a time, exactly as the page sends it (chunking, GuardedScorer,
query extraction, the saved chat file, the baselines it compares against).
The chat model is replaced by the chat's own next assistant turn, so the
memory sees the conversation BEAM recorded. After the last message every
probing question whose answer is in its evidence is asked as a message, and
the notes each memory would send the model (MemGate, P0 window, RAG) are
checked for the answer, the paper's answer-recall metric. The shipped scorer
(PR #12) was fitted on BEAM 100K and 500K, so 500K is in-sample and 1M and 10M
are chats it never saw. S = 26,000, the paper's BEAM budget, and the app's
default 4,096.

Part 2, load, over HTTP to a real server:
  * one chat typed to 5,000 messages: latency of the first and last 100;
  * a whole BEAM 1M chat (~1M tokens) typed into one chat: latency, the saved
    file's size, the process's peak memory, and the time to reopen it;
  * 200 visitors at once, 10 messages each, in public mode: errors, leaks
    between visitors, p50/p95 latency;
  * 100 messages sent into one chat at the same moment.

Results go to results/scale/app_*.json, one record per chat as it finishes,
so a restarted run skips finished chats.

    python3 scale_app.py --part beam --scales 500K,1M,10M
    python3 scale_app.py --part load
    python3 scale_app.py --part summary      # part 1's table, with CIs
"""
import argparse
import json
import os
import random
import resource
import tempfile
import threading
import time

os.environ.setdefault("MEMGATE_SCORER", "learned")

import app as A
from memgate.data import load_beam
from memgate.reader import ChatModel
from memgate.utils import answer_tokens
import scale_cache
from stress_inputs import Client, serve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "scale")


class Scripted(ChatModel):
    """Replies with the chat's recorded assistant turns, in order, and keeps
    the notes it was handed."""

    def __init__(self):
        super().__init__("none")
        self.queue, self.notes = [], None

    @property
    def enabled(self):
        return True

    @property
    def name(self):
        return "scripted"

    def reply(self, notes, message):
        self.notes = notes
        return self.queue.pop(0) if self.queue else "noted."


def pairs(turns):
    """(user message, recorded reply) in order; runs of one speaker are
    joined, as one person typing twice is one message to the app."""
    out, cur_u, cur_a = [], [], []
    for t in turns:
        if t.speaker == "user":
            if cur_a:
                out.append(("\n\n".join(cur_u), "\n\n".join(cur_a)))
                cur_u, cur_a = [], []
            cur_u.append(t.text)
        else:
            cur_a.append(t.text)
    if cur_u or cur_a:
        out.append(("\n\n".join(cur_u) or "ok", "\n\n".join(cur_a) or "noted."))
    return [(u, a) for u, a in out if u.strip()]


def has(answer, text):
    return set(answer_tokens(answer)) <= set(answer_tokens(text))


def beam_chat(sid, turns, qs, S):
    model = Scripted()
    app = A.App(model, A.Chats(tempfile.mkdtemp()))
    app.new("p1s", S, "full")
    t0 = time.time()
    msgs = pairs(turns)
    for u, a in msgs:
        model.queue = [a]
        app.say(u)
    typed = time.time() - t0
    sess = app.session
    said = sess.said_tokens
    rows = []
    for q in qs:
        if not q.answer_recoverable:
            continue
        model.queue = ["noted."]
        app.say(q.text)
        cmp = sess.last_compare["rows"]
        rows.append({"q": q.text[:120], "kind": q.kind,
                     **{k: has(q.answer, cmp[k]["text"]) for k in ("memgate", "window", "rag")}})
    return {"chat": sid, "S": S, "messages": len(msgs), "said_tokens": said,
            "stored_tokens": sess.state()["stored_tokens"],
            "type_s": round(typed, 1), "ms_per_message": round(typed / len(msgs) * 1000, 1),
            "questions": rows}


def part_beam(scales, budgets):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "app_beam.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["chat"], r["S"]))
            except ValueError:
                pass
    scale_cache.load_shards()
    for scale in scales:
        for sid, turns, qs in load_beam(scales=(scale,)):
            if not any(q.answer_recoverable for q in qs):
                continue
            for S in budgets:
                if (sid, S) in done:
                    continue
                r = beam_chat(sid, turns, qs, S)
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                n = len(r["questions"])
                print(f"  {sid} S={S}: {r['messages']} messages, "
                      f"{r['said_tokens']:,} tokens said, {r['ms_per_message']} ms/msg; "
                      + ", ".join(f"{k} {sum(x[k] for x in r['questions'])}/{n}"
                                  for k in ("memgate", "window", "rag")), flush=True)


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def part_load():
    out = []
    rnd = random.Random(5)
    words = open(os.path.join(HERE, "..", "REPORT.md")).read().split()

    def msg():
        i = rnd.randrange(len(words) - 120)
        return " ".join(words[i:i + rnd.choice([8, 25, 60, 120])])

    # 1. 5,000 messages in one chat
    model = Scripted()
    app = A.App(model, A.Chats(tempfile.mkdtemp()))
    srv, url = serve(app)
    c = Client(url)
    c.call("/api/chats/new", {"scale": "full"})
    lat = []
    for i in range(5000):
        st, _, secs = c.call("/api/chat", {"text": msg()})
        assert st == 200, st
        lat.append(secs)
    st, state, _ = c.call("/api/state")
    out.append({"test": "5,000 messages in one chat",
                "first_100_mean_s": round(sum(lat[:100]) / 100, 3),
                "last_100_mean_s": round(sum(lat[-100:]) / 100, 3),
                "p95_s": round(sorted(lat)[int(len(lat) * .95)], 3),
                "max_s": round(max(lat), 2),
                "stored_tokens": state.get("stored_tokens"),
                "store_budget": state.get("store_budget")})
    print(" ", out[-1], flush=True)
    srv.shutdown()

    # 2. a whole BEAM 1M chat typed into one chat, over HTTP
    scale_cache.load_shards()
    sid, turns, _qs = load_beam(scales=("1M",))[0]
    model = Scripted()
    folder = tempfile.mkdtemp()
    app = A.App(model, A.Chats(folder))
    srv, url = serve(app)
    c = Client(url)
    c.call("/api/chats/new", {"scale": "full", "store_budget": 26000})
    lat = []
    msgs = pairs(turns)
    for u, a in msgs:
        model.queue = [a]
        st, _, secs = c.call("/api/chat", {"text": u})
        assert st == 200, st
        lat.append(secs)
    st, state, _ = c.call("/api/state")
    fn = app.chats._path(app.chat["id"])
    t0 = time.time()
    app2 = A.App(Scripted(), A.Chats(folder))
    reopen = time.time() - t0
    same = app2.session.state()["stored_tokens"] == app.session.state()["stored_tokens"]
    out.append({"test": f"BEAM 1M chat ({sid}) typed into one chat",
                "messages": len(msgs), "said_tokens": state.get("said_tokens"),
                "stored_tokens": state.get("stored_tokens"),
                "first_100_mean_s": round(sum(lat[:100]) / 100, 3),
                "last_100_mean_s": round(sum(lat[-100:]) / 100, 3),
                "p95_s": round(sorted(lat)[int(len(lat) * .95)], 3),
                "max_s": round(max(lat), 2),
                "saved_file_mb": round(os.path.getsize(fn) / 2**20, 1),
                "reopen_s": round(reopen, 1), "reopened_same_state": same,
                "peak_rss_mb": round(peak_mb())})
    print(" ", out[-1], flush=True)
    srv.shutdown()

    # 3. 200 visitors at once, 10 messages each
    vis = A.Visitors(Scripted(), tempfile.mkdtemp())
    srv, url = serve(vis)
    errors, lat = [], []

    def visitor(k):
        cl = Client(url)
        for i in range(10):
            st, _o, secs = cl.call("/api/chat", {"text": f"Visitor {k} note {i}: my seat is {k}-{i}."})
            lat.append(secs)
            if st != 200:
                errors.append(st)
        st, o, _ = cl.call("/api/state")
        if not all(f"Visitor {k} " in m["text"] or m["speaker"] != "user"
                   for m in o.get("transcript", [])):
            errors.append("leak")

    t0 = time.time()
    ths = [threading.Thread(target=visitor, args=(k,)) for k in range(200)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    lat.sort()
    out.append({"test": "200 visitors x 10 messages at once",
                "wall_s": round(time.time() - t0, 1), "errors": len(errors),
                "leaks": errors.count("leak"), "messages": len(lat),
                "p50_s": round(lat[len(lat) // 2], 2),
                "p95_s": round(lat[int(len(lat) * .95)], 2)})
    print(" ", out[-1], flush=True)
    srv.shutdown()

    # 4. 100 messages into one chat at the same moment
    capp = A.App(Scripted(), A.Chats(tempfile.mkdtemp()))
    srv, url = serve(capp)
    c = Client(url)
    ths = [threading.Thread(target=lambda k=k: c.call(
        "/api/chat", {"text": f"Parallel note {k}: code {k * 11}."})) for k in range(100)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    tr = capp.session.transcript
    got = {m["text"] for m in tr if m["speaker"] == "user"}
    out.append({"test": "100 messages into one chat at once",
                "kept": sum(f"Parallel note {k}: code {k * 11}." in got for k in range(100)),
                "transcript": len(tr)})
    print(" ", out[-1], flush=True)
    srv.shutdown()
    with open(os.path.join(OUT, "app_load.json"), "w") as f:
        json.dump(out, f, indent=1)


def summary():
    """Part 1's table: recall per memory, and MemGate against each baseline
    with a 95% CI over chats and an exact McNemar test."""
    from math import comb
    import numpy as np
    rs = [json.loads(l) for l in open(os.path.join(OUT, "app_beam.jsonl")) if l.strip()]
    rng = np.random.default_rng(1)
    out = []
    for scale in ("500K", "1M", "10M"):
        r = [x for x in rs if f"-{scale}-" in x["chat"] and x["questions"]]
        if not r:
            continue
        q = [y for x in r for y in x["questions"]]
        row = {"scale": scale, "chats": len(r), "questions": len(q),
               "said_tokens_mean": round(float(np.mean([x["said_tokens"] for x in r]))),
               "ms_per_message_mean": round(float(np.mean([x["ms_per_message"] for x in r]))),
               "max_stored_tokens": max(x["stored_tokens"] for x in r),
               **{k: round(100 * sum(y[k] for y in q) / len(q), 1)
                  for k in ("memgate", "window", "rag")}}
        for base in ("rag", "window"):
            groups = [np.array([y["memgate"] - y[base] for y in x["questions"]], float)
                      for x in r]
            means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                     for _ in range(10000)]
            lo, hi = np.percentile(means, [2.5, 97.5])
            b = sum(1 for y in q if y["memgate"] and not y[base])
            c = sum(1 for y in q if y[base] and not y["memgate"])
            p = min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
            row[f"vs_{base}"] = {"diff": round(100 * float(np.concatenate(groups).mean()), 1),
                                 "ci": [round(100 * lo, 1), round(100 * hi, 1)],
                                 "memgate_only": b, f"{base}_only": c, "mcnemar_p": p}
        out.append(row)
        print(json.dumps(row))
    with open(os.path.join(OUT, "app_beam_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="beam")
    ap.add_argument("--scales", default="500K,1M,10M")
    ap.add_argument("--budgets", default="26000,4096")
    args = ap.parse_args()
    if args.part == "beam":
        part_beam(args.scales.split(","), [int(b) for b in args.budgets.split(",")])
    elif args.part == "summary":
        summary()
    else:
        part_load()


if __name__ == "__main__":
    main()
