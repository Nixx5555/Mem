#!/usr/bin/env python3
"""MemGate app: chat through MemGate and watch what each tier holds.

    python3 app.py                 # then open http://localhost:8765
    python3 app.py --llm claude    # answer with Claude (ANTHROPIC_API_KEY)
    python3 app.py --llm hf        # transformers, any OS
    python3 app.py --no-llm        # no model: replies show the context only
    python3 app.py --scorer heuristic   # the P1 rule scorer instead of the learned one
    python3 app.py --port 9000

Conversations are saved under memgate/chats/ and survive a restart: each one
is stored as its messages and rebuilt by replaying them through observe(),
which is deterministic, so the tiers come back exactly as they were.

Every message -- yours and the model's reply -- goes through the same
`policy.observe()` the experiments use, and every question is answered from
the same `policy.build_context()`: the model sees ONLY the memory MemGate
assembled, never the raw history. The page shows the three tiers as they are,
an event log of each routing decision (score, label, destination), and the
exact context sent with each question.

Each chat also feeds every message to the two baselines MemGate is measured
against, P0 sliding window and RAG store-all, at the same store budget S, so
the page can show the same question answered from all three memories. Any
note's score can be broken down into the parts that add up to it
(memgate/explain.py).

The benchmark panel runs `run_live_check.py` in a subprocess and streams its
output: the LoCoMo answer-recall numbers re-computed in front of the panel and
compared with the committed CSV.

Nothing here is a mock-up. Chats run at the sizes the experiments used. A
"compact" size, chosen per chat, shrinks the capacities so that eviction
happens within a few messages instead of a few hundred; those are the only
settings that differ from the experiments.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memgate.types import Turn, SHORT_TERM, WORKING, LONG_TERM
from memgate.policies import (MemGatePolicy, MemGateAdaptivePolicy,
                              MemGateSelectPolicy, SlidingWindowPolicy,
                              RAGPolicy)
from memgate.explain import explain
from memgate.reader import ChatModel, BACKENDS
from memgate.utils import backend_info, count_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "chat.html")          # the chat app
INSPECT = os.path.join(HERE, "web", "index.html")    # all tiers at once
CHATS = os.path.join(HERE, "chats")
# Longest message accepted, about 5,000 tokens: a long paste fits, and the
# local model still answers in reasonable time.
MAX_CHARS = 20000
TIER_KEY = {SHORT_TERM: "t1", WORKING: "t2", LONG_TERM: "t3"}


class NotFound(Exception):
    """A request the page can fix (unknown note, nothing asked yet): a 404."""

# (Tier 1 capacity, context budget, default store budget S), in messages and
# tokens. "full" is what the experiments ran with (policies.py defaults, the
# 2048-token context and S = 4096 of the headline result). "compact" shrinks
# them so that eviction is visible within a few messages instead of a few
# hundred. Chats saved under the old name "demo" open as "compact".
SCALES = {
    "full": (6, 2048, 4096),
    "compact": (4, 512, 150),
}
SCALE_ALIASES = {"demo": "compact"}

MODES = {
    "p1": "P1 MemGate · threshold routing (slide 6)",
    "p1s": "P1-S select · scored eviction under a store budget",
    "p1a": "P1-A adaptive · demote to a gist under pressure",
}

SAMPLE = [  # A scripted opening, loaded instantly, for asking questions about.
    ("user", "Hi!"),
    ("assistant", "Hello! How can I help you today?"),
    ("user", "My name is Simar and I'm a final-year student at NMIMS Indore."),
    ("assistant", "Nice to meet you, Simar. What are you working on?"),
    ("user", "My capstone final review is on 6 November and the report is due on 2 November."),
    ("assistant", "Got it: report due 2 November, final review on 6 November."),
    ("user", "ok thanks"),
    ("assistant", "You're welcome!"),
    ("user", "My guide wants the slides in the NMIMS template."),
    ("assistant", "Understood, the slides should use the NMIMS template."),
    ("user", "lol yes"),
    ("assistant", "Sure."),
    ("user", "Actually the report deadline moved to 4 November."),
    ("assistant", "Updated: the report is now due on 4 November."),
    ("user", "My teammate Yash runs the LongMemEval experiments."),
    ("assistant", "Noted, Yash is running LongMemEval."),
    ("user", "cool"),
    ("assistant", "Anything else?"),
]


# ---------------------------------------------------------------- session
def _item(it):
    return {"id": it.id, "text": it.text, "tier": TIER_KEY[it.tier],
            "kind": it.kind, "utility": round(float(it.utility), 2),
            "label": it.type_label, "turn": it.turn_index,
            "tokens": count_tokens(it.text),
            "superseded": it.superseded_by is not None}


class Session:
    def __init__(self, mode="p1s", store_budget=None, scale="full"):
        self.mode = mode if mode in MODES else "p1s"
        scale = SCALE_ALIASES.get(scale, scale)
        self.scale = scale if scale in SCALES else "full"
        self.short_capacity, self.context_budget, s_default = SCALES[self.scale]
        self.store_budget = int(store_budget or s_default)
        kw = dict(short_capacity=self.short_capacity, budget=self.context_budget)
        if self.mode == "p1":
            self.policy = MemGatePolicy(**kw)
        elif self.mode == "p1s":
            self.policy = MemGateSelectPolicy(store_budget=self.store_budget, **kw)
        else:
            self.policy = MemGateAdaptivePolicy(store_budget=self.store_budget, **kw)
        # The two baselines MemGate is measured against, fed the same messages
        # under the same store budget S and asked with the same context budget,
        # so a question can be answered from each memory side by side. They
        # are the experiments' own classes: P0 keeps the most recent S tokens,
        # RAG keeps every message verbatim and forgets the oldest past S.
        self.rivals = {"window": SlidingWindowPolicy(store_budget=self.store_budget),
                       "rag": RAGPolicy(store_budget=self.store_budget)}
        self.said_tokens = 0   # everything said so far, as memory sees it
        self.raw = {}          # message number -> (scored text, speaker), for "why"
        self.turn_of = {}      # item id -> the message numbers it was made from
        self.last_compare = None
        self.n = 0
        self.log = []          # every routing event, newest last
        self.transcript = []   # what was said, in order: the saved form
        self.lock = threading.Lock()

    # -- what the store holds right now
    def _held(self):
        st = self.policy.store
        return ({i.id: i for i in st.short}, {i.id: i for i in st.working},
                {i.id: i for i in st.long})

    def state(self):
        st = self.policy.store
        s = self.policy.stats()
        return {
            "mode": self.mode, "mode_label": MODES[self.mode],
            "store_budget": self.store_budget if self.mode != "p1" else None,
            "scale": self.scale,
            "short_capacity": self.short_capacity,
            "context_budget": self.context_budget,
            "tau_fact": self.policy.tau_fact, "tau_low": self.policy.tau_low,
            "t1": [_item(i) for i in st.short],
            "t2": [_item(i) for i in st.working],
            "t3": [_item(i) for i in st.long],
            "stored_tokens": s["stored_tokens"],
            "counts": dict({k: s[k] for k in ("dropped", "evicted", "demoted",
                                               "merges", "consolidations")},
                           compressed=self.policy.routed["working"] + s["demoted"]),
            "messages": self.n,
            "said_tokens": self.said_tokens,
        }

    # -- the write path, with every decision it made written to the log
    def add(self, speaker, text):
        # The transcript keeps the message as written (line breaks in code and
        # pasted paragraphs survive a reload); memory gets it on one line, the
        # form every experiment scored.
        shown = text.replace("\r\n", "\n").strip()
        text = " ".join(text.split())
        if not text:
            return []
        p, st = self.policy, self.policy.store
        s0, w0, l0 = self._held()
        live0 = {i.id for i in st.long if i.is_active}
        c0 = dict(p.routed, dropped=st.dropped, demoted=st.demoted,
                  merges=st.merges, reclaimed=st.reclaimed)

        self.transcript.append({"speaker": speaker, "text": shown})
        turn = Turn(session=1, index=self.n, speaker=speaker, text=text)
        self.n += 1
        p.observe(turn)                      # exactly the experiments' code path
        for r in self.rivals.values():
            r.observe(turn)

        s1, w1, l1 = self._held()
        new = next(i for i in st.short if i.id not in s0)
        self.raw[turn.index] = (new.text, speaker)
        self.turn_of[new.id] = {turn.index}
        # a gist or fact made just now inherits its sources' message numbers,
        # so a gist of three merged gists still names all the messages in it
        for it in list(w1.values()) + list(l1.values()):
            if it.id not in self.turn_of:
                src = [self.turn_of[k] for k in it.source_ids if k in self.turn_of]
                self.turn_of[it.id] = set().union(*src) if src else {it.turn_index}
        self.said_tokens += count_tokens(new.text)
        ev = [{"type": "enter", "tier": "t1", "text": new.text,
               "msg": f"entered Tier 1 (short-term buffer, holds the last "
                      f"{self.short_capacity} messages)"}]

        out = [s0[k] for k in s0 if k not in s1]
        if out:
            e = out[0]
            u, lab = e.utility, e.type_label
            score = f"score u = {u:.2f} ({lab})"
            went = [i for i in list(w1.values()) + list(l1.values())
                    if e.id in i.source_ids]
            if any(i.tier == LONG_TERM for i in went) and self.mode == "p1":
                why = f"u ≥ τ_fact = {p.tau_fact}"
                ev.append({"type": "t3", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1 → Tier 3 as a durable fact. "
                                  f"{score}, {why}"})
                if any(i.tier == WORKING for i in went):
                    ev.append({"type": "t2", "tier": "t2", "text": e.text,
                               "msg": "very high score, so its gist also goes to Tier 2"})
            elif any(i.tier == LONG_TERM for i in went):
                ev.append({"type": "t3", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1 → kept verbatim in Tier 3. "
                                  f"{score}; the score sets its place in the "
                                  f"forgetting queue"})
            elif any(i.tier == WORKING for i in went):
                g = next(i for i in went if i.tier == WORKING)
                ev.append({"type": "t2", "tier": "t2", "text": g.text,
                           "msg": f"evicted from Tier 1 → compressed into Tier 2. "
                                  f"{score}, between τ_low = {p.tau_low} and "
                                  f"τ_fact = {p.tau_fact}"})
            elif p.routed["long"] > c0["long"] or p.routed["filled"] > c0["filled"]:
                ev.append({"type": "dup", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1; near-duplicate of a fact "
                                  f"already in Tier 3, which is reinforced instead. "
                                  f"{score}"})
            else:
                ev.append({"type": "drop", "tier": None, "text": e.text,
                           "msg": f"evicted from Tier 1 → forgotten. {score}, "
                                  f"below τ_low = {p.tau_low}"})

        # what storage pressure, consolidation and supersession did meanwhile
        retired = {k for k, it in l0.items()
                   if k in live0 and it.superseded_by is not None}
        for k in retired:
            ev.append({"type": "supersede", "tier": "t3", "text": l0[k].text,
                       "msg": "superseded: a newer fact about the same thing "
                              "replaced it" + (", and its space was reclaimed"
                                               if k not in l1 else "")})
        for k, it in l0.items():
            if k not in l1 and k in live0 and k not in retired:
                demoted = [g for g in w1.values() if k in g.source_ids]
                if demoted:
                    ev.append({"type": "demote", "tier": "t2", "text": demoted[0].text,
                               "msg": f"store over budget: lowest-score fact "
                                      f"(u = {it.utility:.2f}) demoted from Tier 3 "
                                      f"to a shorter gist in Tier 2"})
                else:
                    ev.append({"type": "drop", "tier": None, "text": it.text,
                               "msg": f"store over budget (S = {self.store_budget}): "
                                      f"lowest-score fact forgotten, u = "
                                      f"{it.utility:.2f}"})
        for k, it in w1.items():
            if k not in w0 and it.type_label == "merged":
                ev.append({"type": "merge", "tier": "t2", "text": it.text,
                           "msg": f"Tier 2 full: {len(it.source_ids)} older gists "
                                  f"merged into one"})
        promoted = [i for k, i in l1.items() if k not in l0
                    and any(s in w0 for s in i.source_ids)]
        for it in promoted:
            ev.append({"type": "t3", "tier": "t3", "text": it.text,
                       "msg": "Tier 2 full: a high-score gist promoted to Tier 3"})
        for k, it in w0.items():
            if (k not in w1 and it.type_label != "merged"
                    and not any(k in g.source_ids for g in w1.values())
                    and not any(k in f.source_ids for f in l1.values())):
                ev.append({"type": "drop", "tier": None, "text": it.text,
                           "msg": "store over budget: gist dropped from Tier 2"})

        for e in ev:
            e["n"] = turn.index
            e["speaker"] = speaker
        self.log.extend(ev)
        return ev

    # -- the read path
    def context(self, query):
        ctx = self.policy.build_context(query, self.context_budget)
        return ctx, [_item(i) for i in ctx.items]

    def compare(self, query, ctx):
        """The same question asked of each memory, at the same budgets.

        `ctx` is MemGate's context for `query`, already built. Each row lists
        which messages (by number) its context draws on; a MemGate gist
        counts for every message it was made from. `only` marks notes whose
        messages none of the baselines sent.
        """
        def turns(it):
            return self.turn_of.get(it.id) or {it.turn_index}
        rows = {"memgate": (ctx, self.policy.stats()["stored_tokens"])}
        for k, r in self.rivals.items():
            rows[k] = (r.build_context(query, self.context_budget), r.stored_tokens())
        cover = {k: set().union(*[turns(i) for i in c.items]) if c.items else set()
                 for k, (c, _) in rows.items()}
        theirs = cover["window"] | cover["rag"]
        out = {"query": query, "said_tokens": self.said_tokens,
               "store_budget": self.store_budget,
               "context_budget": self.context_budget, "rows": {}}
        for k, (c, stored) in rows.items():
            notes = []
            for i in c.items:
                d = {"text": i.text, "tier": TIER_KEY.get(i.tier, "t1"),
                     "turns": sorted(turns(i))}
                if k == "memgate":
                    d["only"] = not (set(d["turns"]) & theirs)
                notes.append(d)
            out["rows"][k] = {"tokens": c.tokens, "stored_tokens": stored,
                              "messages": len(cover[k]), "notes": notes,
                              "text": c.text}
        self.last_compare = out
        return out

    def why(self, item_id):
        """Explain the score an item in memory got, from the message it was
        made from: a gist is scored on its original, not on the gist, and a
        merged gist is shown by its earliest message."""
        st = self.policy.store
        it = next((i for i in list(st.short) + list(st.working) + list(st.long)
                   if i.id == item_id), None)
        if it is None:
            raise NotFound("that note is no longer in memory")
        made_from = sorted(self.turn_of.get(it.id) or {it.turn_index})
        src = self.raw.get(made_from[0])
        if src is None:
            raise NotFound("the message this note came from is not on record")
        out = explain(self.policy.scorer, *src)
        if out is None:
            raise NotFound(f"{self.policy.scorer.name} cannot be broken down")
        out.update(tier=TIER_KEY[it.tier], made_from=made_from)
        return out


# ----------------------------------------------------------------- bench
class Bench:
    """run_live_check.py in a subprocess, its output kept for the page."""

    def __init__(self):
        self.lines, self.proc, self.code = [], None, None
        self.lock = threading.Lock()

    def start(self, store):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False
            self.lines, self.code = [], None
            self.proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "run_live_check.py"),
                 "--store", str(int(store))],
                cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line and not line.lstrip().startswith(("Warning", "warnings.")):
                self.lines.append(line)
        self.code = self.proc.wait()

    def state(self):
        running = bool(self.proc) and self.proc.poll() is None
        return {"running": running, "lines": self.lines[-40:], "code": self.code}


# ------------------------------------------------------------------ chats
class Chats:
    """Saved conversations, one JSON file each, rebuilt by replay."""

    def __init__(self, folder=CHATS):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _path(self, cid):
        if not cid or not all(c.isalnum() or c == "-" for c in cid):
            raise ValueError(f"bad chat id {cid!r}")
        return os.path.join(self.folder, cid + ".json")

    def list(self):
        out = []
        for f in os.listdir(self.folder):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.folder, f), encoding="utf-8") as fh:
                        d = json.load(fh)
                    row = {k: d.get(k) for k in
                           ("id", "title", "mode", "scale", "updated")}
                    row["messages"] = len(d.get("transcript", []))
                    out.append(row)
                except (OSError, ValueError):
                    continue
        return sorted(out, key=lambda c: c["updated"] or 0, reverse=True)

    def save(self, chat):
        sess = chat["session"]
        d = {"id": chat["id"], "title": chat["title"], "mode": sess.mode,
             "scale": sess.scale, "store_budget": sess.store_budget,
             "created": chat["created"], "updated": time.time(),
             "transcript": sess.transcript}
        tmp = self._path(chat["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self._path(chat["id"]))   # a crash can't truncate it

    def load(self, cid):
        with open(self._path(cid), encoding="utf-8") as fh:
            d = json.load(fh)
        sess = Session(d.get("mode"), d.get("store_budget"), d.get("scale"))
        for m in d.get("transcript", []):
            sess.add(m["speaker"], m["text"])        # same observe() path
        return {"id": d["id"], "title": d.get("title") or "Untitled",
                "created": d.get("created") or time.time(), "session": sess}

    def delete(self, cid):
        try:
            os.remove(self._path(cid))
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------- server
def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                b = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}
            return b if isinstance(b, dict) else {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/inspect":
                with open(INSPECT, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, app.full_state())
            if self.path == "/api/chats":
                return self._send(200, {"chats": app.chats.list(),
                                        "current": app.chat["id"]})
            if self.path == "/api/bench":
                return self._send(200, app.bench.state())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            b = self._body()
            try:
                if self.path == "/api/chat":
                    text = b.get("text")
                    text = text if isinstance(text, str) else ""
                    if not text.strip():
                        return self._send(400, {"error": "the message is empty"})
                    if len(text) > MAX_CHARS:
                        return self._send(413, {"error": (
                            f"that message is {len(text):,} characters; the "
                            f"limit is {MAX_CHARS:,}. Send it in parts.")})
                    return self._send(200, app.say(text))
                if self.path == "/api/sample":
                    return self._send(200, app.sample())
                if self.path in ("/api/reset", "/api/chats/new"):
                    return self._send(200, app.new(b.get("mode"),
                                                   b.get("store_budget"),
                                                   b.get("scale")))
                if self.path == "/api/chats/open":
                    return self._send(200, app.open(str(b.get("id", ""))))
                if self.path == "/api/chats/delete":
                    return self._send(200, app.delete(str(b.get("id", ""))))
                if self.path == "/api/why":
                    return self._send(200, app.why(b.get("id"), b.get("text"),
                                                   b.get("speaker", "user")))
                if self.path == "/api/rival":
                    return self._send(200, app.ask_rival(str(b.get("rival", ""))))
                if self.path == "/api/bench":
                    app.bench.start(b.get("store", 4096))
                    return self._send(200, app.bench.state())
            except NotFound as exc:
                return self._send(404, {"error": str(exc)})
            except Exception as exc:          # show it on the page, keep serving
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            self._send(404, {"error": "not found"})
    return H


class App:
    def __init__(self, model, chats=None):
        self.model = model
        self.chats = chats or Chats()
        self.bench = Bench()
        self.lock = threading.Lock()
        saved = self.chats.list()
        self.chat = None
        if saved:                              # pick up where it was left
            try:
                self.chat = self.chats.load(saved[0]["id"])
            except (OSError, ValueError, KeyError):
                self.chat = None
        if self.chat is None:
            self.new()

    @property
    def session(self):
        return self.chat["session"]

    def full_state(self):
        s = self.session.state()
        s.update(log=self.session.log[-80:], model=self.model.name,
                 model_note=self.model.note,
                 model_enabled=bool(self.model.enabled),
                 scorer=self.session.policy.scorer.name,
                 backends=backend_info(), chat_id=self.chat["id"],
                 title=self.chat["title"], transcript=self.session.transcript)
        return s

    def new(self, mode=None, store_budget=None, scale=None):
        with self.lock:
            if self.chat and not self.chat["session"].transcript:
                self.chats.delete(self.chat["id"])   # don't pile up empties
            self.chat = {"id": uuid.uuid4().hex[:12], "title": "New chat",
                         "created": time.time(),
                         "session": Session(mode or "p1s", store_budget, scale)}
            self.chats.save(self.chat)
        return self.full_state()

    def open(self, cid):
        with self.lock:
            self.chat = self.chats.load(cid)
        return self.full_state()

    def delete(self, cid):
        self.chats.delete(cid)
        if cid == self.chat["id"]:
            rest = self.chats.list()
            return self.open(rest[0]["id"]) if rest else self.new()
        return self.full_state()

    def _title(self, text):
        if self.chat["title"] == "New chat":
            t = " ".join(text.split())
            self.chat["title"] = t[:48] + ("…" if len(t) > 48 else "")

    def sample(self):
        sess = self.session
        with sess.lock:
            for spk, txt in SAMPLE:
                sess.add(spk, txt)
            self._title("Sample chat")
            self.chats.save(self.chat)
        return self.full_state()

    def say(self, text):
        sess = self.session
        with sess.lock:
            events = sess.add("user", text)
            ctx, items = sess.context(text)
            compare = sess.compare(text, ctx)
            try:
                reply = self.model.reply(ctx.text, text)
            except Exception:
                self._title(text)
                self.chats.save(self.chat)   # the message is in memory; keep disk in step
                raise
            reply = reply or "(the model returned an empty reply)"
            events += sess.add("assistant", reply)
            self._title(text)
            self.chats.save(self.chat)
        out = self.full_state()
        out.update(reply=reply, context=items, context_tokens=ctx.tokens,
                   events=events, compare=_public(compare))
        return out

    def ask_rival(self, rival):
        """Answer the last question again from a baseline's memory instead.

        Nothing is stored: the chat and every memory stay as they were, so
        this only shows what the model would have said without MemGate."""
        sess = self.session
        with sess.lock:
            cmp = sess.last_compare
            if not cmp:
                raise NotFound("ask something first")
            if rival not in cmp["rows"] or rival == "memgate":
                raise NotFound(f"no baseline called {rival!r}")
            row = cmp["rows"][rival]
            reply = self.model.reply(row["text"], cmp["query"])
        return {"rival": rival, "query": cmp["query"],
                "reply": reply or "(the model returned an empty reply)"}

    def why(self, item_id=None, text=None, speaker="user"):
        sess = self.session
        with sess.lock:
            if item_id:
                return sess.why(item_id)
            t = " ".join((text or "").split())
            if not t:
                raise NotFound("nothing to score")
            spk = speaker if speaker in ("user", "assistant") else "user"
            return explain(sess.policy.scorer, f"[{spk}] {t}", spk)


def _public(cmp):
    """The comparison without the raw context strings (the notes carry them)."""
    return {**cmp, "rows": {k: {f: v for f, v in r.items() if f != "text"}
                            for k, r in cmp["rows"].items()}}


def main():
    # The private settings setup saved (the Anthropic key), before anything
    # reads the environment. A key exported in the terminal still wins.
    from memgate.envfile import ENV_PATH, load_env
    loaded = load_env()
    ap = argparse.ArgumentParser(description="MemGate app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to let other devices on the network open it")
    ap.add_argument("--llm", choices=BACKENDS,
                    default=os.environ.get("MEMGATE_LLM", "auto"),
                    help="which model answers (default: auto)")
    ap.add_argument("--no-llm", action="store_true",
                    help="run without a chat model (memory still works)")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open the browser on start")
    ap.add_argument("--scorer", choices=("learned", "heuristic"),
                    default=os.environ.get("MEMGATE_SCORER", "learned"),
                    help="what scores each message: the learned scorer trained on "
                         "LoCoMo + REALTALK (default), or the P1 rule scorer")
    args = ap.parse_args()
    # Read by MemGatePolicy when each chat's policy is built. The benchmark
    # check pins its own scorer, so this does not move the committed numbers.
    os.environ["MEMGATE_SCORER"] = args.scorer

    model = ChatModel("none" if args.no_llm else args.llm)
    print("MemGate")
    if loaded:
        print(f"  settings from {ENV_PATH}: {', '.join(loaded)}")
    print("  warming the embedder ...", flush=True)
    from memgate.utils import embed
    embed("warm up")
    print(f"  embedder: {backend_info()['embedder']}", flush=True)
    print(f"  scorer: {MemGatePolicy().scorer.name}", flush=True)
    if model.enabled:
        print(f"  loading {model.name} ...", flush=True)
        try:
            try:
                model.load()
            except Exception as exc:
                if not model.fallback:
                    raise
                model._fall_back(exc)
                model.load()
        except Exception as exc:
            print(f"  [MemGate] WARNING: chat model did not load "
                  f"({type(exc).__name__}: {str(exc)[:120]}); continuing "
                  f"without it. The memory side works unchanged.", flush=True)
            model.disable()
    else:
        print("  no chat model: replies will show what it would have been sent")
    app = App(model)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"\n  model: {model.name}")
    print(f"  chats saved in {CHATS}")
    url = f"http://localhost:{args.port}"
    print(f"\n  open {url}   (Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
