#!/usr/bin/env python3
"""MemGate app: chat through MemGate and watch what each tier holds.

    python3 app.py                 # then open http://localhost:8765
    python3 app.py --llm claude    # answer with Claude (ANTHROPIC_API_KEY)
    python3 app.py --llm hf        # transformers, any OS
    python3 app.py --no-llm        # no model: replies show the context only
    python3 app.py --scorer heuristic   # the P1 rule scorer instead of the learned one
    python3 app.py --port 9000
    python3 app.py --public --passcode XYZ   # a public link: own chats per visitor

Online (a Hugging Face Space) it is configured by environment instead:
HOST, PORT, MEMGATE_PUBLIC=1, MEMGATE_PASSCODE. See deploy/huggingface/.

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
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memgate.types import Turn, SHORT_TERM, WORKING, LONG_TERM
from memgate.policies import (MemGatePolicy, MemGateAdaptivePolicy,
                              MemGateSelectPolicy, SlidingWindowPolicy,
                              RAGPolicy)
from memgate.explain import explain
from memgate.reader import ChatModel, BACKENDS
from memgate.utils import backend_info, count_tokens, embed_many
from memgate.ingest import (GuardedScorer, ExtractError, chunk, clean, focus,
                            extract, decode_upload, one_line, MAX_DOC_CHARS,
                            MAX_UPLOAD_BYTES)

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "chat.html")          # the chat app
INSPECT = os.path.join(HERE, "web", "index.html")    # all tiers at once
CHATS = os.path.join(HERE, "chats")
# Longest message accepted, about 50,000 tokens. A long message is split into
# turn-sized pieces before it reaches memory (memgate/ingest.py), so its size
# only sets how long ingesting takes. The model is sent its questions, not
# the whole paste: see LONG_MESSAGE.
MAX_CHARS = 200_000
# Largest request body read: an upload at its 15 MB limit, base64-encoded,
# with room for the rest of the JSON. Without a cap a single request could
# make the server read any amount into memory before refusing it.
MAX_BODY = MAX_UPLOAD_BYTES * 4 // 3 + (1 << 20)
# Past this many tokens a message is stored in memory piece by piece, and the
# model gets the notes MemGate recalls for it plus the message's own
# questions (or its opening and closing lines) instead of all of it.
LONG_MESSAGE = 1500
STORE_RANGE = (40, 1_000_000)   # tokens; the page's Store S box starts at 40
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


class BadRequest(ValueError):
    """Input the page or a caller got wrong: answered 400, not 500."""


def store_budget_from(value, default):
    """A store budget as sent by the page: a whole number of tokens in
    STORE_RANGE, or the default when none was given."""
    if value in (None, "", 0):
        return default
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest(f"the store budget must be a number of tokens, "
                         f"not {value!r}") from None
    lo, hi = STORE_RANGE
    if isinstance(value, bool) or not lo <= n <= hi:
        raise BadRequest(f"the store budget must be between {lo:,} and "
                         f"{hi:,} tokens")
    return n


class Session:
    def __init__(self, mode="p1s", store_budget=None, scale="full"):
        self.mode = mode if mode in MODES else "p1s"
        scale = SCALE_ALIASES.get(scale, scale)
        self.scale = scale if scale in SCALES else "full"
        self.short_capacity, self.context_budget, s_default = SCALES[self.scale]
        self.store_budget = store_budget_from(store_budget, s_default)
        kw = dict(short_capacity=self.short_capacity, budget=self.context_budget)
        if self.mode == "p1":
            self.policy = MemGatePolicy(**kw)
        elif self.mode == "p1s":
            self.policy = MemGateSelectPolicy(store_budget=self.store_budget, **kw)
        else:
            self.policy = MemGateAdaptivePolicy(store_budget=self.store_budget, **kw)
        # Every chat sees open-ended input the scorers were not trained on;
        # GuardedScorer covers those cases and leaves ordinary English
        # messages to the scorer unchanged (memgate/ingest.py).
        self.policy.scorer = GuardedScorer(self.policy.scorer)
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
        self.n = 0             # memory turns observed (pieces of messages)
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
            "messages": len(self.transcript),
            "pieces": self.n,
            "said_tokens": self.said_tokens,
        }

    # -- the write path, with every decision it made written to the log
    def add(self, speaker, text, file=None):
        """Put one message (or one uploaded file's text) into memory.

        The transcript keeps it as written (line breaks in code and pasted
        paragraphs survive a reload). Memory gets it on one line, the form
        every experiment scored, and a long one in turn-sized pieces
        (ingest.chunk), each observed as a turn of its own. Pieces of a file
        are tagged with its name, so a recalled piece says where it came from.
        """
        shown = clean(text)
        pieces = chunk(shown)
        if not pieces:
            return []
        entry = {"speaker": speaker, "text": shown}
        tag = speaker
        if file:
            name = one_line("".join(c for c in str(file.get("name") or "file")
                                    if c not in "[]"))[:32] or "file"
            entry["file"] = dict(file, name=name)
            tag = f"file {name}"
        self.transcript.append(entry)
        if len(pieces) > 1:          # one batched embedder pass, not one per piece
            embed_many([f"[{tag}] {x}" for x in pieces])
        ev = []
        for x in pieces:
            ev += self._observe(tag, x)
        return ev

    def _observe(self, speaker, text):
        p, st = self.policy, self.policy.store
        s0, w0, l0 = self._held()
        live0 = {i.id for i in st.long if i.is_active}
        c0 = dict(p.routed, dropped=st.dropped, demoted=st.demoted,
                  merges=st.merges, reclaimed=st.reclaimed)

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
        # A long message would be embedded by its first 256 word pieces only;
        # retrieve with its questions instead (ingest.focus).
        ctx = self.policy.build_context(focus(query), self.context_budget)
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
            rows[k] = (r.build_context(focus(query), self.context_budget),
                       r.stored_tokens())
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


def _shown(m):
    """A transcript entry as the page gets it: an uploaded file is sent as
    its name, size and opening, not its whole text, on every refresh."""
    if "file" not in m:
        return m
    t = m["text"]
    return {"speaker": m["speaker"], "file": m["file"],
            "text": t[:400] + ("…" if len(t) > 400 else "")}


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
            sess.add(m["speaker"], m["text"], m.get("file"))   # same observe() path
        return {"id": d["id"], "title": d.get("title") or "Untitled",
                "created": d.get("created") or time.time(), "session": sess}

    def delete(self, cid):
        try:
            os.remove(self._path(cid))
        except FileNotFoundError:
            pass


class Visitors:
    """One App per visitor, for a public link (`--public`).

    Each browser gets a random id in a cookie and its own chats folder, so two
    people on the same link never see or overwrite each other's chats. The
    chat model and the benchmark runner are shared.
    """
    KEEP = 200                 # visitors held in memory; the rest reload from disk

    def __init__(self, model, folder=CHATS):
        self.model, self.folder, self.bench = model, folder, Bench()
        self.apps = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def valid(vid):
        return bool(vid) and len(vid) == 32 and all(
            c in "0123456789abcdef" for c in vid)

    def get(self, vid):
        with self.lock:
            app = self.apps.pop(vid, None)
            if app is None:
                app = App(self.model, Chats(os.path.join(self.folder, vid)),
                          bench=self.bench)
            self.apps[vid] = app
            while len(self.apps) > self.KEEP:
                self.apps.popitem(last=False)
            return app


LOGIN = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MemGate</title><style>
body{font-family:system-ui,sans-serif;background:#f6f7f9;color:#1d2430;margin:0;
display:grid;place-items:center;min-height:100vh}
form{background:#fff;padding:28px;border-radius:14px;box-shadow:0 2px 12px #0001;
width:min(320px,calc(100vw - 32px));box-sizing:border-box}
h1{font-size:20px;margin:0 0 6px}p{margin:0 0 16px;color:#5b6573;font-size:14px}
input,button{width:100%;box-sizing:border-box;font:inherit;padding:10px;
border-radius:8px;border:1px solid #cdd3dc}
button{margin-top:10px;background:#2f5bea;color:#fff;border:0;cursor:pointer}
.err{color:#c0392b;font-size:14px;margin-top:10px}
@media (prefers-color-scheme:dark){body{background:#14171c;color:#e6e9ee}
form{background:#1d2128}input{background:#14171c;color:inherit;border-color:#39404b}
p{color:#9aa4b2}}
</style></head><body><form method="post" action="/login">
<h1>MemGate</h1><p>Enter the passcode to open the app.</p>
<input type="password" name="passcode" autofocus autocomplete="current-password">
<button>Open</button>{error}</form></body></html>"""


# ---------------------------------------------------------------- server
def make_handler(app, passcode=None):
    """`app` is an App, or Visitors for a public link. With a passcode, every
    page and API call needs it first (it is kept in a cookie)."""
    public = isinstance(app, Visitors)
    pass_tag = (hmac.new(passcode.encode(), b"memgate", "sha256").hexdigest()
                if passcode else None)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cookie(self, name):
            c = SimpleCookie()
            try:
                c.load(self.headers.get("Cookie") or "")
            except Exception:
                return None
            return c[name].value if name in c else None

        def _app(self):
            """The App this request talks to; for a public link, this
            browser's own, with a new visitor id when it has none."""
            if not public:
                return app
            vid = (self.headers.get("X-MemGate-Visitor")
                   or self._cookie("mg_visitor"))
            if not Visitors.valid(vid):
                vid = secrets.token_hex(16)
                self._set_cookie = f"mg_visitor={vid}"
            return app.get(vid)

        def _allowed(self):
            if not pass_tag:
                return True
            got = self._cookie("mg_pass") or ""
            return hmac.compare_digest(got, pass_tag)

        def _send(self, code, body, ctype="application/json", headers=()):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            cookie = getattr(self, "_set_cookie", None)
            if cookie:
                # Behind https (a Hugging Face Space shows the app in an
                # iframe on another site) the cookie has to be cross-site.
                https = self.headers.get("X-Forwarded-Proto") == "https"
                self.send_header("Set-Cookie", f"{cookie}; Path=/; Max-Age=31536000; "
                                 "HttpOnly; " + ("SameSite=None; Secure" if https
                                                 else "SameSite=Lax"))
                self._set_cookie = None
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _login(self, error=""):
            page = LOGIN.replace(
                "{error}", f'<div class="err">{error}</div>' if error else "")
            return self._send(401 if error else 200, page.encode(),
                              "text/html; charset=utf-8")

        def _locked(self):
            """Answer a request made without the passcode."""
            if self.path.startswith("/api/"):
                return self._send(401, {"error": "enter the passcode first: "
                                        "reload the page"})
            return self._login()

        def _length(self):
            try:
                return max(0, int(self.headers.get("Content-Length") or 0))
            except ValueError:
                return 0

        def _body(self):
            try:
                b = json.loads(self.rfile.read(self._length()) or b"{}")
            except (ValueError, RecursionError):   # not JSON, or nested past
                return {}                          # the parser's depth limit
            return b if isinstance(b, dict) else {}

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"ok": True})
            if not self._allowed():
                return self._locked()
            app = self._app()
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
            if self.path == "/login" and pass_tag:
                from urllib.parse import parse_qs
                n = min(int(self.headers.get("Content-Length") or 0), 4096)
                got = parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
                if not hmac.compare_digest(
                        (got.get("passcode") or [""])[0].strip(), passcode):
                    return self._login("That passcode is not right.")
                self._set_cookie = f"mg_pass={pass_tag}"
                return self._send(303, b"", "text/plain", [("Location", "/")])
            if not self._allowed():
                return self._locked()
            if self._length() > MAX_BODY:
                self.close_connection = True       # the body is left unread
                return self._send(413, {"error": (
                    f"that request is {self._length() / 1e6:.0f} MB; files "
                    f"can be up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")})
            app = self._app()
            b = self._body()
            try:
                if self.path == "/api/chat":
                    text = b.get("text")
                    text = text if isinstance(text, str) else ""
                    if not clean(text):          # nothing but space or invisible marks
                        return self._send(400, {"error": "the message is empty"})
                    if len(text) > MAX_CHARS:
                        return self._send(413, {"error": (
                            f"that message is {len(text):,} characters; the "
                            f"limit is {MAX_CHARS:,}. Send it in parts.")})
                    return self._send(200, app.say(text))
                if self.path == "/api/upload":
                    try:
                        return self._send(200, app.upload(
                            str(b.get("name") or "file"), b.get("data")))
                    except ExtractError as exc:
                        return self._send(422, {"error": str(exc)})
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
                    app.bench.start(store_budget_from(b.get("store"), 4096))
                    return self._send(200, app.bench.state())
            except NotFound as exc:
                return self._send(404, {"error": str(exc)})
            except BadRequest as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:          # show it on the page, keep serving
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            self._send(404, {"error": "not found"})
    return H


class App:
    def __init__(self, model, chats=None, bench=None):
        self.model = model
        self.chats = chats or Chats()
        self.bench = bench or Bench()
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
                 title=self.chat["title"], transcript=[
                     _shown(m) for m in self.session.transcript])
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
            t = one_line(clean(text))
            self.chat["title"] = t[:48] + ("…" if len(t) > 48 else "")

    def sample(self):
        sess = self.session
        with sess.lock:
            for spk, txt in SAMPLE:
                sess.add(spk, txt)
            self._title("Sample chat")
            self.chats.save(self.chat)
        return self.full_state()

    def upload(self, name, data):
        """A file's text into memory, piece by piece. No model call: the
        person asks about it next, and recall finds the pieces."""
        name = one_line(clean(name))[:120] or "file"
        text = extract(name, decode_upload(data))
        if len(text) > MAX_DOC_CHARS:
            raise ExtractError(
                f"{name} has {len(text):,} characters of text; the limit is "
                f"{MAX_DOC_CHARS:,} per file. Split it and upload the parts.")
        sess = self.session
        with sess.lock:
            n0 = sess.n
            info = {"name": name, "chars": len(text),
                    "words": len(text.split()), "tokens": count_tokens(text)}
            events = sess.add("user", text, info)
            pieces = sess.n - n0
            self._title(name)
            self.chats.save(self.chat)
        out = self.full_state()
        tag = f"[file {sess.transcript[-1]['file']['name']}]"
        held = [i for k in ("t1", "t2", "t3") for i in out[k]
                if i["text"].startswith(tag)]
        mine = [e for e in events if e["text"].startswith(tag)]
        # Where the pieces that are not held went: near-duplicates of a piece
        # already stored reinforce it instead (store dedupe), and the rest
        # were forgotten for space or scored as filler.
        out["upload"] = dict(info, name=sess.transcript[-1]["file"]["name"],
                             pieces=pieces, held=len(held),
                             held_tokens=sum(i["tokens"] for i in held),
                             duplicates=sum(e["type"] == "dup" for e in mine),
                             forgotten=sum(e["type"] == "drop" for e in mine))
        return out

    def say(self, text):
        sess = self.session
        with sess.lock:
            n0 = sess.n
            events = sess.add("user", text)
            pieces = sess.n - n0
            ctx, items = sess.context(text)
            message = text
            if count_tokens(text) > LONG_MESSAGE:
                message = (f"[The user sent a long text, {len(text.split()):,} "
                           f"words. MemGate stored it in memory in pieces; the "
                           f"notes hold the pieces relevant to this message. "
                           f"Its questions, or its opening and closing lines:]"
                           f"\n\n{focus(text)}")
            compare = sess.compare(text, ctx)
            compare["message"] = message   # what ask_rival sends the model
            try:
                reply = self.model.reply(ctx.text, message)
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
                   events=events, message_pieces=pieces,
                   compare=_public(compare))
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
            reply = self.model.reply(row["text"], cmp.get("message", cmp["query"]))
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
    """The comparison without the raw context strings (the notes carry them),
    and with a long pasted question cut to its opening."""
    q = cmp["query"]
    return {**{k: v for k, v in cmp.items() if k != "message"},
            "query": q[:400] + ("…" if len(q) > 400 else ""),
            "rows": {k: {f: v for f, v in r.items() if f != "text"}
                     for k, r in cmp["rows"].items()}}


def _on(value):
    return (value or "").lower() in ("1", "true", "yes")


def main():
    # The private settings setup saved (the Anthropic key), before anything
    # reads the environment. A key exported in the terminal still wins.
    from memgate.envfile import ENV_PATH, load_env
    loaded = load_env()
    ap = argparse.ArgumentParser(description="MemGate app")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT") or 8765))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="0.0.0.0 to let other devices on the network open it")
    ap.add_argument("--public", action="store_true",
                    default=_on(os.environ.get("MEMGATE_PUBLIC")),
                    help="for a public link: each visitor gets their own chats")
    ap.add_argument("--passcode", default=os.environ.get("MEMGATE_PASSCODE"),
                    help="ask for this passcode before opening the app")
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
    if args.public:
        app = Visitors(model)
        print("  public: each visitor gets their own chats")
    else:
        app = App(model)
    if args.passcode:
        print("  passcode required to open the app")
    srv = ThreadingHTTPServer((args.host, args.port),
                              make_handler(app, args.passcode or None))
    print(f"\n  model: {model.name}")
    print(f"  chats saved in {CHATS}")
    if args.public:
        args.no_browser = True
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
