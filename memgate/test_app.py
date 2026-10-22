#!/usr/bin/env python3
"""App tests: whatever a person types, the chat answers or says why not.
Run: python test_app.py"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import app as A
from memgate.reader import ChatModel

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


class Echo(ChatModel):
    """Stands in for the chat model: echoes, or returns nothing on cue."""
    def __init__(self):
        super().__init__("none")

    @property
    def enabled(self):
        return True

    @property
    def name(self):
        return "echo"

    def reply(self, notes, message):
        return "" if message == "silent" else "you said:\n" + message


app = A.App(Echo(), A.Chats(tempfile.mkdtemp()))
srv = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(app))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/api/chat"


def post(raw):
    req = urllib.request.Request(URL, data=raw, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def say(text):
    return post(json.dumps({"text": text}).encode())


print("\n-- any input --")
code = "```python\ndef f(x):\n    return '<b>&amp;</b>'\n```"
inputs = ["?", "word " * 3000, "मेरा नाम सिमर है", "我的考试在十一月八日",
          "🎉🎂 12/03/2003 👨‍👩‍👧", code, "<script>alert(1)</script>",
          "₹40,000.50 due 2026-11-08 14:30 +91 98765 43210",
          "null\x00byte ‮﻿", "Para one.\n\nPara two.",
          "What did I say earlier about my exam?"]
codes = [say(t)[0] for t in inputs]
check("varied inputs all answered", codes == [200] * len(inputs), str(codes))
check("empty and whitespace refused",
      [say(t)[0] for t in ("", "   \n\t")] == [400, 400])
check("over the length limit refused with a reason",
      say("x" * (A.MAX_CHARS + 1)) == (413, {"error": (
          f"that message is {A.MAX_CHARS + 1:,} characters; the limit is "
          f"{A.MAX_CHARS:,}. Send it in parts.")}))
check("malformed bodies refused, not crashed",
      [post(b)[0] for b in (b"not json", b"[1, 2]", b'{"text": null}',
                            b'{"text": 5}', b'\xff\xfe')] == [400] * 5)
check("an empty model reply is shown as such",
      say("silent")[1]["reply"] == "(the model returned an empty reply)")

errs = []
ts = [threading.Thread(target=lambda i=i: errs.append(say(f"rapid {i}")[0]))
      for i in range(12)]
[t.start() for t in ts]
[t.join() for t in ts]
check("rapid messages all land", errs == [200] * 12 and
      len(app.session.transcript) == 2 * (len(inputs) + 1 + 12), str(errs))

print("\n-- saved chats --")
tr = app.session.transcript
check("line breaks kept in the transcript",
      any(m["text"] == code for m in tr) and
      any(m["text"] == "you said:\n" + code for m in tr))
re = app.chats.load(app.chat["id"])["session"]
check("reload replays the same transcript", re.transcript == tr)
check("reload rebuilds the same memory",
      [i["text"] for i in re.state()["t3"]]
      == [i["text"] for i in app.session.state()["t3"]])

print("\n-- pages and endpoints --")
BASE = URL.rsplit("/api/", 1)[0]


def get(path):
    try:
        with urllib.request.urlopen(BASE + path) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def api(path, body=None):
    if body is None:
        code, raw, _ = get(path)
        return code, json.loads(raw)
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


code, raw, ctype = get("/")
check("the chat page is served", code == 200 and b"MemGate" in raw
      and ctype.startswith("text/html"))
code, raw, _ = get("/inspect")
check("the all-tiers page is served", code == 200 and b"<html" in raw.lower())
check("unknown paths are 404",
      get("/nope")[0] == 404 and api("/api/nope", {})[0] == 404)
check("nonsense store budgets refused with a reason",
      [api("/api/chats/new", {"mode": "p1s", "store_budget": v})[0]
       for v in (-5, "abc", 10**12, True, [1])] == [400] * 5
      and "between" in api("/api/chats/new", {"store_budget": -5})[1]["error"])
check("the benchmark refuses a nonsense budget",
      api("/api/bench", {"store": -1})[0] == 400)
code, st = api("/api/state")
check("state carries what the page shows",
      code == 200 and {"t1", "t2", "t3", "log", "transcript", "model", "scorer",
                       "backends", "chat_id", "title", "counts",
                       "stored_tokens", "context_budget"} <= set(st), str(set(st)))
check("the chat title is the first message, cut at 48 characters",
      st["title"] == "?")

print("\n-- chats --")
code, st = api("/api/chats/new", {"mode": "p1a", "scale": "compact"})
first = st["chat_id"]
check("a new chat takes the chosen policy and sizes",
      code == 200 and st["mode"] == "p1a" and st["scale"] == "compact"
      and st["store_budget"] == A.SCALES["compact"][2] and st["messages"] == 0)
code, st2 = api("/api/chats/new", {"mode": "p1s", "scale": "full"})
check("an empty chat is not kept when another is started",
      first not in [c["id"] for c in api("/api/chats")[1]["chats"]])
check("the old size name 'demo' opens as compact",
      api("/api/chats/new", {"scale": "demo"})[1]["scale"] == "compact")
check("unknown policy and size fall back to the defaults",
      (lambda s: (s["mode"], s["scale"]))(
          api("/api/chats/new", {"mode": "zzz", "scale": "huge"})[1])
      == ("p1s", "full"))
check("a custom store budget is used",
      api("/api/chats/new", {"mode": "p1s", "store_budget": 777})[1]
      ["store_budget"] == 777)
long_title = "My capstone final review is on 6 November and the report is late"
say(long_title)
c = api("/api/state")[1]
check("titles longer than 48 characters end with an ellipsis",
      c["title"] == long_title[:48] + "…")
code, st = api("/api/sample", {})
check("the sample chat loads all its messages",
      code == 200 and st["messages"] == 1 + 1 + len(A.SAMPLE)
      and len(st["transcript"]) == st["messages"])
cid = st["chat_id"]
api("/api/chats/new", {"mode": "p1s", "scale": "compact"})
say("second chat")
listed = api("/api/chats")[1]
check("chats are listed newest first with their sizes",
      listed["chats"][0]["id"] == listed["current"]
      and {"id", "title", "mode", "scale", "messages"} <= set(listed["chats"][0]))
code, st = api("/api/chats/open", {"id": cid})
check("opening a saved chat brings back its memory",
      code == 200 and st["chat_id"] == cid and st["messages"] == 1 + 1 + len(A.SAMPLE)
      and len(st["t3"]) > 0)
check("opening a missing chat is an error, not a crash",
      api("/api/chats/open", {"id": "doesnotexist"})[0] == 500
      and api("/api/state")[0] == 200)
check("a chat id cannot reach outside the chats folder",
      api("/api/chats/open", {"id": "../../etc/passwd"})[0] == 500
      and "bad chat id" in api("/api/chats/open", {"id": "../x"})[1]["error"])
before = len(api("/api/chats")[1]["chats"])
code, st = api("/api/chats/delete", {"id": cid})
check("deleting the open chat removes it and opens another",
      code == 200 and st["chat_id"] != cid
      and len(api("/api/chats")[1]["chats"]) == before - 1)
check("deleting a chat that is already gone is harmless",
      api("/api/chats/delete", {"id": cid})[0] == 200)

print("\n-- restart --")
folder = app.chats.folder
cur = app.chat["id"]
app2 = A.App(Echo(), A.Chats(folder))
check("a restart reopens the most recent chat", app2.chat["id"] == cur)
with open(os.path.join(folder, "broken.json"), "w") as fh:
    fh.write("{not json")
check("a corrupt chat file is skipped, not fatal",
      "broken" not in [c["id"] for c in A.Chats(folder).list()]
      and A.App(Echo(), A.Chats(folder)).chat is not None)
empty = tempfile.mkdtemp()
check("a first start with no chats opens a new one",
      A.App(Echo(), A.Chats(empty)).chat["title"] == "New chat")
pa = A.App(Echo(), A.Chats(tempfile.mkdtemp()))
pa.new("p1a"); pa.say("My exam is on 8 November.")
check("a new chat defaults to P1-S", A.App(Echo(), A.Chats(empty)).session.mode == "p1s")
restarted = A.App(Echo(), A.Chats(pa.chats.folder))
check("a restart after a P1-A chat opens a fresh P1-S chat and keeps the old one",
      restarted.session.mode == "p1s" and restarted.chat["id"] != pa.chat["id"]
      and pa.chat["id"] in [c["id"] for c in restarted.chats.list()])

print("\n-- where each message is (the tag beside it) --")
fa = A.App(Echo(), A.Chats(tempfile.mkdtemp()))
fa.new("p1s", 300, "full")
DEMO = ["My capstone final review is on 5 November at 10 am in room 402.",
        "hi", "lol nice", "ok cool", "haha yes",
        "My teammate Yash runs the LongMemEval experiments.",
        "When is my final review?"]
for t in DEMO:
    fa.say(t)
st = fa.full_state()
f = st["fates"]
user = [x for x, m in zip(f, st["transcript"]) if m["speaker"] == "user"]
check("every transcript entry has a fate with a score",
      len(f) == len(st["transcript"]) and all(x["score"] is not None for x in f))
check("the review date goes to Tier 3 at once, with its score",
      user[0]["tier"] == "t3" and user[0]["score"] >= fa.session.policy.tau_fact)
check("hi, lol nice, ok cool and haha yes are forgotten on arrival as small talk",
      all(x["tier"] == "gone" and x["why"] == "small talk" for x in user[1:5]),
      str([(x["tier"], x.get("why")) for x in user[1:5]]))
check("small talk takes no space in memory",
      not any(i["text"].endswith(("] hi", "lol nice", "ok cool", "haha yes"))
              for k in ("t1", "t2", "t3") for i in st[k]))
check("the forgotten count includes small talk", st["counts"]["dropped"] >= 4)
check("a question stays in Tier 1 for context", user[-1]["tier"] == "t1")
w = fa.why(msg=2)
check("why-this-score works for a forgotten message",
      w["label"] == "filler" and abs(w["utility"] - user[1]["score"]) < 0.01)
check("why-this-score works for a fact", fa.why(msg=0)["utility"] >= 0.45)
fa2 = A.App(Echo(), A.Chats(fa.chats.folder))
check("tags are rebuilt the same after a restart", fa2.full_state()["fates"] == f)
for mode in ("p1a", "p1"):
    fm = A.App(Echo(), A.Chats(tempfile.mkdtemp()))
    fm.new(mode, 300, "full")
    for t in DEMO:
        fm.say(t)
    fu = [x for x, m in zip(fm.full_state()["fates"], fm.session.transcript)
          if m["speaker"] == "user"]
    check(f"{mode}: small talk forgotten, the review kept",
          all(x["tier"] == "gone" for x in fu[1:5]) and fu[0]["tier"] != "gone",
          str([x["tier"] for x in fu]))
off = A.Session("p1s", 300, "full", gate=False)
for t in DEMO:
    off.add("user", t)
check("with the gate off, small talk sits in Tier 1 like the experiments",
      [x["tier"] for x in off.fates()][-3:] == ["t1", "t1", "t1"]
      and not off.gated)
for name in os.listdir(folder):
    if name.endswith(".tmp"):
        check("saves leave no temp files behind", False, name)
        break
else:
    check("saves leave no temp files behind", True)

srv.shutdown()

print("\n-- public link: own chats per visitor, passcode --")
pub = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(
    A.Visitors(Echo(), tempfile.mkdtemp()), passcode="jury2026"))
threading.Thread(target=pub.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{pub.server_address[1]}"
ANNA, BEN = "a" * 32, "b" * 32
PASS = None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def call(path, body=None, visitor=None, cookie=None, form=None):
    hdr = {}
    if visitor:
        hdr["X-MemGate-Visitor"] = visitor
    if cookie:
        hdr["Cookie"] = cookie
    data = None
    if form is not None:
        data = form.encode()
        hdr["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=hdr,
                                 method="POST" if data is not None else "GET")
    try:
        r = urllib.request.build_opener(NoRedirect).open(req)
    except urllib.error.HTTPError as e:
        r = e
    raw = r.read()
    try:
        out = json.loads(raw)
    except ValueError:
        out = raw.decode("utf-8", "replace")
    return r.status, out, r.headers


code, page, _ = call("/")
check("without the passcode the page asks for it",
      code == 200 and "passcode" in page)
check("without the passcode the API refuses",
      call("/api/state", visitor=ANNA)[0] == 401)
check("the health check needs no passcode", call("/healthz")[0] == 200)
check("a wrong passcode is refused",
      call("/login", form="passcode=nope")[0] == 401)
code, _, h = call("/login", form="passcode=jury2026")
PASS = (h.get("Set-Cookie") or "").split(";")[0]
check("the right passcode sets a cookie and opens the app",
      code == 303 and PASS.startswith("mg_pass="))
check("a forged cookie is refused",
      call("/api/state", visitor=ANNA, cookie="mg_pass=x")[0] == 401)

call("/api/chat", {"text": "my name is Anna"}, ANNA, PASS)
code, st, _ = call("/api/state", visitor=BEN, cookie=PASS)
check("a second visitor starts with an empty chat",
      code == 200 and st["transcript"] == [])
code, lst, _ = call("/api/chats", visitor=BEN, cookie=PASS)
check("a second visitor does not see the first one's chats",
      [c["messages"] for c in lst["chats"]] == [0])
code, st, _ = call("/api/state", visitor=ANNA, cookie=PASS)
check("the first visitor still has their chat",
      [m["text"] for m in st["transcript"]][0] == "my name is Anna")
code, _, h = call("/api/state", cookie=PASS)
check("a browser with no id is given one in a cookie",
      code == 200 and "mg_visitor=" in (h.get("Set-Cookie") or ""))
check("a malformed visitor id is replaced, not used as a path",
      "mg_visitor=" in (call("/api/state", visitor="../../etc",
                             cookie=PASS)[2].get("Set-Cookie") or ""))
pub.shutdown()

print("\n-- many requests at once --")
# The standard library's accept queue is 5 deep; 200 visitors at once had
# 314 of 2,000 requests reset before a thread picked them up (scale_app.py).
burst = A.Server(("127.0.0.1", 0), A.make_handler(A.App(Echo(), A.Chats(tempfile.mkdtemp()))))
threading.Thread(target=burst.serve_forever, daemon=True).start()
codes = []
def _one(k):
    req = urllib.request.Request(
        f"http://127.0.0.1:{burst.server_address[1]}/api/chat",
        data=json.dumps({"text": f"burst note {k}"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            codes.append(r.status)
    except Exception as e:
        codes.append(type(e).__name__)
_ths = [threading.Thread(target=_one, args=(k,)) for k in range(60)]
[t.start() for t in _ths]
[t.join() for t in _ths]
check("the app server queues a burst of 60 requests instead of resetting them",
      A.Server.request_queue_size >= 128 and codes.count(200) == 60,
      str([c for c in codes if c != 200][:3]))
burst.shutdown()

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
