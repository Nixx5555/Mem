#!/usr/bin/env python3
"""App tests: whatever a person types, the chat answers or says why not.
Run: python test_app.py"""
import json
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

srv.shutdown()
print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
