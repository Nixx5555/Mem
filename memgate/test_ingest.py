#!/usr/bin/env python3
"""Any-input tests: cleaning, chunking, files, the scorer guards, and the app
taking long pastes and uploads. Run: python test_ingest.py"""
import base64
import io
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

os.environ["MEMGATE_SCORER"] = "learned"

from memgate import ingest as I
from memgate.utils import count_tokens

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


PROSE = ("MemGate decides what a chat assistant keeps. It scores every message "
         "when it leaves the short-term buffer. High scores stay verbatim in "
         "long-term memory, and low scores are forgotten first when the store "
         "is full. The context the model sees is assembled from what is left. ")

print("\n-- clean --")
check("control characters, zero-width and bidi marks removed",
      I.clean("a\x00b​c‮d﻿e\x07") == "abcde")
check("line breaks and tabs kept, \\r\\n folded",
      I.clean("  one\r\ntwo\rthree\tfour  ") == "one\ntwo\nthree\tfour")
fam = "\U0001F468‍\U0001F469‍\U0001F467"
check("emoji joined by ZWJ survive", I.clean(fam) == fam)
check("decomposed accents are normalised (NFC)",
      I.clean("café") == "café")
check("non-strings are empty", I.clean(None) == "" and I.clean(5) == "")

print("\n-- chunk --")
check("a normal message is one piece, exactly as memory always stored it",
      I.chunk("My flight is on 14 December.\n\nAt 6:40 am.")
      == ["My flight is on 14 December. At 6:40 am."])
check("a long chat message (up to WHOLE tokens) is not split",
      len(I.chunk(" ".join(["word"] * (I.WHOLE - 5)))) == 1)
check("empty and invisible input give no pieces",
      I.chunk("") == [] and I.chunk(" \n\t​ ") == [])
long = PROSE * 40
pc = I.chunk(long)
check("long prose: every piece within MAX_CHUNK tokens",
      len(pc) > 10 and max(count_tokens(p) for p in pc) <= I.MAX_CHUNK,
      str(max(count_tokens(p) for p in pc)))
check("long prose: no word lost, none reordered",
      " ".join(pc).split() == long.split())
check("pieces end at sentence ends where they can",
      sum(p.endswith(".") for p in pc) == len(pc))
for name, txt in (("one huge token", "x" * 5000),
                  ("CJK with no spaces", "我的考试在十一月八日。" * 300),
                  ("Hindi", "मेरी परीक्षा आठ नवंबर को है। " * 200),
                  ("code", "def f(x):\n    return x * 2\n" * 200),
                  ("a CSV", "name,age,city\n" + "Simar,22,Indore\n" * 400),
                  ("one line, no punctuation", "word " * 4000),
                  ("emoji", "🎉" * 3000)):
    pc = I.chunk(txt)
    check(f"{name}: split, all pieces within MAX_CHUNK tokens, none empty",
          len(pc) > 1 and all(p and count_tokens(p) <= I.MAX_CHUNK for p in pc),
          f"{len(pc)} pieces, max {max(map(count_tokens, pc)) if pc else 0}")
check("CJK text survives chunking whole",
      "".join(I.chunk("我的考试在十一月八日。" * 300)).replace(" ", "")
      == "我的考试在十一月八日。" * 300)

print("\n-- focus (what a long message retrieves with) --")
check("a short message is its own query", I.focus("When is my exam?") == "When is my exam?")
q = I.focus(PROSE * 30 + "\n\nSo when is the final review?")
check("a long message retrieves with its closing question",
      q == "So when is the final review?", q)
q = I.focus("Here are my notes.\n\n" + PROSE * 30 + "\n\nSummarise them.")
check("with no question: its opening and closing pieces",
      q.startswith("Here are my notes.") and q.endswith("Summarise them.")
      and count_tokens(q) <= I.WHOLE, q[:80])

print("\n-- reading files --")


def docx(paras):
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paras)
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml",
                   f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')
    return b.getvalue()


def pptx(slides):
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for i, lines in enumerate(slides, 1):
            ps = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in lines)
            z.writestr(f"ppt/slides/slide{i}.xml",
                       f'<p:sld xmlns:p="x" xmlns:a="{A}"><a:txBody>{ps}</a:txBody></p:sld>')
    return b.getvalue()


def pdf(lines):
    """A one-page PDF with a real text layer, built by hand (no writer lib)."""
    ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(
        "(" + t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") '"
        for t in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream"]
    out, offs = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n"
    return out.encode("latin-1")


def refused(name, data, why):
    try:
        I.extract(name, data)
        return False
    except I.ExtractError as e:
        return why in str(e)


check("UTF-8 text", I.extract("a.txt", "naïve café ✓".encode()) == "naïve café ✓")
check("UTF-8 with a BOM", I.extract("a.md", b"\xef\xbb\xbf# Title") == "# Title")
check("UTF-16 (Windows Notepad)",
      I.extract("a.txt", "मेरा नाम".encode("utf-16")) == "मेरा नाम")
check("Windows-1252", I.extract("a.txt", "price: 40€ – ok".encode("cp1252")) == "price: 40€ – ok")
h = I.extract("page.html", b"<html><head><style>p{}</style><script>alert(1)</script>"
              b"</head><body><h1>Title</h1><p>First &amp; second.</p></body></html>")
check("HTML: visible text only, entities decoded",
      h == "Title\n\nFirst & second.", repr(h))
check("Word (.docx), without python-docx",
      I.extract("r.docx", docx(["Report due 4 November.", "Review on 8 November."]))
      == "Report due 4 November.\n\nReview on 8 November.")
check("PowerPoint (.pptx), slide by slide",
      I.extract("d.pptx", pptx([["MemGate"], ["Results", "P1-S 37.5%"]]))
      == "Slide 1\nMemGate\n\nSlide 2\nResults\nP1-S 37.5%")
nb = json.dumps({"cells": [{"source": ["import numpy as np\n", "x = 1"]},
                           {"source": "# Notes"}]}).encode()
check("Jupyter notebook: cell sources",
      I.extract("n.ipynb", nb) == "import numpy as np\nx = 1\n\n# Notes")
check("code keeps its line breaks", I.extract("f.py", b"def f():\r\n    return 1\r\n")
      == "def f():\n    return 1")
try:
    import pypdf  # noqa: F401
    got = I.extract("paper.pdf", pdf(["The deposit is Rs 50,000.", "Due 1 March 2027."]))
    check("PDF text layer", "deposit is Rs 50,000" in got and "1 March 2027" in got, repr(got))
except ImportError:
    print("  SKIP  PDF (pypdf not installed)")
    check("PDF without pypdf says how to fix it",
          refused("a.pdf", pdf(["x"]), "pip install pypdf"))
check("binary refused", refused("blob.bin", bytes(range(256)) * 50, "binary"))
check("images refused with a reason", refused("photo.png", b"\x89PNG....", "not a text file"))
check("old Office formats refused with what to do",
      refused("old.doc", b"\xd0\xcf\x11\xe0", "save it as .docx"))
check("a damaged .docx refused, not a crash", refused("bad.docx", b"not a zip", "damaged"))
check("an empty file refused", refused("e.txt", b"  \n ", "no text"))
check("over the size limit refused",
      refused("big.txt", b"a" * (I.MAX_UPLOAD_BYTES + 1), "limit"))
for bad in ("", None, "@@not base64@@"):
    try:
        I.decode_upload(bad)
        ok = False
    except I.ExtractError:
        ok = True
    check(f"bad upload data {bad!r} refused", ok)
check("a data: URL upload decodes",
      I.decode_upload("data:text/plain;base64," + base64.b64encode(b"hi").decode()) == b"hi")

print("\n-- the scorer on input it was not trained on --")
from memgate.pretrained import default_scorer
from memgate.scoring import HeuristicScorer
L = default_scorer()
check("the learned scorer loaded (MiniLM present)", L.name == "P3-learned", L.name)
FACT = ["My flight to Delhi is on 14 December at 6:40 am.",
        "My budget for the laptop is 85,000 rupees.",
        "Actually the review moved to 8 November.",
        "My sister Priya lives in Pune and works at Infosys.",
        "I am allergic to peanuts."]
TALK = ["Hi!", "hey", "ok thanks", "lol", "haha", "hmm", "🎉🎉🎉", "👍", "...",
        "thx", "Good morning!", "sure"]
OTHER = ["मेरी फ्लाइट 14 दिसंबर को सुबह 6:40 बजे है।", "मेरा नाम सिमर है और मैं इंदौर में रहता हूँ",
         "我的考试在十一月八日上午九点。", "Моя сестра живёт в Пуне.",
         "ਮੇਰਾ ਨਾਮ ਸਿਮਰ ਹੈ ਅਤੇ ਮੈਂ ਇੰਦੌਰ ਤੋਂ ਹਾਂ"]
CODE = ["def add(a, b):\n    return a + b", "const total = items.reduce((a, b) => a + b, 0);",
        "SELECT name FROM users WHERE id = 5;", "import numpy as np"]
for nm, inner in (("learned", L), ("heuristic", HeuristicScorer())):
    G = I.GuardedScorer(inner)
    u = lambda t: G.score(t)[0]
    check(f"[{nm}] every fact outscores every greeting, emoji and 'lol'",
          min(map(u, FACT)) > max(map(u, TALK)),
          f"min fact {min(map(u, FACT)):.2f}, max talk {max(map(u, TALK)):.2f}")
    check(f"[{nm}] 'Hi!' is small talk, not a fact (BEAM model put it at 0.68)",
          G.score("Hi!") == (min(0.05, inner.score("Hi!")[0]), "filler"))
    check(f"[{nm}] other languages are kept at a neutral score, not dropped as small talk",
          all(u(t) == I.NEUTRAL for t in OTHER) and min(map(u, OTHER)) > max(map(u, TALK)),
          str([round(u(t), 2) for t in OTHER]))
    check(f"[{nm}] code scores at least neutral, so P1 keeps it verbatim",
          all(u(t) >= I.NEUTRAL and G.score(t)[1] == "code" for t in CODE),
          str([round(u(t), 2) for t in CODE]))
    check(f"[{nm}] the same words over and over score low",
          u("The quick brown fox. " * 4) <= 0.15)
    check(f"[{nm}] an ordinary English message is scored by the scorer itself",
          all(G.score(t) == inner.score(t) and G.guard(t) is None for t in FACT))
    ODD = ["", " ", "?", "\x00", "‮", "x" * 20000, "word " * 3000, "<script>alert(1)</script>",
           "₹40,000.50 on 2026-11-08 at 14:30"] + TALK + OTHER + CODE
    check(f"[{nm}] any input gives a utility in [0, 1] and a label",
          all(0.0 <= G.score(t, s)[0] <= 1.0 and G.score(t, s)[1]
              for t in ODD for s in ("user", "assistant", "file notes.pdf")))

from memgate.data import DEFAULT_LOCOMO_PATH, load_locomo
if os.path.exists(DEFAULT_LOCOMO_PATH):
    G = I.GuardedScorer(L)
    turns = [t for _s, ts, _q in load_locomo() for t in ts]
    fired = [t.text for t in turns if G.guard(f"[{t.speaker}] {t.text}")]
    check("on LoCoMo the guards touch under 1% of turns (the scorer's own ground)",
          len(fired) < 0.01 * len(turns), f"{len(fired)} of {len(turns)}")
else:
    print("  SKIP  LoCoMo guard share (data/locomo/locomo10.json not fetched)")

print("\n-- the evicted turn is scored as its own speaker --")
from memgate.policies import MemGatePolicy
from memgate.scoring import Scorer
from memgate.types import Turn


class Spy(Scorer):
    name = "spy"

    def __init__(self):
        self.seen = []

    def score(self, text, speaker="user"):
        self.seen.append((text.split("]")[0][1:], speaker))
        return 0.5, "context"


spy = Spy()
p = MemGatePolicy(scorer=spy, short_capacity=2)
for i, spk in enumerate(["user", "user", "user", "assistant", "user", "assistant", "user"]):
    p.observe(Turn(session=1, index=i, speaker=spk, text=f"message number {i} here"))
check("speaker passed to the scorer is the evicted item's, not the newest turn's",
      len(spy.seen) == 5 and all(a == b for a, b in spy.seen), str(spy.seen))

print("\n-- the app: long pastes and uploads --")
import app as A
from memgate.reader import ChatModel


class Echo(ChatModel):
    def __init__(self):
        super().__init__("none")
        self.last = None

    @property
    def enabled(self):
        return True

    @property
    def name(self):
        return "echo"

    def reply(self, notes, message):
        self.last = (notes, message)
        return "ok"


echo = Echo()
app = A.App(echo, A.Chats(tempfile.mkdtemp()))
srv = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(app))
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"


def api(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


b64 = lambda b: base64.b64encode(b).decode()
check("the session's scorer is guarded",
      isinstance(app.session.policy.scorer, I.GuardedScorer))

FILLER = [PROSE.replace("MemGate", f"Section {i}: MemGate") for i in range(14)]
FACT_DOC = "The security deposit for the Indore flat is Rs 48,500, refundable on 30 June."
paras = FILLER[:7] + [FACT_DOC] + FILLER[7:]
code, st = api("/api/upload", {"name": "lease.docx", "data": b64(docx(paras))})
u = st.get("upload", {})
check("a .docx upload goes into memory in pieces",
      code == 200 and u.get("pieces", 0) > 5 and u["name"] == "lease.docx", str(code))
check("the upload reports how much of it memory holds",
      0 < u.get("held", 0) <= u.get("pieces", 0) and u.get("held_tokens", 0) > 0)
check("every piece is accounted for: held, merged as a duplicate, or forgotten",
      u["held"] + u["duplicates"] + u["forgotten"] == u["pieces"], str(u))
check("pieces are tagged with the file name",
      any(i["text"].startswith("[file lease.docx] ") for k in ("t1", "t3") for i in st[k]))
check("the transcript entry shows the file, not its whole text",
      st["transcript"][-1].get("file", {}).get("name") == "lease.docx"
      and len(st["transcript"][-1]["text"]) <= 401)
check("a file is one message, however many pieces",
      st["messages"] == 1 and st["pieces"] == u["pieces"])
check("the chat is titled after the file", st["title"] == "lease.docx")
code, st = api("/api/chat", {"text": "How much is the security deposit for the flat?"})
check("a question about the file recalls the right piece",
      code == 200 and any("48,500" in i["text"] for i in st["context"]),
      str([i["text"][:50] for i in st["context"]]))

code, st = api("/api/upload", {"name": "deck.pptx", "data": b64(pptx([["Results"], ["P1-S 37.5% at S=4096"]]))})
check("a .pptx upload", code == 200 and st["upload"]["pieces"] >= 1)
code, st = api("/api/upload", {"name": "notes.txt", "data": b64("मेरी परीक्षा आठ नवंबर को है।\n\n我的考试在十一月八日。".encode("utf-16"))})
check("a UTF-16 text file in two scripts", code == 200, str(st))
for name, data, want in (("x.png", b64(b"\x89PNG"), 422), ("x.txt", "@@@", 422),
                         ("x.txt", None, 422), ("x.docx", b64(b"nope"), 422)):
    code, st = api("/api/upload", {"name": name, "data": data})
    check(f"bad upload {name} refused with a reason ({want})",
          code == want and st.get("error"), str((code, st)))
code, st = api("/api/upload", {"name": "huge.txt", "data": b64(b"a " * (I.MAX_DOC_CHARS // 2 + 10))})
check("a file with more text than one upload takes is refused, saying so",
      code == 422 and "limit" in st["error"], str(st)[:120])

paste = ("Here are my notes from the meeting.\n\n" + PROSE * 14 +
         "\n\nMy train to Pune leaves at 7:15 pm on 9 November from platform 3.\n\n" +
         PROSE * 14 + "\n\nWhich platform does my train leave from?")
code, st = api("/api/chat", {"text": paste})
check("a long paste is answered", code == 200, str(code))
check("a long paste is stored as many pieces", st.get("message_pieces", 0) > 10,
      str(st.get("message_pieces")))
check("the fact in the middle of the paste is recalled for its question",
      any("platform 3" in i["text"] for i in st["context"]))
check("the model gets the paste's question, not all of it",
      echo.last[1].endswith("Which platform does my train leave from?")
      and count_tokens(echo.last[1]) < 200 and count_tokens(paste) > A.LONG_MESSAGE)
code, st = api("/api/chat", {"text": "Which platform is my train?"})
check("and later, in a new message, it is still recalled",
      any("platform 3" in i["text"] for i in st["context"]))

re_ = app.chats.load(app.chat["id"])["session"]
check("reloading a chat with files and pastes rebuilds the same memory",
      [i["text"] for i in re_.state()["t3"]] == [i["text"] for i in app.session.state()["t3"]]
      and re_.n == app.session.n)
page = urllib.request.urlopen(BASE + "/").read().decode()
check("the page has the attach button and drop handling",
      'id="attach"' in page and "/api/upload" in page and "'drop'" in page)

print("\n-- what stress_inputs.py broke, pinned --")
check("a lone surrogate (half an emoji) is replaced, not stored",
      I.clean("cut \ud83d here \U0001F600") == "cut � here \U0001F600")
raw = b'{"text": "my locker is 214 \\ud800 ok"}'
req = urllib.request.Request(BASE + "/api/chat", data=raw, method="POST")
with urllib.request.urlopen(req) as r:
    code = r.status
check("a message with a lone surrogate is answered", code == 200)
code, st = api("/api/chat", {"text": "and my bike lock is 5521"})
check("... and the chat still works and saves after it", code == 200, str(code))
check("... and still reloads",
      app.chats.load(app.chat["id"])["session"].n == app.session.n)


def post_raw(path, data, headers=()):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
    c.putrequest("POST", path)
    for k, v in headers:
        c.putheader(k, v)
    c.endheaders()
    if data:
        c.send(data)
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"{}")
    c.close()
    return out


code, st = post_raw("/api/chat", b'{"text": ' + b"[" * 50_000 + b"]" * 50_000 + b"}",
                    [("Content-Length", str(100_010))])
check("JSON nested past the parser's limit is a 400, not a dropped connection",
      code == 400, str(code))
code, st = post_raw("/api/upload", b"", [("Content-Length", str(A.MAX_BODY + 1))])
check("a body over the limit is refused before it is read",
      code == 413 and "MB" in st.get("error", ""), str((code, st)))
check("a message of only invisible marks is empty",
      api("/api/chat", {"text": "​ ​\n"})[0] == 400)
_lim = I.MAX_XML_BYTES
I.MAX_XML_BYTES = 2000
try:
    code, st = api("/api/upload", {"name": "big.docx", "data": b64(docx(FILLER))})
finally:
    I.MAX_XML_BYTES = _lim
check("a Word file that unpacks too far is refused from its zip directory",
      code == 422 and "unpacks to" in st.get("error", ""), str((code, st)))
try:
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    code, st = api("/api/upload", {"name": "locked.pdf", "data": b64(buf.getvalue())})
    check("a password-protected PDF says so",
          code == 422 and "password" in st.get("error", ""), str((code, st)))
except ImportError:
    print("  SKIP  password-protected PDF (pypdf not installed)")

print("\n-- a question buried in a long message --")
long_msg = (" ".join(FILLER[:4]) + "\n\nAnyway, what is the deposit for the Indore flat?"
            "\n\nAlso: " + " ".join(FILLER[4:6]) + " Did you see the match? How was it?")
qs = I.queries(long_msg)
check("every question in a long message is its own query, closing ones first",
      qs[:2] == ["Did you see the match?", "How was it?"]
      and "Anyway, what is the deposit for the Indore flat?" in qs, str(qs))
check("a short message is its own single query", I.queries("where is it?") == ["where is it?"])
ctx, _ = app.session.context(long_msg)
check("the buried question still recalls its fact", "48,500" in ctx.text)

srv.shutdown()
print(f"\n{'=' * 46}\n  {passed} passed, {failed} failed\n{'=' * 46}")
sys.exit(1 if failed else 0)
