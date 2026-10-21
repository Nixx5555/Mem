#!/usr/bin/env python3
"""Stress test 2: big documents and odd input, through the running app.

Everything goes over HTTP to a real app.py server, exactly as the page sends
it, with a stand-in chat model that records the notes MemGate hands it. No
chat model is needed: the question is whether the right note reaches the
model, which is what the model's answer can never be better than.

Part A, needles in a document. A long document is built from real long-form
text (BEAM's assistant answers when data/beam is there, the repository's own
docs otherwise) with eight planted facts at depths from 5% to 95%. It is
uploaded as .txt, .pdf, .docx, .pptx or .html, then each fact is asked about,
and the notes sent to the model are checked for the answer. Sizes run from
5,000 tokens to the 400,000-character upload limit (about 100,000 tokens),
under three store budgets: the app's default (4,096), 32,768, and one bigger
than the document.

Part B, odd input. Text no test was written for: one 200,000-character word,
a book's worth of Chinese, right-to-left text with bidi marks, an emoji flood,
"zalgo" text, NULs and lone surrogates, deeply nested JSON, a 100,000-row CSV,
a Word file that unzips to hundreds of megabytes, a PowerPoint with 1,500
slides, a 400-page PDF, an encrypted PDF, a body far over the upload limit,
and more. Each must get an answer (200) or a reason (4xx) in bounded time,
never a crash (500), and the app must still chat normally afterwards.

Part C, load. Many messages in one chat, many visitors at once, and reloading
a big chat from disk.

    python3 stress_inputs.py               # all parts
    python3 stress_inputs.py --parts a     # just the needles
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import app as A
from memgate.reader import ChatModel
from memgate.utils import count_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "stress")

NEEDLES = [
    ("The access code for the Lab 4B storage locker is 7319.",
     "What is the access code for the Lab 4B storage locker?", "7319"),
    ("Priya Raghavan will chair the ethics review panel on 14 March 2027.",
     "Who will chair the ethics review panel?", "Priya Raghavan"),
    ("The backup generator in Block C must be serviced every 45 days.",
     "How often must the Block C backup generator be serviced?", "45 days"),
    ("Our Zurich supplier, Kellermann AG, quoted 18,400 euros for the sensor batch.",
     "How much did Kellermann AG quote for the sensor batch?", "18,400"),
    ("The project codename was changed from Falcon to Heron in August.",
     "What is the project's codename now?", "Heron"),
    ("Dr. Okafor's office hours moved to Thursdays at 4 pm.",
     "When are Dr. Okafor's office hours?", "Thursdays at 4 pm"),
    ("The training cluster has 64 A100 GPUs split across 8 nodes.",
     "How many GPUs does the training cluster have?", "64 A100"),
    ("The final dataset contains 12,850 annotated clinical notes.",
     "How many annotated clinical notes are in the final dataset?", "12,850"),
]


class Recorder(ChatModel):
    """Stands in for the chat model and keeps what it was sent."""

    def __init__(self):
        super().__init__("none")
        self.notes = []

    @property
    def enabled(self):
        return True

    @property
    def name(self):
        return "recorder"

    def reply(self, notes, message):
        self.notes.append(notes)
        return "noted."


class Client:
    def __init__(self, url):
        self.url = url
        self.cookie = None

    def call(self, path, body=None, raw=None, timeout=600):
        data = raw if raw is not None else (
            None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.url + path, data=data,
                                     method="POST" if data is not None else "GET")
        req.add_header("Content-Type", "application/json")
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                c = r.headers.get("Set-Cookie")
                if c:
                    self.cookie = c.split(";")[0]
                out = r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                out = e.code, json.loads(e.read() or b"{}")
            except ValueError:
                out = e.code, {}
        except Exception as e:                 # connection dropped, timeout
            out = 0, {"error": f"{type(e).__name__}: {e}"}
        return out + (time.time() - t0,)


def serve(app):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ------------------------------------------------------------- haystack
def haystack_paragraphs():
    """Distinct long-form paragraphs: BEAM's assistant answers if present."""
    paras = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "beam", "100K", "*", "chat.json")),
                    key=lambda p: int(p.split(os.sep)[-2]))[:6]:
        with open(f) as fh:
            for batch in json.load(fh):
                for sess in batch["turns"]:
                    for t in sess:
                        if t.get("role") == "assistant" and len(t.get("content", "")) > 400:
                            paras += [p.strip() for p in t["content"].split("\n\n")
                                      if len(p.strip()) > 80]
    if len(paras) < 2000:
        for f in [os.path.join(HERE, "..", "REPORT.md"),
                  os.path.join(HERE, "..", "SESSION_RECORD.md")] + \
                sorted(glob.glob(os.path.join(HERE, "..", "docs", "*.md"))):
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    paras += [p.strip() for p in fh.read().split("\n\n")
                              if len(p.strip()) > 80]
    return list(dict.fromkeys(paras))


def build_doc(paras, tokens, seed=0):
    """About `tokens` tokens of haystack with the needles planted at even depths."""
    rnd = random.Random(seed)
    pool = paras[:]
    rnd.shuffle(pool)
    body, n = [], 0
    for p in pool:
        if n >= tokens:
            break
        body.append(p)
        n += count_tokens(p)
    room = A.MAX_DOC_CHARS - sum(len(f) + 2 for f, _q, _a in NEEDLES) - 1000
    while len("\n\n".join(body)) > room:     # the upload limit, needles included
        body.pop()
    for k, (fact, _q, _a) in enumerate(NEEDLES):
        depth = 0.05 + 0.9 * k / (len(NEEDLES) - 1)
        body.insert(int(depth * len(body)), fact)
    return "\n\n".join(body)


def has(answer, text):
    """The answer as whole words ("Heron" is not in "coherent")."""
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text, re.I) is not None


def as_file(text, fmt):
    """`text` packaged as a file of format `fmt`, with real libraries."""
    paras = text.split("\n\n")
    if fmt == "txt":
        return text.encode("utf-8")
    if fmt == "html":
        import html
        return ("<html><head><style>p{}</style><script>var x=1;</script></head><body>"
                + "".join(f"<p>{html.escape(p)}</p>" for p in paras)
                + "</body></html>").encode("utf-8")
    if fmt == "docx":
        import docx
        d = docx.Document()
        for p in paras:
            d.add_paragraph(p)
        b = io.BytesIO()
        d.save(b)
        return b.getvalue()
    if fmt == "pptx":
        import pptx
        from pptx.util import Inches
        prs = pptx.Presentation()
        for i in range(0, len(paras), 3):
            s = prs.slides.add_slide(prs.slide_layouts[5])
            s.shapes.title.text = f"Part {i // 3 + 1}"
            tb = s.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
            tb.text_frame.text = "\n".join(paras[i:i + 3])
        b = io.BytesIO()
        prs.save(b)
        return b.getvalue()
    if fmt == "pdf":
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate
        from xml.sax.saxutils import escape
        b = io.BytesIO()
        st = getSampleStyleSheet()["Normal"]
        SimpleDocTemplate(b).build([Paragraph(escape(p), st) for p in paras])
        return b.getvalue()
    raise ValueError(fmt)


def part_a(client, rec, paras, quick=False):
    sizes = [5_000, 20_000, 60_000, 100_000] if not quick else [5_000, 20_000]
    budgets = [None, 32_768, 200_000]
    runs = [("txt", n, S) for n in sizes for S in budgets]
    runs += [(fmt, 20_000, S) for fmt in ("pdf", "docx", "pptx", "html")
             for S in (None, 200_000)]
    rows = []
    for fmt, n, S in runs:
        doc = build_doc(paras, n, seed=n)
        data = as_file(doc, fmt)
        st, _, _ = client.call("/api/chats/new", {"scale": "full", "store_budget": S})
        assert st == 200
        st, up, secs = client.call("/api/upload", {
            "name": f"report.{fmt}", "data": base64.b64encode(data).decode()})
        row = {"format": fmt, "doc_tokens": count_tokens(doc),
               "file_kb": round(len(data) / 1024), "S": S or 4096,
               "status": st, "upload_s": round(secs, 1)}
        if st != 200:
            row["error"] = up.get("error")
            rows.append(row)
            print(f"  {fmt:<5} {n:>7,} tok S={row['S']:>7,}  upload {st}: {row['error']}")
            continue
        u = up["upload"]
        row.update(pieces=u["pieces"], held=u["held"], held_tokens=u["held_tokens"],
                   forgotten=u["forgotten"])
        hits, ask_s = [], []
        for fact, q, ans in NEEDLES:
            rec.notes.clear()
            st, out, secs = client.call("/api/chat", {"text": q})
            ask_s.append(secs)
            notes = rec.notes[-1] if rec.notes else ""
            rag = " ".join(n["text"] for n in out.get("compare", {}).get(
                "rows", {}).get("rag", {}).get("notes", []))
            hits.append((has(ans, notes), has(ans, rag)))
        row.update(found=sum(h for h, _ in hits), rag_found=sum(r for _, r in hits),
                   of=len(NEEDLES), ask_s_max=round(max(ask_s), 2),
                   by_depth="".join("x" if h else "." for h, _ in hits))
        rows.append(row)
        print(f"  {fmt:<5} {row['doc_tokens']:>7,} tok S={row['S']:>7,}  "
              f"upload {row['upload_s']:>5.1f}s  {u['pieces']:>5} pieces, "
              f"{u['held']:>5} held  needles MemGate {row['found']}/8 "
              f"[{row['by_depth']}]  RAG {row['rag_found']}/8  "
              f"slowest ask {row['ask_s_max']}s")
    return rows


# ------------------------------------------------------------ odd input
def zip_bomb_docx(mb):
    """A real .docx whose document.xml unzips to `mb` megabytes."""
    para = "<w:p><w:r><w:t>" + "A" * 1000 + "</w:t></w:r></w:p>"
    body = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxml'
            'formats.org/wordprocessingml/2006/main"><w:body>'
            + para * (mb * 1000) + "</w:body></w:document>")
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", body)
    return b.getvalue()


def many_slides(n):
    import pptx
    prs = pptx.Presentation()
    for i in range(n):
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = f"Slide {i}"
        s.placeholders[1].text = f"Point {i}: revenue grew {i % 97}% in region {i % 13}."
    b = io.BytesIO()
    prs.save(b)
    return b.getvalue()


def long_pdf(pages):
    from reportlab.pdfgen import canvas
    b = io.BytesIO()
    c = canvas.Canvas(b)
    for i in range(pages):
        for k in range(40):
            c.drawString(40, 800 - k * 19, f"Page {i} line {k}: the quarterly figure "
                                           f"for unit {i * 40 + k} was {i * k % 991}.")
        c.showPage()
    c.save()
    return b.getvalue()


def encrypted_pdf():
    from pypdf import PdfWriter, PdfReader
    w = PdfWriter()
    w.append(PdfReader(io.BytesIO(long_pdf(1))))
    w.encrypt("secret")
    b = io.BytesIO()
    w.write(b)
    return b.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


def odd_cases():
    rnd = random.Random(7)
    cjk = "".join(chr(rnd.randint(0x4E00, 0x9FA5)) for _ in range(150_000))
    words = ["alpha", "beta", "gamma", "delta"]
    return [
        # (name, endpoint, body or raw bytes)
        ("one 199,000-char word", "/api/chat", {"text": "x" * 199_000}),
        ("random base64 blob 150k", "/api/chat",
         {"text": b64(os.urandom(110_000))}),
        ("Chinese, 150,000 chars, no spaces", "/api/chat", {"text": cjk}),
        ("Arabic + Hebrew with bidi marks", "/api/chat",
         {"text": ("‮مرحبا بالعالم، موعد التسليم في ٤ نوفمبر ‬ שלום עולם " * 400)}),
        ("emoji flood 60k", "/api/chat", {"text": "😀👍🏽👨‍👩‍👧‍👦🇮🇳" * 6000}),
        ("zalgo text", "/api/chat",
         {"text": "M̴̢̛̖̗̘̙̜̝̞̟̠̤̥̦̩̪̫̬̭̮̯̰̱̲̳̹̺̻̼͇͈͉͍͎̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̽̾̿̀́͂̓̈́͆͊͋͌̕̚ͅy exam is on 5 May" * 300}),
        ("NUL and control chars", "/api/chat",
         {"text": "deadline\x00 is\x07 \x1b[31m 9 June\x00" * 2000}),
        ("lone surrogates", "/api/chat",
         None, b'{"text": "broken \\ud800 surrogate \\udfff here, the room is 214"}'),
        ("only whitespace and zero-width", "/api/chat", {"text": " \n\t​​  "}),
        ("exactly the 200,000-char limit", "/api/chat",
         {"text": ("The meeting is at 3 pm in room 12. " * 6000)[:200_000]}),
        ("one char over the limit", "/api/chat", {"text": "a" * 200_001}),
        ("same sentence 5,000 times", "/api/chat",
         {"text": "I like pizza. " * 5000}),
        ("4,000 one-word lines", "/api/chat",
         {"text": "\n".join(rnd.choice(words) for _ in range(4000))}),
        ("deeply nested JSON body", "/api/chat", None,
         b'{"text": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"),
        ("text is a number", "/api/chat", {"text": 12345}),
        ("not JSON at all", "/api/chat", None, b"\xff\xfe garbage"),
        ("100,000-row CSV", "/api/upload",
         {"name": "rows.csv", "data": b64("\n".join(
             f"{i},user{i},{i * 7 % 1000},2026-0{i % 9 + 1}-1{i % 9}"
             for i in range(100_000)).encode())}),
        ("minified JSON 1 MB", "/api/upload",
         {"name": "dump.json", "data": b64(json.dumps(
             [{"id": i, "v": "x" * 20} for i in range(30_000)]).encode())}),
        ("UTF-16 text file", "/api/upload",
         {"name": "notes.txt", "data": b64(("My flight is on 9 June. " * 50).encode("utf-16"))}),
        ("Latin-1 text file", "/api/upload",
         {"name": "notes.txt", "data": b64("Café crème costs 4 € in Zürich.".encode("cp1252"))}),
        ("binary renamed .txt", "/api/upload",
         {"name": "x.txt", "data": b64(os.urandom(50_000))}),
        ("PNG", "/api/upload", {"name": "a.png", "data": b64(b"\x89PNG\r\n" + os.urandom(999))}),
        ("damaged .docx", "/api/upload", {"name": "a.docx", "data": b64(b"PK\x03\x04junk")}),
        ("docx that unzips to 300 MB", "/api/upload",
         {"name": "bomb.docx", "data": b64(zip_bomb_docx(300))}),
        ("pptx with 1,500 slides", "/api/upload",
         {"name": "big.pptx", "data": b64(many_slides(1500))}),
        ("400-page PDF", "/api/upload", {"name": "long.pdf", "data": b64(long_pdf(400))}),
        ("password-protected PDF", "/api/upload",
         {"name": "locked.pdf", "data": b64(encrypted_pdf())}),
        ("HTML, 20,000 nested divs", "/api/upload",
         {"name": "deep.html", "data": b64(("<div>" * 20000 + "the code is 4417"
                                            + "</div>" * 20000).encode())}),
        ("file over 15 MB", "/api/upload",
         {"name": "big.txt", "data": b64(b"word " * 3_300_000)}),
        ("bad base64", "/api/upload", {"name": "a.txt", "data": "%%%not base64%%%"}),
        ("empty file", "/api/upload", {"name": "a.txt", "data": ""}),
        ("file name with path and markup", "/api/upload",
         {"name": "../../etc/<b>passwd</b>].txt", "data": b64(b"root code 0000")}),
    ]


def part_b(client):
    rows = []
    for case in odd_cases():
        name, path = case[0], case[1]
        body = case[2]
        raw = case[3] if len(case) > 3 else None
        st, out, secs = client.call(path, body, raw=raw)
        ok_after = client.call("/api/chat", {"text": "My bike lock code is 5521."})[0] == 200
        verdict = ("ok" if st == 200 else "refused" if 400 <= st < 500 else
                   "CRASH" if st >= 500 else "DROPPED")
        detail = out.get("error") or (
            f"{out.get('upload', {}).get('pieces', out.get('message_pieces', '?'))} pieces")
        rows.append({"case": name, "endpoint": path, "status": st, "verdict": verdict,
                     "secs": round(secs, 2), "detail": str(detail)[:140],
                     "app_ok_after": ok_after})
        print(f"  {verdict:<8} {st:>3} {secs:>6.1f}s  {name:<36} {str(detail)[:80]}"
              + ("" if ok_after else "   APP BROKEN AFTER"))
    return rows


# ------------------------------------------------------------------ load
def part_c(client, url, paras):
    rows = []
    # 1. one long chat: 600 typed messages, latency first vs last
    client.call("/api/chats/new", {"scale": "full"})
    lat = []
    rnd = random.Random(3)
    for i in range(600):
        p = rnd.choice(paras)
        msg = p[:rnd.choice([40, 120, 300, 600])]
        st, _, secs = client.call("/api/chat", {"text": msg})
        assert st == 200, st
        lat.append(secs)
    st, state, _ = client.call("/api/state")
    rows.append({"test": "600 messages in one chat",
                 "first_50_mean_s": round(sum(lat[:50]) / 50, 3),
                 "last_50_mean_s": round(sum(lat[-50:]) / 50, 3),
                 "max_s": round(max(lat), 2),
                 "stored_tokens": state.get("stored_tokens"),
                 "pieces": state.get("pieces")})
    print(f"  600 messages: mean {rows[-1]['first_50_mean_s']}s first 50, "
          f"{rows[-1]['last_50_mean_s']}s last 50, max {rows[-1]['max_s']}s")
    return rows


def part_c_reload_and_visitors(paras):
    rows = []
    # 2. reload: a chat with a 100k-token upload, re-opened from disk
    folder = tempfile.mkdtemp()
    rec = Recorder()
    app = A.App(rec, A.Chats(folder))
    doc = build_doc(paras, 100_000, seed=1)
    app.upload("big.txt", b64(doc.encode()))
    for i in range(50):
        app.say(f"Remember item {i}: the shelf {i} holds box {i * 3}.")
    path = app.chats._path(app.chat["id"])
    kb = os.path.getsize(path) / 1024
    t0 = time.time()
    app2 = A.App(rec, A.Chats(folder))
    secs = time.time() - t0
    same = app2.session.state()["stored_tokens"] == app.session.state()["stored_tokens"]
    rows.append({"test": "reopen chat with 100k-token file + 100 messages",
                 "file_kb": round(kb), "reload_s": round(secs, 1), "same_state": same})
    print(f"  reload of a {kb:,.0f} KB chat: {secs:.1f}s, state identical: {same}")

    # 3. public mode: 40 visitors at once, 5 messages each
    vfolder = tempfile.mkdtemp()
    vis = A.Visitors(Recorder(), vfolder)
    srv, url = serve_handler(vis)
    errors, lat = [], []

    def visitor(k):
        c = Client(url)
        for i in range(5):
            st, out, secs = c.call("/api/chat", {"text": f"Visitor {k} note {i}: my seat is {k}{i}."})
            lat.append(secs)
            if st != 200:
                errors.append(st)
        st, out, _ = c.call("/api/state")
        mine = all(f"Visitor {k} " in m["text"] or m["speaker"] != "user"
                   for m in out.get("transcript", []))
        if not mine:
            errors.append("leak")

    t0 = time.time()
    ths = [threading.Thread(target=visitor, args=(k,)) for k in range(40)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    lat.sort()
    rows.append({"test": "40 visitors x 5 messages at once",
                 "wall_s": round(time.time() - t0, 1), "errors": len(errors),
                 "leaks": errors.count("leak"),
                 "p50_s": round(lat[len(lat) // 2], 2), "p95_s": round(lat[int(len(lat) * .95)], 2)})
    print(f"  40 visitors at once: {rows[-1]['wall_s']}s wall, errors {len(errors)}, "
          f"p50 {rows[-1]['p50_s']}s p95 {rows[-1]['p95_s']}s")
    srv.shutdown()

    # 4. concurrent messages into ONE chat: nothing lost, order kept per message
    capp = A.App(Recorder(), A.Chats(tempfile.mkdtemp()))
    srv, url = serve_handler(capp)
    c = Client(url)
    ths = [threading.Thread(target=lambda k=k: c.call(
        "/api/chat", {"text": f"Parallel note {k}: code {k * 11}."})) for k in range(30)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    tr = capp.session.transcript
    got = {m["text"] for m in tr if m["speaker"] == "user"}
    rows.append({"test": "30 messages into one chat at once",
                 "kept": sum(f"Parallel note {k}: code {k * 11}." in got for k in range(30)),
                 "transcript": len(tr)})
    print(f"  30 parallel messages into one chat: {rows[-1]['kept']}/30 kept, "
          f"transcript {len(tr)} entries")
    srv.shutdown()
    return rows


def serve_handler(app):
    return serve(app)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    paras = haystack_paragraphs()
    print(f"haystack: {len(paras):,} distinct paragraphs")
    rec = Recorder()
    app = A.App(rec, A.Chats(tempfile.mkdtemp()))
    srv, url = serve(app)
    client = Client(url)
    out = {}
    if "a" in args.parts:
        print("\nPart A: needles in a document")
        out["needles"] = part_a(client, rec, paras, args.quick)
    if "b" in args.parts:
        print("\nPart B: odd input")
        out["odd"] = part_b(client)
    if "c" in args.parts:
        print("\nPart C: load")
        out["load"] = part_c(client, url, paras) + part_c_reload_and_visitors(paras)
    with open(os.path.join(RESULTS, f"inputs_{args.parts}.json"), "w") as f:
        json.dump(out, f, indent=1)
    srv.shutdown()


if __name__ == "__main__":
    main()
