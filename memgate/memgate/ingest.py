"""Any text in: cleaning, chunking, file extraction and open-input scoring.

The experiments feed MemGate dialogue turns: a sentence or two of English, the
shape LoCoMo and REALTALK are made of. A person using the app types whatever
they like -- a 30-page PDF, a code file, a paragraph in Hindi, a row of emoji.
This module is what stands between that and `policy.observe()`, so the memory
layer itself never sees anything the experiments did not test it on:

    clean(text)       control characters, zero-width and bidi-override marks
                      removed; Unicode normalised; line breaks kept
    chunk(text)       long input split into turn-sized pieces (paragraphs, then
                      sentences, then lines, then words) of at most MAX_CHUNK
                      tokens, so each piece is scored, stored and retrieved on
                      its own, the way a dialogue turn is
    extract(name, b)  the text inside an uploaded file: plain text in any
                      encoding, Markdown, code, CSV, JSON, HTML, PDF, Word
                      (.docx) and PowerPoint (.pptx)
    focus(text)       what to retrieve with when a message is long: its
                      questions, or its opening and closing pieces
    GuardedScorer     wraps the scorer for input it was never trained on

Why chunk. One 5,000-token paste stored as one item cannot fit in a
2,048-token context, so nothing in it could ever be recalled; MiniLM reads
only its first 256 word pieces, so its embedding describes the opening and
nothing else; and the learned scorer's length features would be ten standard
deviations outside anything it was fitted on (it scores a paragraph repeated
200 times at 1.00). Turn-sized pieces fix all three at once.

Nothing here changes an experiment. The benchmark scripts build their
policies without it, and test_bench.py pins their numbers.
"""
import base64
import binascii
import html
import io
import re
import unicodedata
import zipfile
from html.parser import HTMLParser
from typing import List, Tuple
from xml.etree import ElementTree

from .scoring import (FILLER_PAT, MONEY_PAT, MONTH_PAT, NUM_PAT, TIME_PAT,
                      HeuristicScorer, Scorer, message_only)
from .utils import count_tokens

# A message up to WHOLE tokens is stored whole, as it always was: long chat
# messages stay one memory item. Longer text is cut into pieces of at most
# MAX_CHUNK tokens. LoCoMo turns average 21 words (sd 19, from
# scorer_pooled_lr.json), about 28 tokens, and a 48-token piece (a sentence
# or two) stays near that. Measured on REPORT.md uploaded as a file, bigger
# pieces push the learned scorer towards 1.0 for all of them: 46% of 80-token
# pieces score above 0.9, 36% at 48 tokens and 32% at 40, and when most
# pieces tie near 1.0 eviction has little to choose between. 48 keeps most
# of the gain with a fifth fewer items than 40.
WHOLE = 120
MAX_CHUNK = 48
# Largest text one upload may add, about 100,000 tokens: a long report or a
# small book. Ingesting it takes seconds; much more and a chat stalls.
MAX_DOC_CHARS = 400_000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Most XML a Word or PowerPoint upload may unpack to. A .docx is a zip, and
# 300 KB of it can inflate to 300 MB of XML (3.8 GB of memory and a minute to
# parse, measured by stress_inputs.py); the 400,000 characters MAX_DOC_CHARS
# allows need a few MB of even heavily formatted XML.
MAX_XML_BYTES = 64 * 1024 * 1024

# ------------------------------------------------------------------ clean
# Zero-width spaces/joiners, the BOM, and bidi embedding/override/isolate
# marks: invisible, and the bidi ones can make stored text display reversed.
# ZWJ (U+200D) is kept: it is what holds family and flag emoji together.
_INVISIBLE = re.compile("[​‌‎‏‪-‮⁠-⁤"
                        "⁦-⁩﻿­]")
# Half of an emoji or other astral character: valid in JSON ("\ud83d") and
# what a browser sends when a paste is cut mid-character, but not encodable
# as UTF-8, so one stored in a chat made every later save of it fail.
SURROGATE = re.compile("[\ud800-\udfff]")


def clean(text: str) -> str:
    """Text safe to store: NFC, no control characters but line breaks and
    tabs, no invisible marks, lone surrogates replaced by U+FFFD, \\r\\n
    folded to \\n, outer space trimmed."""
    if not isinstance(text, str):
        return ""
    text = SURROGATE.sub("\ufffd", text)
    t = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    t = _INVISIBLE.sub("", t)
    t = "".join(c for c in t if c in "\n\t" or unicodedata.category(c) != "Cc")
    return t.strip()


def one_line(text: str) -> str:
    """The form memory stores: whitespace runs collapsed to single spaces."""
    return " ".join(text.split())


# ------------------------------------------------------------------ chunk
# Sentence ends: Latin and Devanagari punctuation followed by space, or CJK
# full stops, which are not followed by a space.
_SENT = re.compile(r"(?<=[.!?…।॥])\s+|(?<=[。！？])")
_PARA = re.compile(r"\n\s*\n")


def _split_hard(s: str, limit: int) -> List[str]:
    """Pieces of one over-long run with no sentence or line break in it:
    by words, and by characters for a run with no spaces (CJK, a hash)."""
    out, cur = [], ""
    for w in s.split(" "):
        if count_tokens(w) > limit:            # one "word" bigger than a piece
            if cur:
                out.append(cur)
                cur = ""
            step = max(1, limit * 2)           # >= 2 chars per token, CJK ~1
            while w:
                n = step
                while n > 1 and count_tokens(w[:n]) > limit:
                    n = n * 3 // 4
                out.append(w[:n])
                w = w[n:]
            continue
        nxt = f"{cur} {w}" if cur else w
        if cur and count_tokens(nxt) > limit:
            out.append(cur)
            cur = w
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


def _units(para: str, limit: int) -> List[str]:
    """A paragraph as pieces no bigger than `limit`: whole if it fits, else its
    sentences, else its lines (code, lists, tables), else words."""
    if count_tokens(one_line(para)) <= limit:
        return [para]
    sents = [s for s in _SENT.split(para) if s.strip()]
    if len(sents) == 1:
        lines = [x for x in para.split("\n") if x.strip()]
        if len(lines) > 1:
            return [u for x in lines for u in _units(x, limit)]
        return _split_hard(one_line(para), limit)
    return [u for s in sents for u in _units(s, limit)]


def chunk(text: str, limit: int = MAX_CHUNK, whole: int = WHOLE) -> List[str]:
    """`text` as turn-sized pieces, in order, each one line, none empty.

    A message of up to `whole` tokens is returned whole, exactly as memory
    always stored it, so ordinary chat is unchanged. Longer text is cut at
    paragraph, sentence, line and finally word boundaries, and neighbouring
    small units are packed back together up to `limit` so a list of short
    lines does not become a pile of one-word "messages" the scorer would
    discard as small talk.
    """
    t = clean(text)
    if not t:
        return []
    if count_tokens(one_line(t)) <= max(limit, whole):
        return [one_line(t)]
    units = [one_line(u) for p in _PARA.split(t) if p.strip()
             for u in _units(p, limit)]
    out, cur = [], ""
    for u in units:
        if not u:
            continue
        nxt = f"{cur} {u}" if cur else u
        if cur and count_tokens(nxt) > limit:
            out.append(cur)
            cur = u
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


def focus(text: str, limit: int = WHOLE) -> str:
    """What to retrieve with for this message.

    Short messages are their own query. For a long one, the embedder would
    read only its opening, so the query is the questions it asks, or failing
    that its first and last pieces (where people put "here is X" and "please
    do Y").
    """
    t = clean(text)
    if count_tokens(one_line(t)) <= limit:
        return one_line(t)
    paras = [p for p in _PARA.split(t) if p.strip()]
    ask = lambda ps: [one_line(s) for p in ps for line in p.split("\n")
                      for s in _SENT.split(line) if s.strip().endswith(("?", "？"))]
    qs = ask(paras[-1:]) or ask(paras)       # the closing ask comes first
    if qs:
        q = " ".join(qs[-3:])
    else:
        pieces = chunk(t)
        q = f"{pieces[0]} {pieces[-1]}"
    return q if count_tokens(q) <= limit else _split_hard(q, limit)[0]


def queries(text: str, limit: int = WHOLE, most: int = 4) -> List[str]:
    """What to retrieve with, as separate queries, for MemGate's own memory.

    focus() joins a long message's questions into one string, which works
    when the ask is at the end. stress_questions.py --long-questions buried a
    LoCoMo question in 150-450 words of other chat, often with more chat (and
    its own questions) after it; focus() then held the real question only 56%
    of the time and answer recall fell from 53% to 42%. Here every question
    in the message (the closing paragraph's first, then the rest from the end
    back, at most `most`) is its own query, and the store interleaves their
    rankings (ThreeTierStore.retrieve), so the right one is in there somewhere
    and its best matches still reach the context.
    """
    t = clean(text)
    if count_tokens(one_line(t)) <= limit:
        return [one_line(t)]
    paras = [p for p in _PARA.split(t) if p.strip()]
    ask = lambda ps: [one_line(s) for p in ps for line in p.split("\n")
                      for s in _SENT.split(line) if s.strip().endswith(("?", "？"))]
    last = ask(paras[-1:])
    rest = [q for q in reversed(ask(paras[:-1])) if q not in last]
    qs = list(dict.fromkeys(last + rest))[:most]
    if not qs:
        return [focus(t, limit)]
    return [q if count_tokens(q) <= limit else _split_hard(q, limit)[0] for q in qs]


# ---------------------------------------------------------------- extract
class ExtractError(ValueError):
    """The upload could not be read; the message says why, for the page."""


TEXT_EXT = {
    "txt", "md", "markdown", "rst", "log", "csv", "tsv", "json", "jsonl",
    "yaml", "yml", "toml", "ini", "cfg", "conf", "xml", "srt", "vtt", "tex",
    "bib", "py", "js", "ts", "tsx", "jsx", "java", "c", "h", "cpp", "hpp",
    "cs", "go", "rs", "rb", "php", "swift", "kt", "scala", "r", "m", "sql",
    "sh", "bash", "zsh", "ps1", "bat", "css", "scss", "ipynb", "env", "",
}


def decode(data: bytes) -> str:
    """Bytes of a text file as text, whatever the encoding.

    Order: a BOM if there is one, then UTF-8, then UTF-16 when the NUL
    pattern says so, then Windows-1252 (which maps every byte, so this never
    fails on a text file). Binary files are refused rather than stored as
    mojibake.
    """
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"),
                     (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return data.decode(enc, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    head = data[:4096]
    if head.count(b"\x00") > len(head) // 4:
        even, odd = head[0::2].count(0), head[1::2].count(0)
        if odd > even * 4:
            return data.decode("utf-16-le", errors="replace")
        if even > odd * 4:
            return data.decode("utf-16-be", errors="replace")
    # Text has no NULs and few other control bytes; executables, images and
    # archives are full of them.
    ctrl = sum(1 for b in head if b < 32 and b not in (9, 10, 12, 13, 27))
    if b"\x00" in head or ctrl > len(head) // 20:
        raise ExtractError("that looks like a binary file, not text")
    return data.decode("cp1252", errors="replace")


class _HTMLText(HTMLParser):
    """Visible text of an HTML page, one block element per line."""
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "header", "footer", "pre", "blockquote",
             "table", "ul", "ol", "dt", "dd", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "template"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "template"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


def html_text(s: str) -> str:
    p = _HTMLText()
    p.feed(s)
    p.close()
    lines = (" ".join(x.split()) for x in "".join(p.out).split("\n"))
    return "\n\n".join(x for x in lines if x)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _zip(data: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ExtractError("the file is damaged or not really that format")


def _check_unpacked(z: zipfile.ZipFile, names) -> None:
    """Refuse a zip whose parts we would read inflate past MAX_XML_BYTES,
    before reading any of them (the sizes are in the zip's directory)."""
    size = sum(z.getinfo(n).file_size for n in names)
    if size > MAX_XML_BYTES:
        raise ExtractError(f"the file unpacks to {size / 1e6:,.0f} MB, more "
                           f"than the {MAX_XML_BYTES // (1024 * 1024)} MB a "
                           f"document can; save it as .pdf or .txt instead")


def docx_text(data: bytes) -> str:
    """Paragraph text of a Word file, tables included, without python-docx."""
    z = _zip(data)
    try:
        _check_unpacked(z, ["word/document.xml"])
        root = ElementTree.fromstring(z.read("word/document.xml"))
    except KeyError:
        raise ExtractError("no document inside this .docx")
    paras = []
    for p in root.iter(_W + "p"):
        bits = []
        for el in p.iter():
            if el.tag == _W + "t" and el.text:
                bits.append(el.text)
            elif el.tag == _W + "tab":
                bits.append("\t")
            elif el.tag in (_W + "br", _W + "cr"):
                bits.append("\n")
        if "".join(bits).strip():
            paras.append("".join(bits))
    return "\n\n".join(paras)


def pptx_text(data: bytes) -> str:
    """Slide text of a PowerPoint file, slide by slide, notes left out."""
    z = _zip(data)
    num = lambda n: int(re.sub(r"\D", "", n) or 0)
    slides = sorted((n for n in z.namelist()
                     if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=num)
    _check_unpacked(z, slides)
    out = []
    for n in slides:
        root = ElementTree.fromstring(z.read(n))
        paras = ["".join(t.text or "" for t in p.iter(_A + "t"))
                 for p in root.iter(_A + "p")]
        body = "\n".join(x for x in paras if x.strip())
        if body:
            out.append(f"Slide {num(n)}\n{body}")
    return "\n\n".join(out)


def pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ExtractError("reading PDFs needs the pypdf package: "
                           "pip install pypdf, then restart the app")
    try:
        r = PdfReader(io.BytesIO(data))
        if r.is_encrypted:
            # Many PDFs are "encrypted" with an empty password just to set
            # permissions; those open. decrypt() returns 0 for a real one.
            try:
                opened = r.decrypt("")
            except Exception:
                opened = 0
            if not opened:
                raise ExtractError("the PDF is password-protected; remove the "
                                   "password or upload the text another way")
        pages = [(p.extract_text() or "").strip() for p in r.pages]
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"could not read the PDF ({type(exc).__name__})")
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise ExtractError("the PDF has no text layer (a scan?); "
                           "copy the text in or run OCR on it first")
    return text


def notebook_text(s: str) -> str:
    import json
    try:
        nb = json.loads(s)
        cells = nb.get("cells", [])
    except (ValueError, AttributeError):
        return s
    out = []
    for c in cells:
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if src.strip():
            out.append(src)
    return "\n\n".join(out)


def extract(name: str, data: bytes) -> str:
    """The text in an uploaded file, cleaned. Raises ExtractError with a
    reason a person can act on when there is none to be had."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractError(f"the file is {len(data) / 1e6:.1f} MB; the limit "
                           f"is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "pdf" or data.startswith(b"%PDF"):
        text = pdf_text(data)
    elif ext == "docx":
        text = docx_text(data)
    elif ext == "pptx":
        text = pptx_text(data)
    elif ext in ("doc", "ppt", "xls", "xlsx", "odt", "pages", "key", "rtf"):
        raise ExtractError(f".{ext} files aren't supported; save it as .docx, "
                           ".pdf or .txt, or paste the text in")
    elif ext in ("png", "jpg", "jpeg", "gif", "webp", "heic", "mp3", "mp4",
                 "wav", "mov", "zip", "gz", "exe", "dmg"):
        raise ExtractError("that's not a text file: MemGate stores text")
    else:
        text = decode(data)
        if ext in ("html", "htm", "xhtml") or (
                ext not in TEXT_EXT and re.search(r"<(html|body|p|div)\b", text[:2000], re.I)):
            text = html_text(text)
        elif ext == "ipynb":
            text = notebook_text(text)
    text = clean(text)
    if not text:
        raise ExtractError("the file has no text in it")
    return text


def decode_upload(b64: str) -> bytes:
    if not isinstance(b64, str) or not b64:
        raise ExtractError("no file data was sent")
    if b64.startswith("data:"):                        # a data: URL
        b64 = b64.split(",", 1)[-1]
    if len(b64) > MAX_UPLOAD_BYTES * 4 // 3 + 4:
        raise ExtractError(f"the file is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ExtractError("the file data was damaged in transit")


# ---------------------------------------------------------------- scoring
_LETTER_OR_DIGIT = re.compile(r"[^\W_]")
_CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯]")
_UWORD = re.compile(r"[^\W_]+")
SMALL_TALK = re.compile(
    r"^(lol|lmao|haha+|hehe+|hmm+|wow|k|kk|ty|thx|thnx|np|yup|yep|nope|nah|"
    r"oh|ohh+|ah|uh|um+|right|alright|okie|okk+|done|wait|what|really|omg|"
    r"brb|gn|gm|good night|ok cool|cool thanks|got it thanks|haan|accha|"
    r"theek hai|ji)[\s!.,?]*$", re.I)
# Strong signs of code: any one is enough. Weak ones need company.
_CODE_STRONG = re.compile(
    r"\bdef \w+\s*\(.*\)\s*(->[^:]*)?:|\bfunction\b\s*\w*\s*\(|\bclass \w+\s*[:({]|"
    r"\bfrom [\w.]+ import\b|^\s*import [\w.]+\s*(;|$|as\b)|#include\s*<|"
    r"\bpublic (static|class|void)\b|\bconsole\.\w+\(|\bSELECT\b.+\bFROM\b|"
    r"\bfor\s*\(.*;.*\)|\b(const|let|var) \w+\s*=|\w+\s*=>|^\s*```", re.I | re.M)
_CODE_WEAK = re.compile(
    r"[{};]\s*$|[{}]|\w+\(\)|</?[a-z]+[^>]*>|^\s*[$#>] \w|\breturn\b|"
    r"\bprint\(|==|!=|\+=|&&|\|\||::|->", re.I | re.M)
NEUTRAL = 0.5


def latin_share(t: str) -> float:
    """Share of the letters that are Latin script: the only script the
    scorers' word features (utils._WORD is [A-Za-z0-9']) can see."""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if ord(c) < 0x250) / len(letters)


def n_words(t: str) -> int:
    """Words in any script; CJK, written without spaces, counts ~2 chars a word."""
    cjk = len(_CJK.findall(t))
    rest = _UWORD.findall(_CJK.sub(" ", t))
    return len(rest) + (cjk + 1) // 2


def looks_like_code(t: str) -> bool:
    if _CODE_STRONG.search(t):
        return True
    weak = len(_CODE_WEAK.findall(t))
    sym = sum(1 for c in t if c in "{}[]()<>;=_#$*/\\|`")
    return weak >= 3 or (len(t) >= 24 and sym / len(t) > 0.12 and weak >= 1)


class GuardedScorer(Scorer):
    """A scorer, made safe for open-ended input. The app wraps its scorer in
    this; the experiments never do.

    The P1 rules and the learned scorer were both built on English dialogue.
    Shown other input they misjudge it in ways that matter:

      * "Hi!" scores 0.48 with the learned scorer, more than some dates
        (the BEAM-trained model put it at 0.68, above a flight date)
      * a sentence in Hindi or Chinese has no [A-Za-z] words, so P1 calls it
        small talk (0.02) and the learned scorer guesses from digits alone
      * code scores 0.15: below τ_low, so P1 mode would forget it at once
      * a paragraph repeated 200 times scores 1.00

    Each guard below fires only on input of one of those kinds and leaves an
    ordinary English message to the wrapped scorer untouched, so on the chat
    the scorer was trained for it is the same scorer.

      no letters or digits    (emoji, punctuation)        -> 0.02 filler
      greeting / thanks / lol, or <= 2 words, no digits   -> at most 0.05 filler
      mostly non-Latin script, >= 3 words                 -> 0.50 neutral
      code, markup, shell                                 -> at least 0.50
      the same few words over and over                    -> at most 0.15

    "Neutral" is the honest answer to text the scorer cannot read: kept while
    there is room, forgotten after facts it could judge and before small talk.
    """

    def __init__(self, inner: Scorer):
        self.inner = inner
        self._p1 = HeuristicScorer()
        self.name = inner.name
        self.last_guard = None

    def guard(self, text: str) -> Tuple[str, float, str]:
        """(guard name, utility, label) for input a guard covers, else None."""
        t = clean(message_only(text))
        if not _LETTER_OR_DIGIT.search(t):
            return "no-words", 0.02, "filler"
        n = n_words(t)
        has_digit = bool(NUM_PAT.search(t))
        if FILLER_PAT.match(t) or SMALL_TALK.match(t) or (n <= 2 and not has_digit):
            return "small-talk", 0.05, "filler"
        if looks_like_code(t):
            return "code", None, "code"
        if n >= 3 and latin_share(t) < 0.5:
            return "script", NEUTRAL, "fact" if has_digit else "context"
        ws = [w.lower() for w in _UWORD.findall(t)]
        if len(ws) >= 12 and len(set(ws)) / len(ws) < 0.3:
            return "repeated", 0.15, "context"
        return None

    # A short message naming a date, a time or an amount is the kind of fact
    # a person types and asks about later ("My flight is on 14 December at
    # 6:40 am"). The learned scorer trained with BEAM, whose long answers are
    # full of numbers that are never asked about, learned to ignore those
    # signals and put that line at 0.04, level with "Hi!"; P1's rules give it
    # 0.62. For such messages the score is at least P1's.
    SPECIFIC_WORDS = 30
    use_floor = True

    def _specific(self, t):
        return (n_words(t) <= self.SPECIFIC_WORDS
                and bool(MONTH_PAT.search(t) or TIME_PAT.search(t)
                         or MONEY_PAT.search(t)))

    def score(self, text, speaker="user"):
        g = self.guard(text)
        self.last_guard = g[0] if g else None
        if g is None:
            u, label = self.inner.score(text, speaker)
            if (self.use_floor and not isinstance(self.inner, HeuristicScorer)
                    and self._specific(clean(message_only(text)))):
                u0, label0 = self._p1.score(text, speaker)
                if u0 > u:
                    self.last_guard = "specific"
                    return u0, label0
            return u, label
        name, u, label = g
        if name == "code":
            u0, _ = self.inner.score(text, speaker)
            return max(NEUTRAL, u0), label
        if name == "small-talk":
            u0, _ = self.inner.score(text, speaker)
            return min(u, u0), label
        return u, label
