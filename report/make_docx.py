"""Build the black book as an editable Word file.

    cd report && ./build.sh && python3 make_docx.py OUT.docx

The LaTeX build must have run first: chapter, figure, table and equation
numbers, cross-references and citation numbers are read from blackbook.aux,
and the reference list from blackbook.bbl. Text, lists, tables and equations
go through pandoc and stay editable. TikZ diagrams, PDF plots and the
pseudocode algorithms are rendered to PNG. The title page, certificate,
contents, list of figures and list of tables are written in Word directly;
the three lists are Word fields that fill in when Word updates fields on open.
"""
import copy
import os
import re
import subprocess
import sys
import tempfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = ["ch1_introduction", "ch2_literature", "ch3_methodology",
            "ch4_results", "ch5_advantages", "ch6_conclusion"]
TEXTWIDTH_IN = 6.27   # A4 with 1.5 in + 1 in side margins


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def braced(tex, i):
    """Return (content, index after) for the {...} group starting at tex[i]."""
    assert tex[i] == "{", tex[i:i + 40]
    depth = 0
    for j in range(i, len(tex)):
        if tex[j] == "{" and tex[j - 1] != "\\":
            depth += 1
        elif tex[j] == "}" and tex[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def command_arg(tex, name):
    """Content of the first \\name{...} (optional [..] skipped), or None."""
    m = re.search(r"\\" + name + r"\*?(\[[^\]]*\])?\s*\{", tex)
    if not m:
        return None
    return braced(tex, m.end() - 1)[0]


def remove_command(tex, name):
    while True:
        m = re.search(r"\\" + name + r"\*?(\[[^\]]*\])?\s*\{", tex)
        if not m:
            return tex
        _, end = braced(tex, m.end() - 1)
        tex = tex[:m.start()] + tex[end:]


# ------------------------------------------------------------- aux data
def load_aux():
    aux = read(os.path.join(HERE, "blackbook.aux"))
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux):
        labels[m.group(1)] = m.group(2)
    cites = {}
    for m in re.finditer(r"\\bibcite\{([^}]*)\}\{([^}]*)\}", aux):
        cites[m.group(1)] = int(m.group(2))
    return labels, cites


def details():
    macros = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$",
                         read(os.path.join(HERE, "details.tex")), re.M):
        macros[m.group(1)] = m.group(2)
    return macros


def expand_details(tex, macros):
    for name, val in sorted(macros.items(), key=lambda kv: -len(kv[0])):
        if name == "blankfield":
            continue
        tex = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda _: val, tex)
    return tex


# ----------------------------------------------------------- rendering
PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[paperwidth=30in,paperheight=40in,margin=0.2in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{graphicx,booktabs,adjustbox,array,tabularx,multirow,amsmath,amssymb}
\usepackage{algorithm,algpseudocode,xcolor,tikz,caption}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit}
\DeclareCaptionFont{tenpt}{\fontsize{10}{12}\selectfont}
\captionsetup{font=tenpt,labelfont={tenpt,bf},labelsep=period}
\input{details}
\newcommand{\colfig}{0.8\textwidth}
\pagestyle{empty}
\setlength{\textheight}{9.69in}
\begin{document}
\input{../paper/diagrams/styles}
"""


class Renderer:
    def __init__(self, workdir):
        self.workdir = workdir
        self.n = 0

    def tex(self, body, width_in=TEXTWIDTH_IN, rotate=False):
        """Typeset body in a box width_in wide; return a trimmed PNG path."""
        self.n += 1
        name = "render%02d" % self.n
        src = (PREAMBLE + r"\noindent\begin{minipage}{%.2fin}" % width_in
               + "\n" + body + "\n" + r"\end{minipage}" + "\n\\end{document}\n")
        with open(os.path.join(HERE, name + ".tex"), "w") as f:
            f.write(src)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            name + ".tex"], cwd=HERE, capture_output=True,
                           text=True)
        pdf = os.path.join(HERE, name + ".pdf")
        if r.returncode != 0 or not os.path.exists(pdf):
            sys.stderr.write(r.stdout[-3000:])
            raise RuntimeError("render failed: " + name)
        out = self.pdf(pdf, rotate=rotate, dpi=220)
        for ext in (".tex", ".pdf", ".aux", ".log"):
            p = os.path.join(HERE, name + ext)
            if os.path.exists(p):
                os.remove(p)
        return out

    def pdf(self, pdf, rotate=False, dpi=200):
        self.n += 1
        stem = os.path.join(self.workdir, "img%02d" % self.n)
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile",
                        "-f", "1", "-l", "1", pdf, stem], check=True)
        png = stem + ".png"
        im = Image.open(png).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        from PIL import ImageChops
        box = ImageChops.difference(im, bg).getbbox()
        if box:
            pad = 12
            box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
                   min(box[2] + pad, im.width), min(box[3] + pad, im.height))
            im = im.crop(box)
        if rotate:
            im = im.rotate(90, expand=True)
        im.save(png)
        return png


# --------------------------------------------------------- conversion
class Converter:
    def __init__(self, workdir):
        self.labels, self.cites = load_aux()
        self.macros = details()
        self.render = Renderer(workdir)
        self.chapter = 0
        self.appendix = False
        self.counts = {}
        self.algorithms = 0

    # numbering mirrors the report class: per chapter, starred not counted
    def chap_tag(self):
        return chr(64 + self.chapter) if self.appendix else str(self.chapter)

    def next(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return "%s.%d" % (self.chap_tag(), self.counts[kind])

    def check(self, label, number):
        if label and label in self.labels and self.labels[label] != number:
            raise RuntimeError("numbering drift at %s: aux %s, mine %s"
                               % (label, self.labels[label], number))

    def inline_inputs(self, tex):
        def repl(m):
            path = m.group(1)
            if "paper/diagrams" in path:
                return m.group(0)   # rendered with its figure
            if not path.endswith(".tex"):
                path += ".tex"
            return strip_comments(read(os.path.join(HERE, path)))
        return re.sub(r"\\input\{([^}]*)\}", repl, tex)

    def refs(self, tex):
        def ref(m):
            return self.labels.get(m.group(1), "??")
        tex = re.sub(r"\\eqref\{([^}]*)\}",
                     lambda m: "(" + ref(m) + ")", tex)
        tex = re.sub(r"\\(?:auto)?ref\{([^}]*)\}", ref, tex)

        def cite(m):
            nums = sorted(self.cites[k.strip()] for k in m.group(1).split(","))
            return "[" + ", ".join(str(n) for n in nums) + "]"
        return re.sub(r"\\cite\{([^}]*)\}", cite, tex)

    def envs(self, tex, names):
        """Yield (start, end, name, body) for top-level float environments."""
        pat = re.compile(r"\\begin\{(" + "|".join(names) + r")\}(\[[^\]]*\])?")
        pos = 0
        while True:
            m = pat.search(tex, pos)
            if not m:
                return
            name = m.group(1)
            end_tag = "\\end{%s}" % name
            e = tex.index(end_tag, m.end())
            yield m.start(), e + len(end_tag), name, tex[m.end():e]
            pos = e + len(end_tag)

    def caption_of(self, body):
        m = re.search(r"\\caption(\[[^\]]*\])?\s*\{", body)
        if not m:
            return None, body
        text, end = braced(body, m.end() - 1)
        return text, body[:m.start()] + body[end:]

    def figure(self, name, body):
        label = command_arg(body, "label")
        body = remove_command(body, "label")
        caption, body = self.caption_of(body)
        if name == "algorithm":
            self.algorithms += 1
            num = str(self.algorithms)
            self.check(label, num)
            alg = (r"\renewcommand{\thealgorithm}{%s}" % num
                   + r"\begin{algorithm}[H]\caption{%s}" % caption
                   + body + r"\end{algorithm}")
            png = self.render.tex(self.refs(alg))
            return "\n\n\\begin{center}\\includegraphics[width=%.2fin]{%s}\\end{center}\n\n" % (
                TEXTWIDTH_IN * 0.95, png)
        num = self.next("figure")
        self.check(label, num)
        inner = body.replace("\\centering", "").strip()
        g = re.fullmatch(r"\\includegraphics(\[([^\]]*)\])?\{([^}]*)\}", inner)
        if g:
            path = os.path.normpath(os.path.join(HERE, g.group(3)))
            if path.endswith(".pdf"):
                path = self.render.pdf(path)
            m = re.search(r"width=([\d.]+)\\textwidth", g.group(2) or "")
            frac = float(m.group(1)) if m else 0.8
        else:
            rotate = name == "sidewaysfigure"
            width = 9.69 if rotate else TEXTWIDTH_IN
            path = self.render.tex(self.refs(inner), width_in=width, rotate=rotate)
            frac = 1.0 if rotate else 0.9
        return ("\n\n\\begin{figure}\\includegraphics[width=%.2fin]{%s}"
                "\\caption{Figure %s. %s}\\end{figure}\n\n"
                % (TEXTWIDTH_IN * frac, path, num, caption))

    def table(self, body):
        label = command_arg(body, "label")
        body = remove_command(body, "label")
        caption, body = self.caption_of(body)
        num = self.next("table")
        self.check(label, num)
        body = self.clean_tabular(body)
        return ("\n\n\\begin{table}\\caption{Table %s. %s}\n%s\n\\end{table}\n\n"
                % (num, caption, body))

    def longtable(self, body):
        spec, rest = braced(body, body.index("{"))[0], braced(body, body.index("{"))[1]
        rows = body[rest:]
        caption, rows = self.caption_of(rows)
        label = command_arg(rows, "label")
        rows = remove_command(rows, "label")
        if "\\endfirsthead" in rows:
            a = rows.index("\\endfirsthead")
            b = rows.index("\\endhead")
            rows = rows[:a] + rows[b + len("\\endhead"):]
        rows = re.sub(r"\\end(head|foot|lastfoot)", "", rows)
        rows = rows.lstrip()
        if rows.startswith("\\\\"):
            rows = rows[2:]
        tab = "\\begin{tabular}{%s}\n%s\n\\end{tabular}" % (spec, rows)
        if caption is None:
            return "\n\n" + self.clean_tabular(tab) + "\n\n"
        num = self.next("table")
        self.check(label, num)
        return ("\n\n\\begin{table}\\caption{Table %s. %s}\n%s\n\\end{table}\n\n"
                % (num, caption, self.clean_tabular(tab)))

    def clean_tabular(self, body):
        body = re.sub(r"\\begin\{adjustbox\}\{[^}]*\}|\\end\{adjustbox\}", "", body)
        body = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont", "", body)
        body = re.sub(r"\\renewcommand\{\\arraystretch\}\{[^}]*\}", "", body)
        body = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", body)
        body = re.sub(r"\\(centering|small|footnotesize|scriptsize)\b", "", body)
        body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
        body = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", body)
        body = body.replace("@{}", "")
        body = re.sub(r"\{\\boldmath\s*(\$[^$]*\$)\}", r"\\textbf{\1}", body)

        def tabx(m):
            spec = m.group(2).replace("X", "l")
            spec = re.sub(r">\{[^}]*\}", "", spec)
            return "\\begin{tabular}{%s}" % spec
        body = re.sub(r"\\begin\{tabularx\}\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", tabx, body)
        body = body.replace("\\end{tabularx}", "\\end{tabular}")
        body = re.sub(r">\{\\raggedright\\arraybackslash\}", "", body)
        return body

    def equation(self, body):
        label = command_arg(body, "label")
        body = remove_command(body, "label")
        num = self.next("equation")
        self.check(label, num)
        return "\n\\[ %s \\qquad (%s) \\]\n" % (body.strip(), num)

    def headings(self, tex):
        out = []
        pat = re.compile(r"\\(chapter|section|subsection)(\*)?(\[[^\]]*\])?\s*\{")
        pos = 0
        for m in pat.finditer(tex):
            if m.start() < pos:
                continue
            title, end = braced(tex, m.end() - 1)
            out.append(tex[pos:m.start()])
            kind, star = m.group(1), m.group(2)
            if kind == "chapter":
                if star:
                    out.append("\\chapter*{%s}" % title)
                else:
                    self.chapter += 1
                    self.counts = {}
                    word = "Appendix" if self.appendix else "Chapter"
                    out.append("\\chapter*{%s %s: %s}" % (word, self.chap_tag(), title))
            elif star:
                out.append("\\%s*{%s}" % (kind, title))
            elif kind == "section":
                n = self.counts["section"] = self.counts.get("section", 0) + 1
                self.counts["subsection"] = 0
                out.append("\\section*{%s.%d %s}" % (self.chap_tag(), n, title))
            else:
                n = self.counts["subsection"] = self.counts.get("subsection", 0) + 1
                out.append("\\subsection*{%s.%d.%d %s}" % (
                    self.chap_tag(), self.counts.get("section", 0), n, title))
            pos = end
        out.append(tex[pos:])
        return "".join(out)

    def process(self, tex):
        """Number headings and floats in document order, then rewrite."""
        parts = []
        pat = re.compile(r"\\(chapter|section|subsection)(\*)?(\[[^\]]*\])?\s*\{"
                         r"|\\begin\{(figure|sidewaysfigure|algorithm|table|equation|longtable)\}"
                         r"(\[[^\]]*\])?")
        pos = 0
        while True:
            m = pat.search(tex, pos)
            if not m:
                parts.append(tex[pos:])
                break
            parts.append(tex[pos:m.start()])
            if m.group(1):
                _, end = braced(tex, m.end() - 1)
                parts.append(self.headings(tex[m.start():end]))
                pos = end
                continue
            name = m.group(4)
            end_tag = "\\end{%s}" % name
            e = tex.index(end_tag, m.end())
            body = tex[m.end():e]
            pos = e + len(end_tag)
            if name == "longtable":
                parts.append(self.longtable(tex[m.end():e] if False else body))
            elif name == "table":
                parts.append(self.table(body))
            elif name == "equation":
                parts.append(self.equation(body))
            else:
                parts.append(self.figure(name, body))
        return "".join(parts)

    def tidy(self, tex):
        tex = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", tex)
        tex = re.sub(r"\\(clearpage|newpage|phantomsection|noindent|centering)\b", "", tex)
        tex = re.sub(r"\\setstretch\{[^}]*\}", "", tex)
        tex = re.sub(r"\\vspace\*?\{[^}]*\}", "", tex)
        tex = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont", "", tex)
        tex = re.sub(r"\\rule\{[^}]*\}\{[^}]*\}", "\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_", tex)
        tex = re.sub(r"\\blankfield\{[^}]*\}", "\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_", tex)
        tex = tex.replace("\\begin{center}", "").replace("\\end{center}", "")
        return tex

    def bibliography(self):
        bbl = read(os.path.join(HERE, "blackbook.bbl"))
        items = re.split(r"\\bibitem\{[^}]*\}", bbl)[1:]
        out = ["\\chapter*{References}"]
        for i, item in enumerate(items, 1):
            item = item.replace("\\end{thebibliography}", "")
            item = re.sub(r"\\BIBentry(ALT|STD)interwordspacing", "", item)
            item = re.sub(r"\\newblock\s*", " ", item)
            item = re.sub(r"\\hskip\s*[\d.]+em(\s*plus\s*[\d.]+em)?(\s*minus\s*[\d.]+em)?(\\relax)?", " ", item)
            item = re.sub(r"\\BIBforeignlanguage\{[^}]*\}", "", item)
            item = re.sub(r"\\url\{([^}]*)\}", r"\\texttt{\1}", item)
            item = re.sub(r"\\(href)\{[^}]*\}", "", item)
            item = re.sub(r"\s+", " ", item).strip()
            out.append("[%d] %s\n" % (i, item))
        return "\n\n".join(out)

    def appendix_d_tail(self, tex):
        # the boxed placeholder page becomes a plain marker on its own page
        i = tex.find("\\vspace*{\\fill}")
        if i == -1:
            return tex
        return tex[:i] + "\n\nPAGEBREAKMARK\n\n[Insert the printed Turnitin similarity and AI-writing report here.]\n"

    def body_tex(self):
        pieces = ["\\chapter*{Acknowledgement}\n" + self.acknowledgement(),
                  "\\chapter*{Abstract}\n" + strip_comments(read(os.path.join(HERE, "chapters/abstract.tex"))),
                  "TOCMARK",
                  strip_comments(read(os.path.join(HERE, "chapters/abbreviations.tex")))]
        main = [strip_comments(read(os.path.join(HERE, "chapters/%s.tex" % c)))
                for c in CHAPTERS]
        app = self.appendix_d_tail(
            strip_comments(read(os.path.join(HERE, "chapters/appendices.tex"))))
        tex = "\n\n".join(pieces)
        tex = self.inline_inputs(tex)
        tex = self.process(tex)
        main_tex = self.process(self.inline_inputs("\n\n".join(main)))
        self.appendix = True
        self.chapter = 0
        app_tex = self.process(self.inline_inputs(app))
        full = tex + "\n\n" + main_tex + "\n\n" + self.bibliography() + "\n\n" + app_tex
        full = self.refs(full)
        full = expand_details(full, self.macros)
        full = self.tidy(full)
        full = re.sub(r"\\text\{\(([^{}]*)\)\}", r"(\\text{\1})", full)
        return ("\\documentclass{report}\n\\usepackage{amsmath}\n"
                "\\begin{document}\n" + full + "\n\\end{document}\n")

    def acknowledgement(self):
        bb = strip_comments(read(os.path.join(HERE, "blackbook.tex")))
        s = bb.index("\\chapter*{Acknowledgement}")
        e = bb.index("\\chapter*{Abstract}")
        ack = bb[s + len("\\chapter*{Acknowledgement}"):e]
        ack = re.sub(r"\\begin\{flushright\}(.*?)\\end\{flushright\}",
                     lambda m: "\n\nRIGHTMARK " + m.group(1).replace("\\\\", "\n\nRIGHTMARK "),
                     ack, flags=re.S)
        return ack


# ------------------------------------------------------------- styling
def set_font(style, size=None, bold=None, italic=None):
    style.font.name = "Times New Roman"
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(a)) is not None:
            del fonts.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    style.font.color.rgb = None
    color = rpr.find(qn("w:color"))
    if color is not None:
        rpr.remove(color)


def style(doc, name):
    for s in doc.styles:
        if s.name == name:
            return s
    raise KeyError(name)


def make_reference(path):
    with open(path, "wb") as f:
        f.write(subprocess.run(["pandoc", "--print-default-data-file",
                                "reference.docx"], capture_output=True, check=True).stdout)
    doc = Document(path)
    st = doc.styles
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
        try:
            s = st[name]
        except KeyError:
            continue
        set_font(s, 12)
        pf = s.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)
        if name != "Compact":
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size, align in (("Heading 1", 18, WD_ALIGN_PARAGRAPH.CENTER),
                              ("Heading 2", 14, None), ("Heading 3", 12, None)):
        s = style(doc, name)
        set_font(s, size, bold=True, italic=False)
        pf = s.paragraph_format
        pf.space_before = Pt(18 if name != "Heading 1" else 0)
        pf.space_after = Pt(12 if name == "Heading 1" else 6)
        pf.line_spacing = 1.15
        pf.keep_with_next = True
        if align is not None:
            pf.alignment = align
        if name == "Heading 1":
            pf.page_break_before = True
    s = style(doc, "TOC Heading")
    set_font(s, 18, bold=True, italic=False)
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.page_break_before = True
    s.paragraph_format.space_before = Pt(0)
    s.paragraph_format.space_after = Pt(12)
    for name in ("Image Caption", "Table Caption", "Caption"):
        try:
            s = st[name]
        except KeyError:
            continue
        set_font(s, 10, italic=False)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.line_spacing = 1.0
        s.paragraph_format.space_after = Pt(10)
    for name in ("Table", ):
        pass
    try:
        set_font(st["Source Code"], 10)
    except KeyError:
        pass
    for name in ("Hyperlink", "Verbatim Char"):
        try:
            set_font(st[name])
        except KeyError:
            pass
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin, sec.right_margin = Inches(1.5), Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(1)
    doc.save(path)


# ------------------------------------------------------- post-process
def field_paragraph(doc, instr, placeholder):
    p = doc.add_paragraph()
    def run_with(el):
        r = OxmlElement("w:r")
        r.append(el)
        p._p.append(r)
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); b.set(qn("w:dirty"), "true")
    run_with(b)
    t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = instr
    run_with(t)
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate")
    run_with(s)
    r = OxmlElement("w:r"); tt = OxmlElement("w:t"); tt.text = placeholder; r.append(tt); p._p.append(r)
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    run_with(e)
    return p


def page_break_para(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def centered(doc, text, size=12, bold=False, before=0, after=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    return p


def left(doc, runs, before=0, after=0, align=WD_ALIGN_PARAGRAPH.LEFT):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.5
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
    return p


def front_matter(doc, m):
    """Title page and certificate, appended at the end; moved to the front later."""
    logo = os.path.join(HERE, "images/nmims_logo.png")
    before = list(doc.element.body)
    centered(doc, m["projtitle"], 18, True, before=10, after=26)
    centered(doc, "Project Report submitted in partial fulfilment", 14, True)
    centered(doc, "of", 14, True, after=10)
    centered(doc, "Bachelor of Technology (B.Tech) in", 14)
    centered(doc, m["stream"].replace("\\&", "&"), 14, after=20)
    centered(doc, "by", 14, True, after=10)
    centered(doc, "%s (SAP ID: %s)" % (m["studentA"], m["sapA"]), 14, True)
    centered(doc, "%s (SAP ID: %s)" % (m["studentB"], m["sapB"]), 14, True, after=20)
    centered(doc, "Under the supervision of", 14, True, after=10)
    centered(doc, m["mentor"].replace("\\ ", " "), 14, True)
    centered(doc, "(%s, STME, Indore Campus)" % m["mentordesig"], 12, after=40)
    centered(doc, "SVKM's NMIMS University", 14, True)
    centered(doc, "(Deemed-to-be University)", 12, after=8)
    p = centered(doc, "", after=8)
    p.add_run().add_picture(logo, width=Cm(2.8))
    centered(doc, "SCHOOL OF TECHNOLOGY MANAGEMENT &", 14, True)
    centered(doc, "ENGINEERING (STME)", 14, True, after=8)
    centered(doc, m["acadyear"].replace("--", "–"), 12, True)
    page_break_para(doc)

    centered(doc, "CERTIFICATE", 18, True, after=14)
    p = centered(doc, "", after=8)
    p.add_run().add_picture(logo, width=Cm(3))
    centered(doc, "SVKM's NMIMS University", 14, True)
    centered(doc, "(Deemed-to-be University)", 14, True, after=14)
    left(doc, [("This is to certify that the project entitled ", False),
               ("“%s”" % m["projtitle"], True),
               (" has been done by ", False), (m["studentA"], True),
               (" (SAP: %s) and " % m["sapA"], False), (m["studentB"], True),
               (" (SAP: %s) under my guidance and supervision & has been submitted "
                "in partial fulfilment of the degree of " % m["sapB"], False),
               ("B.Tech", True), (" in ", False),
               (m["streamshort"].replace("\\&", "&"), True),
               (" of STME, SVKM's NMIMS (Deemed-to-be University), Indore Campus, "
                "India.", False)], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    for name, role in ((m["mentor"].replace("\\ ", " "), "(Internal Guide)"),
                       ("Examiner", "(External)"),
                       (m["assocdean"].replace("\\ ", " "), "(Associate Dean, STME)")):
        left(doc, [("_" * 28, False)], before=50)
        left(doc, [(name, True)])
        left(doc, [(role, False)])
    p = left(doc, [("Date: ", True), ("_" * 16, False), ("\t\t\t\t\t\t", False),
                   ("Place: ", True), ("Indore", False)], before=40)
    page_break_para(doc)
    return new_elements(doc, before)


def new_elements(doc, before):
    return [e for e in doc.element.body
            if not any(e is b for b in before) and e.tag != qn("w:sectPr")]


def contents_block(doc):
    before = list(doc.element.body)
    items = []
    for title, instr in (("Contents", 'TOC \\o "1-3" \\h \\z \\u'),
                         ("List of Figures", 'TOC \\h \\z \\t "Image Caption,1"'),
                         ("List of Tables", 'TOC \\h \\z \\t "Table Caption,1"')):
        h = doc.add_paragraph(title, style=style(doc, "TOC Heading"))
        field_paragraph(doc, instr, "Right-click here and choose Update Field "
                        "(Word fills this in when it updates fields on opening).")
    return new_elements(doc, before)


def move_before(elements, anchor):
    for el in elements:
        anchor.addprevious(el)


def add_page_number(paragraph):
    run = paragraph.add_run()
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); run._r.append(b)
    run2 = paragraph.add_run()
    t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = "PAGE"; run2._r.append(t)
    run3 = paragraph.add_run()
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); run3._r.append(e)


def header_footer(section, m, fmt=None, start=None):
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    hp = section.header.paragraphs[0]
    hp.text = ""
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(5.77), 2)  # right
    r = hp.add_run(m["projtitleshort"] + "\tAcademic Year " + m["acadyear"].replace("--", "–"))
    r.font.size = Pt(10)
    pPr = hp._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "000000")):
        bottom.set(qn(k), v)
    bdr.append(bottom)
    pPr.append(bdr)
    fp = section.footer.paragraphs[0]
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_page_number(fp)
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sectPr.append(pg)
    if fmt:
        pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def section_break_before(paragraph_el):
    """Insert a next-page section break that ends just before paragraph_el."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    sect = OxmlElement("w:sectPr")
    pPr.append(sect)
    p.append(pPr)
    paragraph_el.addprevious(p)
    return sect


def postprocess(path, m):
    doc = Document(path)
    body = doc.element.body
    paras = {id(p._p): p for p in doc.paragraphs}

    # markers left by the converter
    toc_anchor = None
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if t == "TOCMARK":
            toc_anchor = p._p
        elif t == "PAGEBREAKMARK":
            p.text = ""
            p.add_run().add_break(WD_BREAK.PAGE)
        elif t.startswith("RIGHTMARK"):
            text = t[len("RIGHTMARK"):].strip()
            for r in p.runs:
                r.text = ""
            p.runs[0].text = text if p.runs else None
            if not p.runs:
                p.add_run(text)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.bold = True

    # figures and tables keep their caption on the same page
    for p in doc.paragraphs:
        if p.style.name in ("Captioned Figure", "Table Caption"):
            p.paragraph_format.keep_with_next = True
        if p.style.name == "Captioned Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    first = body[0]
    fm = front_matter(doc, m)
    move_before(fm, first)
    cb = contents_block(doc)
    move_before(cb, toc_anchor)
    toc_anchor.getparent().remove(toc_anchor)

    # sections: front pages (no numbers), roman preliminaries, arabic chapters
    heads = [p for p in doc.paragraphs if p.style.name == "Heading 1"]
    ack = next(p for p in heads if p.text.strip() == "Acknowledgement")
    ch1 = next(p for p in heads if p.text.startswith("Chapter 1"))
    s1 = section_break_before(ack._p)
    s2 = section_break_before(ch1._p)
    final = doc.sections[-1]._sectPr
    for s in (s1, s2):
        for child in final:
            if child.tag in (qn("w:pgSz"), qn("w:pgMar")):
                s.append(copy.deepcopy(child))
    doc.save(path)
    doc = Document(path)
    secs = doc.sections
    secs[0].header.is_linked_to_previous = False
    secs[0].footer.is_linked_to_previous = False
    for part in (secs[0].header, secs[0].footer):
        for p in part.paragraphs:
            p.text = ""
    header_footer(secs[1], m, fmt="lowerRoman", start=1)
    header_footer(secs[2], m, fmt="decimal", start=1)
    for s in secs[1:]:
        s.different_first_page_header_footer = False

    # a justified line that ends in a manual line break stretches; set it left
    for p in doc.paragraphs:
        brs = p._p.findall(".//" + qn("w:br"))
        if any(b.get(qn("w:type")) != "page" for b in brs):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # tables: full grid, 10 pt, centred, spread across the text width
    for t in doc.tables:
        t.alignment = 1
        tblPr = t._tbl.tblPr
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            tblPr.append(tblW)
        tblW.set(qn("w:type"), "pct")
        tblW.set(qn("w:w"), "5000")
        layout = tblPr.find(qn("w:tblLayout"))
        if layout is not None:
            tblPr.remove(layout)
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement("w:" + side)
            for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "000000")):
                el.set(qn(k), v)
            borders.append(el)
        old = tblPr.find(qn("w:tblBorders"))
        if old is not None:
            tblPr.remove(old)
        tblPr.append(borders)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    p.alignment = None
                    for r in p.runs:
                        r.font.size = Pt(10)

    # ask Word to refresh the contents and lists when the file opens
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)
    doc.core_properties.title = m["projtitle"]
    doc.core_properties.author = m["studentA"] + ", " + m["studentB"]
    doc.save(path)


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "MemGate_Black_Book.docx")
    work = tempfile.mkdtemp(prefix="bbdocx")
    conv = Converter(work)
    tex = conv.body_tex()
    src = os.path.join(work, "body.tex")
    with open(src, "w") as f:
        f.write(tex)
    ref = os.path.join(work, "reference.docx")
    make_reference(ref)
    r = subprocess.run(["pandoc", src, "-f", "latex", "-t", "docx",
                        "--reference-doc", ref, "-o", out],
                       cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    m = {k: v for k, v in conv.macros.items()}
    postprocess(out, m)
    print("wrote", out, "| work dir", work)


if __name__ == "__main__":
    main()
