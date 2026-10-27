#!/usr/bin/env python3
"""Build the capstone log book and the annexure forms as Word files.

    python make_forms.py OUT_DIR

Writes MemGate_Log_Book.docx and MemGate_Annexure_Forms.docx (formats from
the Capstone project guidelines 2026-27, Annexures A.1-A.7). Fields nobody
has given us yet (roll numbers, department, stream, mentor meeting dates)
are left blank; text that the team must check is highlighted yellow.
"""
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "images", "nmims_logo.png")

SCHOOL = "School of Technology Management & Engineering"
CAMPUS = "SVKM's NMIMS (Deemed-to-be University), Indore Campus"
TITLE = "MemGate: An Adaptive Memory Layer for Long-Running LLM Agents"
MENTOR = "Dr. Praveen Modi, Assistant Professor"
YEAR = "2026-27"
SEMESTER = "VII"
STUDENTS = [  # (SAP ID, name); roll numbers not known yet
    ("70562300097", "Yash Ramchandani"),
    ("70562200088", "Simar Singh Khanuja"),
]

# ------------------------------------------------------------------ helpers

def new_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(2)
    return doc


def para(doc_or_cell, text="", bold=False, size=None, align=None,
         italic=False, highlight=False, space_after=None):
    p = doc_or_cell.add_paragraph()
    if text:
        add_run(p, text, bold=bold, size=size, italic=italic,
                highlight=highlight)
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def add_run(p, text, bold=False, size=None, italic=False, highlight=False):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return r


def field(doc, label, value="", highlight=False, blank_cm=None):
    p = doc.add_paragraph()
    add_run(p, label + " ", bold=True)
    if value:
        add_run(p, value, highlight=highlight)
    if blank_cm:
        add_run(p, "_" * int(blank_cm * 2.2))
    p.paragraph_format.space_after = Pt(3)
    return p


def letterhead(doc, form_title, annex=None):
    if annex:
        para(doc, annex, bold=True, size=11, align="c", space_after=4)
    p = para(doc, align="c", space_after=2)
    p.add_run().add_picture(LOGO, width=Cm(2.3))
    para(doc, SCHOOL, bold=True, size=15, align="c", space_after=0)
    para(doc, CAMPUS, size=11, align="c", space_after=6)
    para(doc, form_title, bold=True, size=13, align="c", space_after=8)


def set_cell(cell, text="", bold=False, size=None, highlight=False,
             align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    lines = text.split("\n") if text else [""]
    for i, line in enumerate(lines):
        if i:
            p = cell.add_paragraph()
        if line:
            add_run(p, line, bold=bold, size=size, highlight=highlight)
        if align == "c":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)


def shade(cell, hex_fill="D9D9D9"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def row_height(row, cm):
    trPr = row._tr.get_or_add_trPr()
    h = OxmlElement("w:trHeight")
    h.set(qn("w:val"), str(int(cm * 567)))
    h.set(qn("w:hRule"), "atLeast")
    trPr.append(h)


def grid(doc, header, rows, widths=None, size=10, heights=None,
         head_fill="D9D9D9"):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        set_cell(t.rows[0].cells[j], h, bold=True, size=size, align="c")
        shade(t.rows[0].cells[j], head_fill)
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            hl = isinstance(v, tuple)
            set_cell(t.rows[i].cells[j], v[0] if hl else v, size=size,
                     highlight=hl)
        if heights:
            row_height(t.rows[i], heights)
    if widths:
        fix_widths(t, widths)
    return t


def fix_widths(t, widths_cm):
    """Pin column widths (Word and LibreOffice both honour tblGrid)."""
    t.autofit = False
    grid_el = t._tbl.tblGrid
    for gc, w in zip(grid_el.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w * 567)))
    for row in t.rows:
        for j, w in enumerate(widths_cm):
            row.cells[j].width = Cm(w)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def signature_block(doc, label, name=""):
    para(doc, space_after=10)
    p = doc.add_paragraph()
    add_run(p, label, bold=True)
    p = doc.add_paragraph()
    if name:
        add_run(p, name)
        p = doc.add_paragraph()
    add_run(p, "Signature: " + "_" * 24 + "     Date: " + "_" * 12)
    p.paragraph_format.space_after = Pt(6)


# ----------------------------------------------------------------- log book
# Week 1 is taken as the week of Monday 13 July 2026, so that Review 1
# (25 July 2026) falls in week 2 as the guidelines' timeline says. Work
# carried out comes from the dated record in the repository: SESSION_RECORD.md
# sessions, docs/REVIEW2_PROGRESS_REPORT.md and the git history. Weeks with
# no record are highlighted for the team to fill in.

NO_RECORD = ("No record of this week's work was found in the project files. "
             "Please fill in.")

WEEKS = [
    ("13/07/2026", "19/07/2026", [
        ("Literature survey of memory for LLM agents and of prompt and "
         "KV-cache compression (MemGPT, MemoryBank, Generative Agents, "
         "StreamingLLM, H2O, LLMLingua, Gist tokens, AutoCompressors, the "
         "agent-memory survey, A-MEM). MemGPT chosen as the base paper. "
         "Problem statement, objectives, scope and Gantt chart drafted for "
         "Review 1.", True)]),
    ("20/07/2026", "26/07/2026", [
        ("Review 1 presentation (25 July 2026): introduction, problem "
         "statement, three objectives (adaptive compression, hierarchical "
         "memory with recall, evaluation on long tasks), scope, proposed "
         "system and architecture, implementation workflow, tools and "
         "technology, feasibility, Gantt chart and comparison of 10 papers.",
         False)]),
    ("27/07/2026", "02/08/2026", [(NO_RECORD, True)]),
    ("03/08/2026", "09/08/2026", [(NO_RECORD, True)]),
    ("10/08/2026", "16/08/2026", [
        ("Working session 12 Aug: prepared answers to the four Review 1 panel "
         "questions (need, feasibility, novelty, end product). Moved the "
         "contribution from the three-tier architecture to the memory "
         "decision policy. Built Step 0: evaluation harness, three-tier "
         "store, policies P0 (sliding window) and P1 (heuristic scorer), a "
         "synthetic test set and 30 passing tests.", False)]),
    ("17/08/2026", "23/08/2026", [
        ("Session 17 Aug: found an evaluation leak (a consolidation rule read "
         "the answer label fact_id), worth 45-47 recall points; fixed it, "
         "added label-blindness tests and withdrew the Review 1 figure "
         "(95% became 50%). Wired the LoCoMo benchmark and the MiniLM "
         "embedder, added the answer-presence metric, fixed consolidation, "
         "and ran the 23-configuration ablation study (commit 18 Aug).",
         False)]),
    ("24/08/2026", "30/08/2026", [
        ("Session 24 Aug: imposed a storage budget on the whole store; found "
         "that scored forgetting beats oldest-first forgetting while "
         "compressing evicted turns loses. Session 27 Aug: scaling study to "
         "182x compression, byte accounting of the embedding index, "
         "clustered bootstrap and McNemar tests, learned scorer "
         "(leave-one-conversation-out), publication figures, and the P2 "
         "LLM judge built.", False)]),
    ("31/08/2026", "06/09/2026", [(NO_RECORD, True)]),
    ("07/09/2026", "13/09/2026", [
        ("P2 LLM judge measured: it loses to the heuristic by 5.87 points. "
         "End-task accuracy pilot through a reader model; models pinned "
         "offline. Review 2 deliverables (9 Sep): progress report, generated "
         "deck, results dashboard and speaker's guide. First draft of the "
         "IEEE conference paper with generated tables.", False)]),
    ("14/09/2026", "20/09/2026", [
        ("Full end-task run on all 715 answer-recoverable LoCoMo questions "
         "(17-18 Sep): +6.0 of the +8.7 recall points are realised as "
         "correct answers (69% of the gap), against a 1.7% closed-book "
         "floor.", False)]),
    ("21/09/2026", "27/09/2026", [
        ("LongMemEval run (25-26 Sep): the headline does not replicate; the "
         "cause was traced to the heuristic scorer, whose evidence "
         "separation reverses sign on that corpus. Paper compiled for the "
         "first time and cut to 6 pages.", False)]),
    ("28/09/2026", "04/10/2026", [
        ("29 Sep: recency-reservation confound ruled out; abstractive "
         "compression and online learning of the scorer run at full scale. "
         "Review 2 deck rebuilt, then put in the NMIMS format with speaker "
         "notes (1 Oct). First live demo: chat through MemGate and watch "
         "the tiers.", False),
        ("Review 2 presentation date: ____________", True)]),
    ("05/10/2026", "11/10/2026", [(NO_RECORD, True)]),
    ("12/10/2026", "18/10/2026", [
        ("13 Oct: the demo became a working chat app (saved chats, sidebar, "
         "memory panel). Found and fixed a scorer defect (speaker tag scored "
         "with the message) and re-ran the results (P1-S 36.8% to 37.5%). "
         "Learned scorer trained on LoCoMo + REALTALK and shipped; scorer "
         "tests added. 15 Oct: app uses the learned scorer; report, paper, "
         "deck and dashboard updated to the fixed numbers.", False)]),
    ("19/10/2026", "25/10/2026", [
        ("Paper completed (learned scorer across corpora, the app). Full app "
         "with one-command setup, any text input, file uploads and a "
         "Compare tab. Stress tests (six input bugs fixed), MemGPT-style "
         "and Mem0-style baselines, online deployment option, CI. 10x "
         "retrain and test on chats up to 10M tokens; BEAM 1M through the "
         "app: 51.9% vs 19.2% for RAG; load-test fix. Gemini replies with a "
         "local Qwen backup, entry gate and score tags. 25 Oct: journal "
         "version of the paper with Dr. Praveen Modi as co-author.", False)]),
    ("26/10/2026", "01/11/2026", [
        ("Journal version reformatted in the Springer Nature style; "
         "abstracts and prose rewritten; mentor's review of the paper "
         "applied; Dr. Praveen Modi made corresponding author; authors "
         "marked as equal contributors. Log book, project report (black "
         "book) and annexure forms prepared.", False)]),
    ("02/11/2026", "08/11/2026", []),
    ("09/11/2026", "15/11/2026", []),
]


def week_box(doc, n, start, end, items):
    t = doc.add_table(rows=6, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = [r.cells[0] for r in t.rows]
    set_cell(c[0], f"Week No: {n}", bold=True)
    set_cell(c[1], "Date of Reporting:")
    set_cell(c[2], f"From {start} to {end}")
    set_cell(c[3], "Work carried out", bold=True)
    shade(c[3], "EDEDED")
    c[4].text = ""
    first = True
    for text, hl in items:
        p = c[4].paragraphs[0] if first else c[4].add_paragraph()
        first = False
        add_run(p, text, size=10.5, highlight=hl)
        p.paragraph_format.space_after = Pt(2)
    row_height(t.rows[4], 4.4)
    set_cell(c[5], "Faculty Remark", bold=True)
    shade(c[5], "EDEDED")
    r = t.add_row()
    row_height(r, 1.6)
    r2 = t.add_row()
    set_cell(r2.cells[0], "Faculty Signature with date", bold=True)
    r3 = t.add_row()
    row_height(r3, 0.9)
    fix_widths(t, [16.6])


def log_book(out):
    doc = new_doc()
    letterhead(doc, "Log Book for Capstone Project", "A.7 Log Book for Capstone Project")
    field(doc, "Department:", blank_cm=14)
    p = field(doc, "Program:", "B.Tech in ")
    add_run(p, "_" * 34)
    field(doc, "Semester:", SEMESTER)
    field(doc, "Academic Year:", YEAR)
    field(doc, "TITLE OF THE PROJECT:", TITLE)
    para(doc, space_after=2)
    field(doc, "Name of the Faculty Mentor:", MENTOR)
    para(doc, space_after=4)
    para(doc, "STUDENT DETAILS", bold=True, align="c", space_after=4)
    rows = [["Student 1", STUDENTS[0][1], "", ""],
            ["Student 2", STUDENTS[1][1], "", ""],
            ["Student 3", "-", "", ""],
            ["Student 4", "-", "", ""]]
    t = grid(doc, ["", "NAME", "ROLL NO.", "CONTACT"], rows,
             widths=[2.6, 6.0, 3.4, 4.6], size=10.5, heights=0.8)
    for i in range(1, 5):
        t.rows[i].cells[0].paragraphs[0].runs[0].bold = True
    para(doc, space_after=6)
    para(doc, "Name of group members:", bold=True, space_after=4)
    grid(doc, ["Roll No.", "Name", "Sign (Students)",
               "Grades (By Faculty Mentor)"],
         [["", n, "", ""] for _, n in STUDENTS],
         widths=[3.0, 6.0, 3.6, 4.0], size=10.5, heights=1.0)
    para(doc, space_after=6)
    para(doc, "SAP IDs: " + ", ".join(f"{n} ({s})" for s, n in STUDENTS),
         size=10, italic=True)

    for i, (start, end, items) in enumerate(WEEKS):
        if i % 2 == 0:
            page_break(doc)
        else:
            para(doc, space_after=8)
        week_box(doc, i + 1, start, end, items)
    doc.save(out)


# ---------------------------------------------------------------- annexures

OBJECTIVES = (
    "1. Build an adaptive memory decision engine that scores every message "
    "and decides whether to keep it verbatim, compress it, forget it, or "
    "recall it later, so that the prompt stays within a fixed token budget.\n"
    "2. Build a three-tier memory store (recent turns, compressed gists, "
    "long-term facts with embeddings) held to a fixed storage budget, with "
    "retrieval to bring dropped information back when it is relevant.\n"
    "3. Evaluate the memory policies on long multi-session conversation "
    "benchmarks (LoCoMo and others), changing one factor at a time, and "
    "report the trade-off between memory budget and answer quality with "
    "proper statistics.")

MOTIVATION = (
    "LLM agents re-send their whole history on every turn, so cost and "
    "latency grow with the conversation and accuracy falls well before the "
    "context limit. Sliding windows forget the oldest messages whatever they "
    "contain, and summaries lose names, numbers and dates. On LongMemEval, "
    "oracle evidence gives about 92% accuracy against about 58% when the "
    "model selects its own: a 34-point gap caused by memory selection, not "
    "reasoning. Existing memory systems are evaluated end to end, so the "
    "value of each memory decision is unknown.")

OUTCOMES = (
    "A working memory layer (Python library) that sits in front of any LLM; "
    "a measurement harness and results on public benchmarks with confidence "
    "intervals; a learned scorer for message importance; a chat application "
    "that shows every keep, compress and forget decision; a research paper.")

REFERENCES = [
    'C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica and '
    'J. E. Gonzalez, "MemGPT: Towards LLMs as Operating Systems," arXiv '
    'preprint arXiv:2310.08560, 2023.',
    'P. Chhikara, D. Khant, S. Aryan, T. Singh and D. Yadav, "Mem0: Building '
    'Production-Ready AI Agents with Scalable Long-Term Memory," arXiv '
    'preprint arXiv:2504.19413, 2025.',
    'W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan and Y. Zhang, "A-MEM: Agentic '
    'Memory for LLM Agents," Advances in Neural Information Processing '
    'Systems (NeurIPS), 2025.',
    'A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri and Y. Fang, '
    '"Evaluating Very Long-Term Conversational Memory of LLM Agents," Proc. '
    'Annual Meeting of the Association for Computational Linguistics (ACL), '
    '2024.',
    'D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang and D. Yu, "LongMemEval: '
    'Benchmarking Chat Assistants on Long-Term Interactive Memory," arXiv '
    'preprint arXiv:2410.10813, 2024.',
]


def team_table(doc, sign_col="Signature"):
    grid(doc, ["S No", "SAP ID", "Roll No", "Name", sign_col],
         [["1", STUDENTS[0][0], "", STUDENTS[0][1], ""],
          ["2", STUDENTS[1][0], "", STUDENTS[1][1], ""],
          ["3", "-", "-", "-", "-"]],
         widths=[1.2, 3.0, 2.4, 5.2, 4.8], size=10.5, heights=0.8)


def dept_block(doc, semester=SEMESTER):
    field(doc, "Department:", blank_cm=14)
    p = field(doc, "Program:", "B.Tech in ")
    add_run(p, "_" * 34)
    if semester:
        field(doc, "Semester:", semester)
    else:
        field(doc, "Semester:", blank_cm=6)


def eval_sheet(doc, annex, title, criteria, total, signer_lines):
    letterhead(doc, title, annex)
    header = ["S No", "SAP ID", "Roll No", "Student Name"] + criteria + [total]
    rows = [[str(i + 1), s, "", n] + [""] * (len(criteria) + 1)
            for i, (s, n) in enumerate(STUDENTS)]
    fixed = [0.9, 2.5, 1.5, 2.4]
    rest = (16.6 - sum(fixed)) / (len(criteria) + 1)
    grid(doc, header, rows, widths=fixed + [rest] * (len(criteria) + 1),
         size=8.5 if len(criteria) > 3 else 9.5, heights=1.2)
    para(doc, "Project: " + TITLE, size=10, italic=True)
    for label, name in signer_lines:
        signature_block(doc, label, name)


def annexures(out):
    doc = new_doc()

    # A.1 Project registration form
    letterhead(doc, "Project registration form", "A.1 Project registration form")
    dept_block(doc, semester=None)
    para(doc, "Project Team details", bold=True, space_after=4)
    team_table(doc)
    para(doc, space_after=6)
    para(doc, "Three broad domains in which the project team intends to work",
         bold=True, space_after=4)
    grid(doc, ["S No", "Domain", "Project idea in brief"],
         [["1", "Generative AI: large language models and LLM agents",
           "A memory layer for long-running LLM agents that decides what to "
           "keep, compress, forget and recall, so long conversations fit a "
           "fixed token budget (MemGate)."],
          ["2", "", ""], ["3", "", ""]],
         widths=[1.2, 5.4, 10.0], size=10.5, heights=1.4)

    # A.2 Topic approval form
    page_break(doc)
    letterhead(doc, "Topic approval form", "A.2 Topic approval form")
    dept_block(doc)
    para(doc, "Project Team details", bold=True, space_after=4)
    team_table(doc)
    para(doc, space_after=4)
    para(doc, "Type of Project (Tick any one)", bold=True, space_after=4)
    grid(doc, ["Type of project", ""],
         [["Application", ""], ["Product", ""], ["Research", "✓"]],
         widths=[5.0, 3.0], size=10.5)
    para(doc, space_after=4)
    para(doc, "Project details", bold=True, space_after=4)
    t = doc.add_table(rows=5, cols=2)
    t.style = "Table Grid"
    for i, (k, v) in enumerate([
            ("Project Title", TITLE),
            ("Project Objectives", OBJECTIVES),
            ("Domain", "Generative AI / Natural Language Processing: memory "
                       "and context management for LLM agents"),
            ("Motivation", MOTIVATION),
            ("Expected outcomes", OUTCOMES)]):
        set_cell(t.rows[i].cells[0], k, bold=True, size=10.5)
        set_cell(t.rows[i].cells[1], v, size=10.5)
    fix_widths(t, [3.6, 13.0])
    para(doc, space_after=4)
    para(doc, "Latest References", bold=True, space_after=4)
    grid(doc, ["S No", "Publication (Author, “Title”, "
                       "Journal/Book/Conference, Date, Page, Volume, year)"],
         [[str(i + 1), r] for i, r in enumerate(REFERENCES)],
         widths=[1.2, 15.4], size=10)
    para(doc, space_after=4)
    t = doc.add_table(rows=5, cols=2)
    t.style = "Table Grid"
    labels = ["Status of the project after Review 1\nComments/Remarks",
              "Name and Signature of the guide",
              "Name and signature of panel members for review 1",
              "Signature of HoD",
              "If any change in project title after review 1 presentation as "
              "per comments received from panel and areas of concern to be "
              "noted here"]
    for i, k in enumerate(labels):
        set_cell(t.rows[i].cells[0], k, bold=True, size=10)
        row_height(t.rows[i], 1.6 if i != 2 else 2.2)
    set_cell(t.rows[1].cells[1], MENTOR, size=10.5)
    fix_widths(t, [6.0, 10.6])

    # A.3 Review 1
    page_break(doc)
    eval_sheet(doc, "A.3 Evaluation sheet for Review 1: Topic approval and feasibility",
               "Evaluation sheet for Review 1: Topic approval and feasibility",
               ["Identification of Problem statement, Purpose and need of "
                "the project (5 marks)",
                "Feasibility of the implementation (5 marks)"],
               "Total (10 marks)",
               [("Name & Signature of the Project mentor (with Date)", MENTOR),
                ("Name & Signature of the panel members (with Date)", ""),
                ("", ""), ("", "")])

    # A.4 Review 2
    page_break(doc)
    eval_sheet(doc, "A.4 Evaluation sheet for Review 2: Midterm review",
               "Evaluation sheet for Review 2: Midterm review",
               ["Literature Review / Market Survey (5 marks)",
                "Clarity of the problem statement (5 marks)",
                "Design / System level representation (5 marks)",
                "Implementation / Results (5 marks)",
                "Interim Report (5 marks)"],
               "Total (25 marks)",
               [("Name & Signature of the Project mentor (with Date)", MENTOR),
                ("Name & Signature of the panel members (with Date)", ""),
                ("", ""), ("", "")])

    # A.5 Review 3
    page_break(doc)
    eval_sheet(doc, "A.5 Evaluation sheet for Review 3: External Panel Member Evaluation",
               "Evaluation sheet for Review 3: Internal and External Panel "
               "Member Evaluation",
               ["Literature review and clarity of the problem statement "
                "(5 marks)",
                "Design / System level representation (5 marks)",
                "Implementation and results (5 marks)",
                "Report (5 marks)", "Presentation (5 marks)",
                "Viva (5 marks)"],
               "Total (30 marks)",
               [("Name & Signature of the Internal mentor (with Date)", MENTOR),
                ("Name & Signature of the External Panel Member (with Date)", ""),
                ("", "")])

    # A.6 Research paper
    page_break(doc)
    letterhead(doc, "Evaluation sheet for Research paper",
               "A.6 Evaluation sheet for Research paper")
    grid(doc, ["S No", "SAP ID", "Roll No", "Student Name", "Marks (10 marks)"],
         [[str(i + 1), s, "", n, ""] for i, (s, n) in enumerate(STUDENTS)],
         widths=[1.2, 3.0, 2.4, 5.6, 4.4], size=10.5, heights=1.0)
    para(doc, space_after=4)
    para(doc, "Paper: “Selection Beats Compression: Memory Decision "
              "Policies for LLM Agents Under a Bounded Store”, Yash "
              "Ramchandani, Simar Singh Khanuja and Dr. Praveen Modi.",
         size=10, italic=True)
    p = para(doc, "Status (submitted / accepted / published, with venue and "
                  "proof attached): ", size=10)
    add_run(p, "_" * 40, size=10)
    signature_block(doc, "Name & Signature of the Project mentor (with Date)",
                    MENTOR)
    doc.save(out)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out_dir, exist_ok=True)
    log_book(os.path.join(out_dir, "MemGate_Log_Book.docx"))
    annexures(os.path.join(out_dir, "MemGate_Annexure_Forms.docx"))
    print("wrote", out_dir)
