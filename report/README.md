# Capstone submission documents

The project report ("black book"), log book and annexure forms, in the format
of the NMIMS Capstone project guidelines 2026-27 (Annexures A.1-A.8) and the
STME Indore front-matter template.

| File | What it is |
|---|---|
| `blackbook.tex`, `chapters/` | The project report. Build with `./build.sh` (pdflatex + bibtex); output `blackbook.pdf`. Figures, tables and diagrams are reused from `../paper/`. |
| `details.tex` | Names, SAP IDs, mentor and program stream. Edit here, then rebuild. |
| `make_forms.py` | Writes `MemGate_Log_Book.docx` (A.7) and `MemGate_Annexure_Forms.docx` (A.1-A.6): `python3 make_forms.py OUT_DIR`. |

Blank fields (roll numbers, department, program stream, mentor meeting dates)
are left for the team to fill in; text to check is highlighted in the Word files.
