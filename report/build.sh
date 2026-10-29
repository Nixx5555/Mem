#!/bin/sh
# Build the project report: blackbook.pdf. Needs a LaTeX install
# (texlive-latex-extra, texlive-science, texlive-publishers).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode blackbook >/dev/null
bibtex blackbook >/dev/null
pdflatex -interaction=nonstopmode blackbook >/dev/null
pdflatex -interaction=nonstopmode blackbook
