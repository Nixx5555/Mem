# Any text in

The experiments feed MemGate dialogue turns: one or two sentences of English,
the shape LoCoMo and REALTALK are made of. A person using the app can type or
upload anything. `memgate/memgate/ingest.py` sits between that input and
`policy.observe()`, so the memory layer only ever receives input in the shape
it was tested on. The benchmark scripts don't use it, and the live check still
matches (RAG 28.1%, P1-S 37.5% at S = 4096).

## What the app accepts

| Input | What happens |
|---|---|
| An ordinary message | Same as before. A message of up to 120 tokens is stored whole. |
| A long paste (up to 200,000 characters) | Cut into pieces of at most 48 tokens, at paragraph, sentence, line and word boundaries. Each piece is scored, stored and recalled on its own. The model receives the recalled notes plus the paste's questions (or its first and last lines), not the whole paste. |
| A file (＋ button, drag and drop, or paste) | Text is extracted and stored the same way. Each piece is tagged `[file name.pdf]`, so a recalled note shows where it came from. The upload reports how many pieces the store holds within S. |
| Other languages, emoji, code, odd characters | Control characters, zero-width marks and bidi overrides are removed, and text is NFC-normalised. Scoring is guarded (see below). |

Supported files: PDF (needs `pypdf`), Word `.docx` and PowerPoint `.pptx`
(read with the standard library), and HTML, Markdown, CSV/TSV, JSON, YAML,
XML, subtitles, notebooks and source code. Plain text is accepted in UTF-8,
UTF-16 or Windows-1252. The limits are 15 MB per file and 400,000 characters
of text. Some inputs are refused with a reason the person can act on:
images, binaries, old `.doc` and `.xls` files, scanned PDFs with no text
layer, and password-protected PDFs.

## Why chunk

A 5,000-token paste stored as one item is useless to recall for three reasons:

1. It can't fit in the 2,048-token context.
2. MiniLM reads only the first 256 word pieces, so the embedding describes just the opening.
3. The learned scorer's length features would be far outside its training
   range. LoCoMo turns average 21 words. A paragraph repeated 200 times scored 1.00.

The piece size was chosen by measurement. With REPORT.md uploaded as a file,
the share of pieces the learned scorer puts above 0.9 depends on piece size:

| Piece size | Share above 0.9 |
|---|---|
| 80 tokens (125 pieces) | 46% |
| 48 tokens (207 pieces, the one used) | 36% |
| 40 tokens (255 pieces) | 32% |

When most pieces score near 1.0, eviction has little to choose between, so
smaller pieces give the scorer more to work with.

## Scoring input the scorer was never trained on

Both scorers were built on English dialogue. Without guards they get these inputs wrong:

| Input | Without the guard |
|---|---|
| "Hi!" | 0.48 with the learned scorer (the BEAM-trained model gave it 0.68, above a flight date) |
| Hindi or Chinese sentences | `utils.words` only sees `[A-Za-z0-9']`, so P1 treats them as small talk (0.02) |
| Python code | 0.15, so P1 forgets it at once |
| Repeated text | 1.00 |

`GuardedScorer` wraps the app's scorer. Each guard covers one kind of input:

| Input | Utility |
|---|---|
| No letters or digits (emoji, punctuation) | 0.02, filler |
| Greeting, thanks, "lol", or 2 words or fewer with no digit | at most 0.05, filler |
| Code, markup, SQL | at least 0.50 |
| Mostly non-Latin script, 3 words or more | 0.50, neutral |
| The same few words over and over | at most 0.15 |

"Neutral" is the honest answer for text the scorer can't read. Such text is
kept while there's room. It is forgotten after facts the scorer could judge,
and before small talk.

**Measured on LoCoMo, P1-S at S = 4096 with the learned scorer:**

- The guards fire on 23 of 5,882 turns (22 small talk, 1 emoji-only).
- Answer recall is 53.29% with or without them (381 of 715), the same in every conversation.

On the input the scorer was trained for, the guarded scorer is the same scorer.

## One fix in the library

`MemGatePolicy.observe` used to score an evicted item using the speaker of
the turn that pushed it out, not the item's own speaker. With alternating
speakers and an even Tier 1 these are always the same, so no committed number
moves; the live check still matches. A long paste or a file breaks the
alternation, so the item now carries its own `speaker`.

## Tests

`memgate/test_ingest.py` has 82 checks, and `run_tests.py` includes it. It covers:

- cleaning and chunking in seven scripts and shapes
- every file type, built in the test itself (no fixtures)
- refusals
- the guards on both scorers
- the speaker fix
- the app end to end: a .docx upload whose fact is recalled by a question, and a long paste whose middle fact is recalled for its closing question and again in a later message
- reload rebuilding the same memory
