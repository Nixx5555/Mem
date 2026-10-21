# Viva prep: likely questions and the answers the results support

Every number below comes from a committed CSV or REPORT.md section. The
file is named next to each answer so you can open it if asked. "SIG" means
the 95% interval (cluster bootstrap over conversations, 10,000 resamples)
excludes zero; "n.s." means it doesn't.

## The project in one breath

MemGate sits in front of a frozen LLM. Each turn it decides what to keep,
compress, forget and recall, under a token budget. The main finding:
**choosing which turns to forget beats forgetting oldest-first by +9.4
answer-recall points at the same storage** (S = 4096, LoCoMo, CI [+5.4,
+14.9], McNemar p = 2.8×10⁻⁵). **Compressing an evicted turn instead of
dropping it loses 7.1 points.** (REPORT.md abstract, §3)

---

## A. Data and scale

**Q: You only used 10 LoCoMo conversations. Isn't that too small?**
That was true for the main study, so the scorer was then trained and tested
on four datasets:

| Dataset | Chats | Turns | Questions with evidence | Written by |
|---|---|---|---|---|
| LoCoMo | 10 | 5,882 | 1,527 | LLM |
| REALTALK | 10 | 8,944 | 660 | real people |
| BEAM 100K | 20 | 5,732 | 355 | LLM, 130k tokens per chat |
| BEAM 500K | 35 | 38,058 | 629 | LLM, 500k tokens per chat |

That's 58,616 turns, 4× the 14,826 of LoCoMo + REALTALK. (docs/BEAM_RESULTS.md)

**Q: Did you test on anything bigger than LoCoMo?**
Yes, twice. The memory policy was run on LongMemEval: 94 haystacks of 38–62
sessions each. There the headline does **not** hold: at matched retention,
scored selection loses to oldest-first by 20.97 points, because the
hand-written scorer ranks LongMemEval's evidence below the rest (REPORT.md
§3.9, deck slide 18). The learned scorer was then trained and tested on BEAM
(below), which shows the same failure out of the box and the fix: train on
some conversations of the new kind. LongMemEval itself wasn't used for
*scorer training*, only because its file is on HuggingFace and Google Drive,
which the training machine couldn't reach. `train_transfer.py --corpora
locomo,realtalk,longmemeval` runs it wherever the file exists.

**Q: How did you get labels for training the scorer?**
A turn is labelled positive if any benchmark question cites it as evidence.
That's distant supervision, not a human judging "is this worth remembering".
The positive rate differs a lot by dataset: 24% of LoCoMo turns, 12% of
REALTALK, 16% of BEAM 100K, 6% of BEAM 500K.

**Q: Could the scorer be cheating by seeing the answer?**
No, and we checked. Features come only from the turn's text and speaker. The
conversation being scored never contributes a turn or a label to its own
model, and the code asserts this for every fold. We guard this because an
early consolidation rule did read the answer key (an evaluation `fact_id`),
and it was worth +45 to +47 points. A test now strips every label from the
input and checks the store comes out bit-identical (REPORT.md §4.1).

---

## B. The learned scorer

**Q: What model is the scorer?**
Logistic regression (balanced class weights) on 15 hand-made features
(length, numbers, dates, named entities, question marks, etc.) plus the
384-d MiniLM sentence embedding. It's shipped as plain JSON
(`memgate/memgate/scorer_pooled_lr.json`) and turned on with
`MEMGATE_SCORER=learned`.

**Q: Why logistic regression and not something stronger?**
We tested boosted trees and a 64-unit MLP on the same folds. Neither beat
LR by more than the spread between conversations (best gap +0.03 AUC, while
AUC varies 0.03 to 0.06 across held-out conversations). Boosted trees cost
3 to 250 ms per turn versus 0.25 ms for LR, and the scorer runs on every
turn. (docs/SCORER_TRANSFER.md, "Which model")

**Q: Does the learned scorer actually help?**
On LoCoMo, yes, a lot. Trained on the other 19 LoCoMo + REALTALK
conversations, it reaches 52.0% answer recall versus 37.5% for the
heuristic: +14.55 [+10.40, +19.70] SIG. On REALTALK at that budget it's
ahead (+3.75) but not significant, because REALTALK has only 160 answerable
questions; with a larger store (S = 8,192) it's +16.88 SIG.
(docs/SCORER_TRANSFER.md)

**Q: Did you pick S = 4096 because it makes the learned scorer look best?**
S = 4096 was the headline budget months before the learned scorer existed.
We then ran the scorer at five budgets. On LoCoMo it's significantly ahead
at 2,048 (+5.87), 4,096 (+14.55), 8,192 (+11.33) and 16,384 (+5.87); only at
1,024, about 5% of a conversation, is it not significant (+3.22). On REALTALK
it becomes significant at 8,192 (+16.88) and 16,384 (+6.88). The gain is
largest in the middle: with too little room no ranking helps, and with room
for nearly everything there's little to choose. (paper Table VIII,
docs/SCORER_TRANSFER.md "Across the store budget")

**Q: What does the embedding add?**
Without MiniLM, the learned scorer is significantly worse than the heuristic
on REALTALK (−8.75 [−17.45, −2.27]). Its hand-feature weights learn LoCoMo's
writing style, which real people don't share. The embedding is what carries
across datasets.

---

## C. The larger-dataset result (BEAM)

**Q: Does the scorer work on a dataset it has never seen?**
No, and this is the most important honest answer. On BEAM, the scorer
trained only on LoCoMo + REALTALK has AUC 0.372. That's *below chance*: it
ranks the evidence turns lower than the rest. The heuristic is also below
chance on BEAM (0.447). This is the same sign flip REPORT.md §3.9 found on
LongMemEval, now reproduced on a second long-horizon dataset.

**Q: So what fixes it?**
Training on some BEAM conversations. With BEAM in the training pool (the
tested conversation is still held out), AUC on BEAM rises to 0.675, and at
20% retention (S = 26,000 tokens of a 130k-token chat):

| Scorer on BEAM 100K (34 answerable) | Answer recall |
|---|---|
| RAG, oldest-first eviction | 32.4% |
| Heuristic (P1) | 35.3% |
| Learned, trained without BEAM | 38.2% |
| **Learned, trained with BEAM** | **76.5%** |

Learned with BEAM minus heuristic: +41.18 [+20.51, +63.64] SIG. With BEAM
minus without BEAM: +38.24 [+18.42, +60.00] SIG. At the tighter S = 4096
(3% retention) it's 32.4% vs 11.8%, +20.59 SIG.
(results/transfer_*_test-beam_S26000.csv)

**Q: Only 34 questions? Is that enough?**
It's the main weakness of the BEAM result. BEAM's answers are free text, and
we only score a question when the reference answer's words appear in the
evidence turns, so 34 of 355 qualify. The intervals are wide ([+20, +64]),
but they're clustered by conversation and still exclude zero. AUC uses every
turn, not just these 34 questions, and tells the same story (0.372 → 0.675).

**Q: Did adding BEAM hurt anything?**
Yes, when BEAM was added without weighting. BEAM is 75% of all training
turns, so it pulled the model towards BEAM's style. On LoCoMo, answer recall
fell from 52.0% to 48.8% (−3.22 [−6.50, −0.16] SIG). It was still +11.33
over the heuristic. Weighting each dataset equally in training fixes it:
LoCoMo goes back to 51.0% against 51.2% without BEAM (−0.14 [−2.96,
+2.56] n.s.), REALTALK is 40.0% against 36.2% (n.s.), and BEAM at 20%
retention is still 73.5% against 35.3% for the heuristic (+38.24 [+15.15,
+63.33] SIG). The cost: at 3% retention the BEAM gain shrinks to +8.8 n.s.

**Q: Which scorer ships, then?**
Since 21 October, the balanced all-data model, with one addition. PR #8's
version failed the app's sanity test: it scored "Hi!" (0.68) above "My
flight to Delhi is on 14 December at 6:40 am." (0.12). BEAM has almost no
small talk, so nothing taught it that a greeting is worth nothing. The
shipped model adds 923 generated greetings, thanks and reactions as
negatives at 5% weight. They are generated, not taken from any test chat.
"Hi!" now scores 0.01, and every test passes.

Held out 5-fold by conversation, it beats the old shipped model:

- BEAM 100K: +41.2 points, significant.
- BEAM 500K: +38.2 points, significant. The old model was below plain retrieval there.
- LoCoMo: +0.7, not significant.
- REALTALK: +6.2, not significant.

With a 20k-token document uploaded into the middle of a LoCoMo chat, the old
model's recall of the chat fell to 10.2% and the new one keeps 43.6%. The
library default is still the heuristic, so every REPORT.md number still
means what it says. (docs/STRESS_TEST.md §6)

---

## D. Method questions

**Q: Why answer recall and not LLM accuracy?**
Answer recall asks whether the reference answer's words survive in what the
memory hands the reader. It isolates the memory decision from the reader's
skill. We also measured through a fixed local reader: +6.0 of the +8.7 point
gap (69%) survives as correct answers, against a 1.7% closed-book floor
(REPORT.md §3.8, run before the scorer fix).

**Q: Why does compression lose?**
A compressed turn costs tokens but often loses the exact name, date or
number the question needs. Dropping a low-value turn and keeping a
high-value one verbatim uses the same budget better. Rate–distortion theory
predicts compression should win under enough pressure; we pushed to 182×
compression and never saw the crossover. Selection stays ahead of RAG at
every ratio: the gap narrows in points as everything nears the floor (+9.5 to
+4.1) but grows as a ratio (1.5× to 1.9× RAG's recall) (REPORT.md §3.4). A better compressor helps but still loses to
dropping (§3.10).

**Q: Why charge for the embedding index?**
Real systems store the vectors. Once the index counts against the budget,
below about 40 KiB a policy with no index beats every retrieval method. At
32 KiB the no-index window gets 18.3% against 15.1% for P1-S; at 64 KiB P1-S
is ahead, 24.9% (REPORT.md §3.5).

**Q: How do you know differences aren't noise?**
Every comparison is paired (same questions, same budget) and the interval
resamples whole conversations, because questions inside one conversation
aren't independent. The headline also passes McNemar (p = 2.8×10⁻⁵).

---

## E. Questions the panel is likely to push on

**Q: How does this compare to Mem0 or MemGPT/Letta?**
We didn't run their code head to head, and should say so. What we did test
is the two ideas they rely on, at the same budget as MemGate. Using an LLM to
judge what matters (as Mem0's extraction and MemGPT's self-editing do) loses
to our regex scorer by 5.87 points, interval excluding zero (REPORT.md §3.7,
pre-fix run). Compressing old turns into a gist instead of dropping them,
the idea behind MemGPT's recursive summary, loses 7.1 points to dropping
(§3.2), and an LLM that rewrites them still loses by 2.7 at 1,850× the cost
(§3.10). Our
claim is about the decision policy, which those systems leave to an LLM.

**Q: 37.5% answer recall sounds low. Why is that good?**
It's a stress setting on purpose. At S = 4096 the store keeps about 22% of a
LoCoMo conversation (4,096 of ~18,600 tokens) and the model sees 2,048 tokens
per question. Full context gets 100% by definition. The question is which 22%
to keep: forgetting oldest-first keeps the answer 28.1% of the time, MemGate
37.5%, the learned scorer 52.0%. With four times the storage (S = 16,384)
MemGate reaches 71.3% (results/storage_sweep.csv).

**Q: Aren't the scorer's labels circular? It learns from the benchmark's
own questions.**
It learns from questions about *other* conversations. A turn's label is
"some question cites it", and the conversation being scored never contributes
a label (asserted in code for every fold). The cross-corpus runs are the
stronger answer: trained only on REALTALK, it still beats the heuristic on
LoCoMo by +9.09 (SIG), so it isn't memorising LoCoMo's question style.

**Q: Why should anyone trust numbers from your own harness?**
Three bugs that changed our own results were found by invariant tests, not
by reading code, and each is reported (REPORT.md §4). The biggest, an
evaluation-label leak, was worth 45–47 points and was withdrawn at Review 2.
Every test suite runs on every push (the GitHub Actions badge on the README).

---

## F. Limits to state before they're asked

* The scorer is tuned to the data it's trained on. It needs some examples
  from a new kind of conversation; out of the box it can be worse than
  chance (BEAM, LongMemEval).
* Labels are "cited by a benchmark question", which is a proxy for
  importance.
* One reader model for the end-task numbers.
* BEAM's answer-recall numbers rest on 34 questions; REALTALK's on 160.
* End-task, LLM-judge, compression and LongMemEval sections of REPORT.md were
  not re-run after the speaker-tag fix (PR #2).
