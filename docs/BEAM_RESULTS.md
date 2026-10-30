# The learned scorer on a 4× larger dataset: adding BEAM

**Question.** PR #4 trained the learned scorer (P3) on LoCoMo + REALTALK,
20 conversations and 14,826 turns. Does it hold up on a larger, different
dataset, and does training on that data help?

**Short answer.**
* Out of the box, no. On BEAM, the scorer trained only on LoCoMo + REALTALK
  ranks evidence turns *below* chance (AUC 0.372). The P1 heuristic is also
  below chance (0.447). This is the sign flip REPORT.md §3.9 saw on
  LongMemEval, now on a second long-horizon dataset.
* With BEAM in training, the scorer works on BEAM conversations it never saw:
  AUC 0.66 to 0.68, and at 20% retention it roughly doubles answer recall over
  the heuristic (+38 to +41 points, significant).
* Adding BEAM as-is hurt LoCoMo (−3.2 points, significant), because BEAM is
  75% of the training turns. Weighting each dataset equally removes that cost
  (−0.1, n.s.) and keeps most of the BEAM gain.
* The shipped scorer is still PR #4's. The balanced all-data model fails the
  app's own sanity test: on short chat messages it scores "Hi!" (0.68) above
  "My flight to Delhi is on 14 December at 6:40 am." (0.12). The benchmarks
  don't contain messages like that; the app does.

## What was added

* **BEAM** (Tavakoli et al., 2025; github.com/mohammadtavakoli78/BEAM, MIT,
  commit b2da22e). Long synthetic user–assistant chats with probing questions
  that cite source turns by id. Loader: `data.load_beam`.
  * 100K scale: 20 chats, 5,732 turns, 355 questions with evidence, about
    130k tokens per chat.
  * 500K scale: 35 chats, 38,058 turns, 629 questions, about 500k tokens per
    chat. Used for training only.
  * 1M scale skipped: 1,720 duplicate turn ids make its evidence ambiguous.
    10M not downloaded.
  * Dropped: abstention questions (no evidence by design, 116 in all).
    Two-sided evidence (`{"original_info": [...], "updated_info": [...]}`) keeps
    both sides.
  * Answers are free text, so only questions whose reference words all appear
    in their evidence turns get an answer-recall score: 34 of 355 at 100K.
    AUC uses every turn.
* **Total: 75 conversations, 58,616 turns**, versus 14,826 before.
* **Still not used: LongMemEval.** huggingface.co and Google Drive are blocked
  here. GitHub copies we found have only a 20-item pilot without turn-level
  labels.

## Setup

Same as docs/SCORER_TRANSFER.md: P1-S select, context 2048, PR #2's scorer,
logistic regression on 15 hand features + MiniLM, clustered bootstrap over
conversations (10,000 resamples). New protocols in `train_transfer.py`:

* *pooled*: every other conversation of all four datasets.
* *pooled-prev*: every other LoCoMo + REALTALK conversation (what PR #4 shipped).
* *cross*: other datasets only. BEAM 100K and 500K count as one family, so
  BEAM's cross model never sees BEAM.
* `--balance`: each dataset family (LoCoMo, REALTALK, BEAM) carries equal
  total weight in the fit.

The scored conversation never contributes a turn or a label to its own model
(asserted).

## Results (hand + MiniLM, answer recall %)

S = 4096 is 25% of a LoCoMo conversation, 21% of REALTALK, 3% of BEAM 100K.
S = 26,000 is 20% of BEAM 100K, the retention LoCoMo is tested at.

| Test | Scorer | AUC | as-is | balanced |
|---|---|---|---|---|
| LoCoMo, S=4096 (715) | P1 heuristic | 0.599 | 37.5 | 37.5 |
| | pooled-prev (PR #4) | 0.791 | 52.0 | 51.2 |
| | pooled, all four | 0.754 / 0.794 | 48.8 | 51.0 |
| REALTALK, S=4096 (160) | P1 heuristic | 0.624 | 32.5 | 32.5 |
| | pooled-prev | 0.749 | 36.2 | 36.2 |
| | pooled, all four | 0.736 / 0.750 | 43.1 | 40.0 |
| BEAM 100K, S=4096 (34) | P1 heuristic | 0.447 | 11.8 | 11.8 |
| | pooled-prev | 0.372 | 8.8 | 8.8 |
| | pooled, all four | 0.675 / 0.659 | 32.4 | 20.6 |
| BEAM 100K, S=26000 (34) | RAG, FIFO | – | 32.4 | 32.4 |
| | P1 heuristic | 0.447 | 35.3 | 35.3 |
| | pooled-prev | 0.372 | 38.2 | 35.3 |
| | pooled, all four | 0.675 / 0.659 | **76.5** | **73.5** |

AUC for "pooled, all four" is as-is / balanced.

Paired differences, answer recall (points, 95% CI):

| Test | Comparison | as-is | balanced |
|---|---|---|---|
| LoCoMo | pooled − P1 | +11.33 [+6.66, +15.04] SIG | +13.57 [+8.55, +18.34] SIG |
| LoCoMo | pooled − pooled-prev | −3.22 [−6.50, −0.16] SIG | −0.14 [−2.96, +2.56] n.s. |
| REALTALK | pooled − P1 | +10.62 [−1.35, +17.59] n.s. | +7.50 [−3.19, +13.81] n.s. |
| REALTALK | pooled − pooled-prev | +6.88 [−3.47, +14.49] n.s. | +3.75 [−3.60, +9.38] n.s. |
| BEAM, S=4096 | pooled − P1 | +20.59 [+3.23, +35.72] SIG | +8.82 [−6.45, +22.86] n.s. |
| BEAM, S=4096 | pooled − pooled-prev | +23.53 [+10.26, +37.50] SIG | +11.76 [0.00, +24.24] n.s. |
| BEAM, S=26000 | pooled − P1 | +41.18 [+20.51, +63.64] SIG | +38.24 [+15.15, +63.33] SIG |
| BEAM, S=26000 | pooled − pooled-prev | +38.24 [+18.42, +60.00] SIG | +38.24 [+15.62, +62.96] SIG |

Notes on reading this:
* Balancing also reweights LoCoMo against REALTALK inside pooled-prev, which
  is why that row moves from 52.0 to 51.2.
* At S = 4096 a BEAM chat keeps 3% of its tokens, so every scorer is near the
  floor. The balanced model's BEAM gain there is no longer significant.
* BEAM's 34 answerable questions make every BEAM interval wide. The AUC
  change (0.372 → 0.66) uses all 5,732 turns and points the same way.

**Hand features alone.** Pooling BEAM in without MiniLM pulls LoCoMo AUC down
to 0.607, and the BEAM cross model is 0.357. As in PR #4, the embedding is
what transfers. Full tables: the `transfer_locomo_realtalk_beam_beam500_*`
CSVs in `memgate/results/` (`_balanced` for the weighted runs).

## Shipped scorer: unchanged, and why

`export_scorer.py --with-beam --balance` builds the balanced all-data model.
On the benchmarks it is the best of the three: it matches PR #4's scorer on
LoCoMo (−0.14, n.s.), is ahead on REALTALK (+3.75, n.s.), and is the only one
that works on BEAM's long chats (+38 at S = 26,000, SIG).

It is not shipped, because it fails `test_scorer.py`'s check that every fact
outscores every greeting (scores as the user speaker):

| Message | PR #4 scorer | balanced all-data |
|---|---|---|
| My flight to Delhi is on 14 December at 6:40 am. | 0.72 | 0.12 |
| My budget for the laptop is 85,000 rupees. | 0.66 | 0.39 |
| Actually the review moved to 8 November. | 0.91 | 0.52 |
| My sister Priya lives in Pune and works at Infosys. | 0.80 | 0.87 |
| Hi! | 0.48 | 0.68 |
| lol yes | 0.15 | 0.46 |
| ok thanks | 0.01 | 0.06 |

BEAM turns average 452 tokens, and 77% of its evidence turns are long user
messages. Short, fact-dense sentences, which are what people type
into the app, barely occur in it, so the model learned "long and descriptive"
rather than "has a date, amount or name". The benchmark average hides this,
and the app's hand-written test catches it. The library default is still P1
either way, so REPORT.md numbers are unaffected.

## Reproducing

```
# data: git clone https://github.com/mohammadtavakoli78/BEAM; for each chat
#   chats/<scale>/<n>/ copy chat.json and probing_questions/probing_questions.json
#   into memgate/data/beam/<scale>/<n>/ (scale = 100K, 500K)
python3 train_transfer.py --corpora locomo,realtalk,beam,beam500 \
    --test-corpora locomo,realtalk,beam --store-budget 4096 [--balance]
python3 train_transfer.py --corpora locomo,realtalk,beam,beam500 \
    --test-corpora beam --store-budget 26000 [--balance]
```

## Limits

* BEAM is LLM-generated, like LoCoMo. REALTALK is still the only human-written
  set, with 160 answerable questions.
* 34 answerable BEAM questions at 100K.
* The scorer needs examples from a new kind of conversation; trained without
  them it can rank evidence below chance.
* LongMemEval still untested with the pooled scorer.
