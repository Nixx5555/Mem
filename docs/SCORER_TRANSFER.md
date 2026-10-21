# The learned scorer on a larger dataset: LoCoMo + REALTALK

**Question.** §39 left one track open: a scorer that transfers across
benchmarks. The learned scorer (P3) had only ever been fitted on nine LoCoMo
conversations and tested on the tenth. Does more training data, from a
different corpus, make it better? And does it transfer to a corpus it never saw?

**Short answer.** More data helps on LoCoMo: the scorer fitted on all other
19 conversations beats one fitted on the other 9 LoCoMo conversations by
+3.4 points of answer recall, and beats the P1 heuristic by +14.6. Both are
significant. On REALTALK, nothing is significant. The learned scorer is ahead
of P1 there (+3.8 pooled, +6.3 in-corpus), but REALTALK has only 160
answer-recoverable questions, so the intervals are wide. Transfer in the
strict sense works one way only. A scorer fitted only on REALTALK beats P1 on
LoCoMo by +9.1. One fitted only on LoCoMo does not beat P1 on REALTALK
(−4.4, n.s.). So the scorer that "transfers" is the pooled one. It's shipped
as `memgate/memgate/scorer_pooled_lr.json` and switched on with
`MEMGATE_SCORER=learned`.

## What was added

* **REALTALK** (Lee et al., 2025; github.com/danny911kr/REALTALK): 10 chats
  between real people over 21 days, annotated in LoCoMo's schema. That's
  8,944 turns and 660 questions with turn-level evidence, after 66 questions
  with unresolvable evidence were dropped and 41 hand-typed evidence lists
  were normalised (ranges like `D12:3-D12:5`, `;`-joined ids). LoCoMo is
  LLM-generated and REALTALK is typed by people, which makes REALTALK the test
  of whether the scorer learned what matters or what LoCoMo looks like.
  Loader: `data.load_realtalk`.
* **LongMemEval was not used.** huggingface.co and Google Drive are blocked
  from the machine these runs used. `train_transfer.py --corpora
  locomo,realtalk,longmemeval` runs it wherever `longmemeval_s_cleaned.json`
  is present.
* **Three training sets per held-out conversation** (`train_transfer.py`):
  *in-corpus* (other conversations of the same corpus, the old protocol),
  *cross* (the other corpus only), and *pooled* (every other conversation).
  The scored conversation never contributes a turn or a label to its own
  model. This is asserted, not assumed.

## Setup

Context 2048, store S = 4096 (24.6% of a LoCoMo conversation, 20.7% of a
REALTALK one), policy P1-S select. Only the scorer varies. Embedder:
all-MiniLM-L6-v2 through onnxruntime, with tiktoken cl100k_base. As a check,
`train_scorer.py` with this setup reproduces the committed
`learned_scorer.csv`: the hand row exactly, and the MiniLM row to within one
question. Scorer code: PR #2's (speaker tag stripped). 95% intervals come from
a bootstrap that resamples whole conversations.

## Results (hand features + MiniLM, S = 4096)

Answer recall in %. AUC is how well the score separates evidence turns from
the rest on unseen conversations.

| Test corpus | Scorer | AUC | strict | answer |
|---|---|---|---|---|
| LoCoMo (715 answerable) | RAG, FIFO at same S | – | 17.8 | 28.1 |
| | P1 heuristic | 0.599 | 20.2 | 37.5 |
| | P3 in-corpus (old protocol) | 0.793 | 32.9 | 48.7 |
| | P3 cross (REALTALK only) | 0.692 | 30.1 | 46.6 |
| | **P3 pooled** | 0.791 | **34.0** | **52.0** |
| REALTALK (160 answerable) | RAG, FIFO at same S | – | 10.5 | 16.9 |
| | P1 heuristic | 0.624 | 12.6 | 32.5 |
| | P3 in-corpus | 0.737 | 18.5 | 38.8 |
| | P3 cross (LoCoMo only) | 0.681 | 11.2 | 28.1 |
| | **P3 pooled** | 0.749 | 17.3 | 36.2 |

Paired differences in answer recall (points, 95% CI):

| Test corpus | Comparison | Δ | CI | |
|---|---|---|---|---|
| LoCoMo | pooled − P1 | +14.55 | [+10.40, +19.70] | SIG |
| LoCoMo | pooled − in-corpus | +3.36 | [+0.27, +6.52] | SIG |
| LoCoMo | cross − P1 | +9.09 | [+4.53, +13.46] | SIG |
| REALTALK | pooled − P1 | +3.75 | [−2.92, +8.65] | n.s. |
| REALTALK | in-corpus − P1 | +6.25 | [−3.01, +13.42] | n.s. |
| REALTALK | cross − P1 | −4.38 | [−12.50, 0.00] | n.s. |
| REALTALK | cross − in-corpus | −10.62 | [−18.31, −3.45] | SIG |

**Hand features alone don't transfer.** Without MiniLM, the learned scorer
is significantly *worse* than P1 on REALTALK when pooled (−8.75 [−17.45,
−2.27]) or cross (−10.00). Its hand-feature weights are LoCoMo's style
(longer turns, sentence-final periods) and REALTALK's people don't write that
way. This is the same failure §30 found on LongMemEval. The embedding is what
carries across.

Full tables: `results/transfer_locomo_realtalk_S4096.csv` and
`transfer_sig_…`.

## The scorer fix, measured both ways

The same runs with the scorer as it was before PR #2 (`_oldscorer` CSVs) and
with PR #2 plus whole-word months (`_tagmonthfix`, from `--scorer-fix`):

| Test corpus, answer % | P1 old → PR #2 → +months | P3 pooled old → PR #2 → +months |
|---|---|---|
| LoCoMo, S=4096 | 36.8 → 37.5 → 37.6 | 52.6 → 52.0 → 47.7 |
| REALTALK, S=4096 | 30.6 → 32.5 → 29.4 | 35.0 → 36.2 → 31.2 |

The tag fix helps P1's AUC on both corpora (0.579 → 0.599 on LoCoMo,
0.587 → 0.624 on REALTALK). Its recall changes are within noise. Tightening
the month pattern on top of it makes recall worse. This agrees with PR #2's
choice to leave `MONTH_PAT` loose. S = 2048 tables are in the CSVs and say the
same thing at lower levels.

## Which model: logistic regression, boosted trees or a small MLP

Same folds and labels, intrinsic AUC / AP (`compare_scorers.py`, run with
`--scorer-fix`):

| Features | Test | Protocol | LR | Boosted trees | MLP (64) |
|---|---|---|---|---|---|
| hand | LoCoMo | in-corpus | **0.751** / 0.501 | 0.725 / 0.454 | 0.744 / 0.491 |
| hand | LoCoMo | cross | 0.688 / 0.462 | 0.649 / 0.379 | **0.691** / 0.445 |
| hand | LoCoMo | pooled | 0.723 / 0.481 | 0.733 / 0.465 | **0.736** / 0.482 |
| hand | REALTALK | in-corpus | **0.679** / 0.289 | 0.645 / 0.264 | 0.674 / 0.287 |
| hand | REALTALK | cross | 0.663 / 0.264 | 0.642 / 0.250 | **0.664** / 0.278 |
| hand | REALTALK | pooled | **0.679** / 0.279 | 0.676 / 0.278 | 0.675 / 0.291 |
| hand+MiniLM | LoCoMo | in-corpus | 0.792 / 0.550 | **0.809** / 0.561 | 0.799 / 0.555 |
| hand+MiniLM | LoCoMo | cross | 0.708 / 0.459 | **0.735** / 0.476 | 0.705 / 0.459 |
| hand+MiniLM | LoCoMo | pooled | 0.785 / 0.538 | **0.787** / 0.528 | 0.780 / 0.537 |
| hand+MiniLM | REALTALK | in-corpus | **0.725** / 0.318 | not run | 0.711 / 0.313 |
| hand+MiniLM | REALTALK | cross | 0.682 / 0.279 | not run | **0.690** / 0.279 |
| hand+MiniLM | REALTALK | pooled | **0.736** / 0.321 | not run | 0.722 / 0.316 |

AUC / AP, best AUC in bold. LR and MLP rows: `results/scorer_models_locomo_realtalk_tagmonthfix.csv`.
The boosted-tree column is from a run that was stopped before its last three
cells: each of its fits took minutes on this machine. Its own LR column matched
the run above exactly, so the folds are the same.

Speed, per turn on the write path: logistic regression and the MLP take
about 0.25 ms. Boosted trees took 3–250 ms here and their fits took minutes,
but the machine was shared with other runs, so read that as "much slower",
not as a benchmark.

**Choice: logistic regression stays.** Boosted trees win only on LoCoMo with
MiniLM (+0.02 AUC in-corpus, +0.03 cross). They tie pooled and lose with hand
features, at far higher cost. The MLP never beats logistic regression by more
than 0.013 AUC. Every one of these gaps is no larger than the spread between conversations
(AUC standard deviation 0.027 to 0.061 across held-out conversations).
A stronger embedder (bge-small, e5) or a cross-encoder could not be
tried, because their weights are only on HuggingFace and Google Storage, both
blocked here.

## Using it

This is still the shipped scorer. A model that adds BEAM was tried and not
shipped; see `docs/BEAM_RESULTS.md`.

```
python3 export_scorer.py              # refit on all 20 conversations -> memgate/scorer_pooled_lr.json
MEMGATE_SCORER=learned python3 ...    # MemGatePolicy() then uses it; unset = P1
```

The library default stays P1, so every experiment script still means what
REPORT.md says. Apps opt in with `MEMGATE_SCORER=learned`. The model is plain
JSON (scaler, 399 coefficients, intercept), so loading it runs no code. It
falls back to P1, with a warning, if MiniLM isn't the embedder. The table
above is the honest estimate of how it does on a new conversation: the
shipped file is fitted on all 20, so testing it on any of them would be
in-sample.

## Limits

* Two corpora and twenty conversations. REALTALK's 160 answerable questions
  leave every REALTALK difference except one non-significant.
* Labels are still "cited as evidence by some question". That's distant
  supervision, and it is denser on LoCoMo (24% of turns) than on REALTALK
  (12%).
* LongMemEval, the corpus where §30 saw the sign flip, is still untested
  with the pooled scorer.
