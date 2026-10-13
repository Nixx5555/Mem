# Scorer fix: old vs new results

**The bug.** The router scored the stored string `"[Speaker] message"`, not
the message (`policies.py:73`, scored at `policies.py:413`). The tag stopped
the filler rule from matching. On LoCoMo it also gave every turn the
proper-noun bonus through the capitalised speaker name, so 5,863 of 5,882
turns were labelled "fact", and the speaker "[Maria]" set off the month rule.
`scoring.message_only()` now strips the tag before the heuristic, the hand
features and the learned scorer's labels look at the text. Embeddings and the
LLM judge still see the full string, so their caches are unchanged.

**How these numbers were made.** The embedder was all-MiniLM-L6-v2 run through
onnxruntime (Chroma's ONNX export of the same weights; the CSVs label it
`all-MiniLM-L6-v2/384d (onnx)`), with tiktoken cl100k_base. As a check, the
unfixed code reproduces the committed `run_live_check.py` numbers exactly
(RAG 28.1%, P1-S 36.8%, both MATCH). Tests: 71/71 pass.

## Headline: answer recall on LoCoMo (storage sweep, context 2048)

Each cell is old → new, in %.

| Policy | S=512 | S=1024 | S=2048 | S=4096 | S=8192 | S=16384 | no cap |
|---|---|---|---|---|---|---|---|
| RAG store-all | 4.9 → 4.9 | 9.1 → 9.1 | 18.3 → 18.3 | 28.1 → 28.1 | 44.3 → 44.3 | 66.4 → 66.4 | 74.8 → 74.8 |
| P1 MemGate | 5.9 → 5.9 | 11.5 → 11.3 | 18.0 → 18.3 | 20.3 → 24.3 | 20.6 → 25.7 | 20.6 → 25.7 | 20.6 → 19.2 |
| P1-S MemGate select | 7.4 → 7.7 | 15.7 → 15.8 | 26.3 → 27.8 | **36.8 → 37.5** | 48.2 → 57.8 | 72.3 → 71.3 | 75.2 → 75.2 |
| P1-A MemGate adaptive | 5.5 → 5.2 | 10.9 → 11.9 | 23.4 → 22.7 | 30.1 → 30.3 | 37.5 → 41.0 | 66.4 → 63.1 | 75.2 → 75.2 |

## Significance at S = 4096 (`significance.csv`)

| Comparison | Metric | Old Δ (pts) | New Δ (pts) | Old McNemar p | New McNemar p |
|---|---|---|---|---|---|
| P1-S vs RAG | answer | +8.7 | **+9.4** | 1.1e-4 | 2.8e-5 |
| P1-S vs RAG | strict | +2.0 | +2.4 | 0.16 | 0.08 (still not significant) |
| P1-S vs P1-A | answer | +6.7 | +7.1 | 3.6e-5 | 2.1e-5 |
| P1-S vs P0 | answer | +18.5 | +19.2 | | |

## The scorer comparison changes most (`learned_scorer.csv`, S = 2048)

| | Old | New |
|---|---|---|
| P1 heuristic, answer recall | 26.3 | 27.8 |
| P3-learned, hand features | 27.7 | 27.8 |
| P3-learned, hand + MiniLM | 31.5 | 30.5 |
| Learned over heuristic | +5.2 | **+2.7** |
| AUC hand / hand + MiniLM | 0.746 / 0.793 | 0.747 / 0.793 |

Once the heuristic gets the clean message, a learned scorer on the same hand
features no longer beats it (27.8 vs 27.8), and the MiniLM version's lead
roughly halves. Part of the earlier "scorer headroom" was the bug handicapping P1.

## Other tables

* **P1 (threshold) on `locomo_results.csv`**: answer recall falls slightly,
  25.0 → 23.4 at 4096 and 20.6 → 19.2 at 2048. I haven't traced the cause; a
  likely one is the filler rule now dropping more short turns.
* **Scoring-signal ablations** (`ablations_2048.csv`): removing "proper" or
  "temporal" used to help (+0.46, +0.39 strict pts). Now every signal's removal
  hurts or is neutral, which is the expected direction.
* **P4 online vs P1** (`online.csv`): still not significant (answer −2.9,
  CI −7.9 to +1.1).
* **Synthetic set** (`step0_results.csv`): P1 at 800 tokens 0.55 → 0.475,
  at 3200 0.60 → 0.675.
* **Not re-run here**: `endtask.csv`, `abstractive.csv`, `judge_eval.csv` and
  `longmemeval*.csv` need the local Qwen models, the P2 judge cache or the
  LongMemEval file, none of which this machine can reach. Rerun them on the
  laptop.

Figures (`results/figures/`) and `frontend/data.js` are regenerated from the
new CSVs. REPORT.md, the paper and the deck still quote the old numbers.
