# Scale test: ten times the training data, chats of up to 10 million tokens

Date: 21–22 October 2026. As in the stress test (docs/STRESS_TEST.md), no chat model is involved: what is checked is whether the right note reaches the model. The metrics are the paper's:

- **answer recall**: the reference answer's words are in the assembled context, over questions whose answer is in their evidence;
- **evidence recall** (soft recall): the share of a question's cited turns that reach the context. Every question with evidence has one, so on BEAM it covers about ten times as many questions as answer recall.

Every comparison has a 95% confidence interval from a bootstrap that resamples whole conversations (10,000 draws). **sig** means the interval excludes zero.

| Script | What it does |
|---|---|
| `memgate/retrain_10x.py` | Builds the 10x training set, fits the candidates 5-fold by conversation, replays every held-out conversation at several budgets, reports with CIs. Checkpointed: every replay is appended to `results/scale/replays.jsonl` as it finishes. |
| `memgate/scale_cache.py` | MiniLM embeddings for the 10x set (628,227 strings), written in shards of 20,000 so a restarted run loses at most one shard. |
| `memgate/scale_app.py` | Types whole BEAM chats into the real app one message at a time, then asks their questions; and the load tests, over HTTP. |
| `memgate/memgate/data.py` | New loaders: BEAM 1M (duplicate ids resolved), BEAM 10M, MemBench. |

## 1. The headline

**Through the real app, on chats of a million tokens, MemGate keeps the answer 52% of the time; plain retrieval keeps it 19% of the time.**

Every BEAM 1M chat with an answerable question (33 of 35, about 1.07 million tokens and 1,060 messages each) and the same 33 of 35 BEAM 500K chats were typed into `app.App` one message at a time, exactly as the page sends them, with the chat's recorded assistant turns as the replies. Store S = 26,000 tokens, the paper's BEAM budget, so MemGate holds about 2.5% of what was said. Then each question with a recoverable answer was asked as a message, and the notes each memory would hand the model were checked.

| Chats typed into the app | Questions | **MemGate** | RAG | P0 window | MemGate − RAG | MemGate − window |
|---|---|---|---|---|---|---|
| BEAM 1M (33 chats, never seen in training) | 52 | **51.9%** | 19.2% | 9.6% | **+32.7 [+19.2, +46.3]**, McNemar p = 0.0009 | **+42.3 [+28.6, +56.1]**, p = 0.00001 |
| BEAM 500K (33 chats, in the shipped scorer's training set) | 55 | **65.5%** | 30.9% | 12.7% | **+34.5 [+19.6, +49.1]**, p = 0.0003 | **+52.7 [+39.6, +65.5]**, p < 0.0001 |

The store never went over S (largest: 25,999 and 26,000 tokens). Typing a message took 0.49 s on average at 1M and 0.43 s at 500K, on a 4-core cloud machine that was running other jobs at the same time.

The shipped scorer (PR #12) was fitted on BEAM 100K and 500K, so the 500K row is in-sample. The 1M row is not: those chats are new to it.

## 2. Held-out, across seven corpora and up to 10 million tokens

Section 4's 5-fold runs also test the **shipped recipe** (PR #12's training set, refitted without the held-out fold) on conversations it never saw, at the paper's budgets. BEAM 1M and 10M are never in its training set at all.

| Corpus (S) | Conversations | Answer: RAG | P1 | **Learned (PR #12 recipe)** | Learned − RAG |
|---|---|---|---|---|---|
| LoCoMo (4,096) | 10 | 28.1 | 37.5 | **50.3** | **+22.2 [+17.5, +27.6]** |
| REALTALK (4,096) | 10 | 16.9 | 32.5 | **36.2** | **+19.4 [+7.5, +26.4]** |
| BEAM 100K (26,000) | 20 | 32.4 | 35.3 | **73.5** | **+41.2 [+14.3, +67.6]** |
| BEAM 500K (26,000) | 35 | 32.7 | 32.7 | **63.6** | **+30.9 [+16.7, +44.8]** |
| BEAM 1M (26,000) | 35 | 13.5 | 11.5 | **50.0** | **+36.5 [+21.8, +51.0]** |
| BEAM 10M (26,000) | 10 | 7.7 | 23.1 | **53.8** | **+46.2 [+10.0, +76.9]** |
| MemBench (2,048) | 874 | 76.8 | **97.5** | 95.6 | **+18.8 [+14.2, +23.7]** |

Evidence recall tells the same story with ten times the questions (BEAM 1M: 622, learned 22.2% against RAG's 0.9%, **+21.3 [+18.2, +24.4]**). RAG keeps the most recent turns, so on a 1M-token chat nearly everything it was asked about is long gone.

**Where the learned scorer is not the best:** MemBench. Its chats are short factual statements ("Team Connect has about three hundred people"), which is exactly what P1's hand-written rules look for. P1 beats the shipped scorer there by 1.9 points of answer recall (n.s.) and 4.6 points of evidence recall (**sig**), and by 18 points at S = 1,024. Section 4's 10x model, which has seen MemBench, closes that gap.

## 3. Budgets

Answer recall, held-out, PR #12 recipe:

| Corpus | Small S | Paper's S | Large S |
|---|---|---|---|
| LoCoMo (1,024 / 4,096 / 16,384) | 19.7 (RAG 9.1) | 50.3 (RAG 28.1) | 77.6 (RAG 66.4) |
| REALTALK (1,024 / 4,096 / 16,384) | 10.6 (RAG 9.4) | 36.2 (RAG 16.9) | 65.6 (RAG 43.1) |
| BEAM 1M (8,000 / 26,000 / 64,000) | 23.1 (RAG 7.7) | 50.0 (RAG 13.5) | 69.2 (RAG 19.2) |
| BEAM 10M (8,000 / 26,000 / 64,000) | 7.7 (RAG 0.0) | 53.8 (RAG 7.7) | 46.2 (RAG 15.4) |
| MemBench (512 / 1,024 / 2,048) | 66.8 (RAG 36.0) | 80.7 (RAG 56.1) | 95.6 (RAG 76.8) |

LoCoMo and REALTALK were also run at S = 2,048 and 8,192; every row is in `results/scale/summary.json`. BEAM 10M answer recall rests on 13 questions, so its row moves by 7.7 points per question.

## 4. Retraining on ten times the data

**Asked for:** train the scorer on ten times the data PR #12 used.

**Data.** PR #12 was fitted on 58,616 turns (LoCoMo, REALTALK, BEAM 100K, BEAM 500K). Everything below is reachable from the cloud machine; LongMemEval, MSC and anything on huggingface.co are blocked by its network policy.

| Added | Source | Chats | Turns | Questions with evidence |
|---|---|---|---|---|
| BEAM 1M | github.com/mohammadtavakoli78/BEAM | 35 | 74,630 | 622 |
| BEAM 10M | same | 10 | 208,696 | 128 |
| MemBench, observation (ThirdAgent) | github.com/import-myself/Membench | 3,455 | 49,504 | 3,455 |
| MemBench, participation (FirstAgent), 125 per kind | same | 875 | 202,554 | 874 |
| **Total with PR #12's set** | | | **594,000 (10.1x)** | |

Two datasets needed repair before they could be used:

- **BEAM 1M.** It had been skipped before because 1,720 turn ids repeat. The cause is that three chats restart their numbering partway (chat 26 runs 0–1,993, then starts again at 0). Each run of ids is now a segment. A question is pinned to the segment whose cited turns share the most words with the question and its answer, and only if that segment leads by a clear margin. 3 of 625 questions could not be pinned and are dropped.
- **BEAM 10M.** Every one of the ten plans numbers its own turns from 0. The questions' `plan_reference` field is free text ("Plan 2-3") and does not match the plan keys. The same pinning is used. 128 of 200 questions are pinned; 24 are abstentions with no evidence, and 48 cite ids no plan holds.

Both are pinned by `test_memgate.py`, along with MemBench's step layout.

**Candidates**, all StandardScaler + logistic regression on the 15 hand features + MiniLM, with the same 923 small-talk negatives as PR #12:

- **PR #12**: PR #12's recipe on its own corpora;
- **10x**: every corpus above, corpus families (LoCoMo, REALTALK, BEAM, MemBench) weighted equally, as in PR #12;
- **10x-eq**: the same 10x set, but every corpus also weighs the same within its family. Under plain family weighting BEAM 10M's 208,696 turns, with only 128 questions, take up 64% of the BEAM family's weight.

**Protocol.** 5 folds by conversation, stratified by corpus. Each conversation is replayed through P1-S by models that never saw it, at 3–5 budgets per corpus. The whole thing was repeated with the folds dealt from two reshuffled orders (seeds 1 and 2). That is 26,444 replays.

**Result at the paper's budgets, seed 0** (difference from PR #12, 95% CI):

| Corpus | Answer: 10x | Answer: 10x-eq | Evidence: 10x | Evidence: 10x-eq |
|---|---|---|---|---|
| LoCoMo | −0.8 [−2.6, +1.5] | +1.8 [+0.0, +4.2] | −0.4 [−2.4, +2.0] | −0.4 [−1.9, +1.9] |
| REALTALK | +2.5 [−3.4, +10.5] | +1.9 [−3.5, +7.3] | +0.1 [−2.4, +2.9] | −0.5 [−3.6, +2.5] |
| BEAM 100K | −2.9 [−9.4, +0.0] | 0.0 | −1.3 [−4.7, +2.1] | +0.6 [−1.9, +3.1] |
| BEAM 500K | −5.5 [−16.1, +5.2] | −7.3 [−17.2, +1.9] | **−5.2 [−8.1, −2.3]** | −2.3 [−4.8, +0.3] |
| BEAM 1M | +1.9 [−7.8, +12.2] | +1.9 [−4.0, +8.9] | **−5.1 [−7.3, −3.0]** | **−2.0 [−3.7, −0.4]** |
| BEAM 10M | −23.1 [−46.2, +0.0] | −7.7 [−33.3, +16.7] | **−2.3 [−5.5, −0.1]** | −1.8 [−5.2, +0.6] |
| MemBench | +1.6 [+0.0, +3.3] | +1.6 [+0.0, +3.3] | **+5.5 [+4.3, +6.8]** | **+5.4 [+4.2, +6.7]** |
| MemBench at S = 512 | **+27.0** | **+27.8** | **+43.1** | **+40.1** |

Through the app path (chunking + GuardedScorer + query extraction), 10x-eq gains on LoCoMo (answer **+2.0 [+0.2, +4.4]**) and BEAM 100K (evidence **+10.6 [+6.1, +15.0]**).

**Across the three fold shuffles**, the effects that repeat every time are:

- MemBench gains: evidence recall +5.4 to +6.2, **sig** in all three, for both models.
- BEAM 1M losses: evidence recall −5.1 to −5.6 for 10x and −2.0 to −3.1 for 10x-eq, **sig** in all three.
- BEAM 500K losses for 10x: −3.0 to −5.2, **sig** in all three. 10x-eq's are smaller and n.s.

LoCoMo and REALTALK move by 2.5 points or less either way, with signs that change between shuffles.

**Decision: PR #12's scorer stays shipped.** The rule since PR #12 has been to ship only a model that loses nowhere. Both 10x models lose significantly on BEAM 1M, the longest chats with enough questions to measure, in every shuffle. Their gain is on MemBench, the corpus that was added. MemBench is easy, short-fact chat where P1's rules already do well.

**Why more data did not help (our reading, not tested).** The scorer is a linear model on 399 features. PR #12's set probably already had enough data for it; what it lacks is capacity. The new data adds new kinds of chat, and a linear model has to trade between them. MemBench pulls the weights toward short factual statements, and BEAM's long technical answers lose ranking. The extra data is dominated by negatives: 10M adds 208,696 turns but only a few hundred evidence labels. More data would need a model with more capacity (PR #4 found boosted trees and a small MLP no better at the old size; that is worth retrying at this size), or a gate that picks the scorer by the kind of chat.

## 5. Load

Over HTTP to a real app server:

| Test | Result |
|---|---|
| 5,000 messages in one chat | 0.04 s per message for the first 100, 0.12 s for the last 100, slowest 0.18 s; the store stayed at 4,093 of S = 4,096 |
| A whole BEAM 1M chat typed into one chat over HTTP | 855 messages, 975,611 tokens; 0.29 s per message for the first 100, 0.37 s for the last 100, p95 0.47 s; store 25,991 of S = 26,000; saved chat 4.2 MB; peak memory 2.8 GB; reopened with identical state in 70 s |
| 200 visitors at once, 10 messages each | 2,000 messages in 25 s: **0 errors** (314 before the fix below), 0 leaks between visitors, p50 2.4 s, p95 2.7 s |
| 100 messages into one chat at the same moment | **100 of 100 kept** (88 before the fix), 200 transcript entries |

**Bug found and fixed.** The first run had **314 of 2,000** visitor requests refused with "connection reset", and **12 of 100** simultaneous messages into one chat never arrived. Python's `ThreadingHTTPServer` listens with an accept queue 5 deep, so a burst overflowed it before any thread could pick the requests up. `app.Server` raises the queue to 256, and `app.py` (and so the Hugging Face Space) now uses it. `test_app.py` pins it with a burst of 60 requests.

**Known slow path, not fixed.** Reopening a saved chat replays its transcript through memory. For the 1M-token chat that took 70 s with every embedding already cached; in a fresh process it would be longer, since the embeddings are not saved. Chats of normal length reopen in under a second (docs/STRESS_TEST.md §5). Saving the store's state with the chat would fix it.

## 6. Reproducing

```bash
cd memgate
git clone https://github.com/mohammadtavakoli78/BEAM /tmp/BEAM
for s in 100K 500K 1M; do for d in /tmp/BEAM/chats/$s/*/; do n=$(basename $d)
  mkdir -p data/beam/$s/$n; cp $d/chat.json data/beam/$s/$n/
  cp $d/probing_questions/probing_questions.json data/beam/$s/$n/; done; done
ln -s /tmp/BEAM/chats/10M data/beam/10M_raw
git clone https://github.com/import-myself/Membench /tmp/Membench
ln -s /tmp/Membench/MemData data/membench

python3 retrain_10x.py --embed-only     # ~2 h on 4 cores, resumable
python3 retrain_10x.py --sweep          # 5 folds, all budgets, ~3 h
python3 retrain_10x.py --seeds 0,1,2    # + two reshuffled fold orders, ~2 h
python3 retrain_10x.py --report         # tables -> results/scale/summary.json
python3 scale_app.py --part beam --scales 1M,500K --budgets 26000   # ~10 h
python3 scale_app.py --part load
```

The 10x run needs about 7 GB of memory. Two at once do not fit in 15 GB.
