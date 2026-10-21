# Stress test: harder questions, bigger and odder input, more load

Date: 21 October 2026. Everything here runs without a chat model. What gets checked is whether the right note reaches the model, because the model's answer can never be better than that. The same metric is used throughout the paper: **answer recall**, meaning the reference answer's words are in the assembled context.

| Script | What it does |
|---|---|
| `memgate/stress_questions.py` | Answer recall by question type, hops, evidence age and question length, on LoCoMo, REALTALK and BEAM. Four set-ups are compared: RAG, P1 heuristic, the shipped learned scorer, and the app's own path (GuardedScorer + chunking + query extraction). |
| `memgate/stress_questions.py --long-questions` | Every question hidden inside 150–450 words of other chat. |
| `memgate/stress_inputs.py` | A real app server over HTTP. It covers needles planted in big documents (A), 32 odd inputs (B), and load (C). |
| `memgate/stress_upload_midchat.py` | Inserts a document halfway through each LoCoMo chat and compares the old and new scorers. |
| `memgate/retrain_scorer.py` | Retrains the scorer on all 75 conversations and tests it on held-out conversations (5-fold). |

Results are in `memgate/results/stress/`.

## 1. Bugs found and fixed

Each fix is pinned by a test in `test_ingest.py`.

| Input | Before | Now |
|---|---|---|
| A message containing half an emoji (a lone surrogate, e.g. a paste cut mid-character) | 500 error, and **every later message in that chat also failed**: the chat could no longer be saved | The half-character is replaced with U+FFFD and the chat carries on |
| A 300 KB `.docx` that unzips to 300 MB | 65 s and **3.8 GB of memory**, then refused | Refused in 0 s from the zip's own size table (64 MB cap) |
| JSON nested 100,000 levels deep | Connection dropped (RecursionError) | 400 |
| A request body of any size | Read into memory in full before being checked | Anything over the 15 MB upload limit gets a 413 before reading |
| A message of only zero-width characters | Accepted, stored as nothing | 400 "the message is empty" |
| A password-protected PDF | "could not read the PDF (FileNotDecryptedError)" | "the PDF is password-protected; …" |

The other 26 odd inputs already worked. They included:

- a 199,000-character word, 150,000 characters of Chinese, and Arabic/Hebrew with bidi overrides;
- 60,000 emoji, zalgo text and NUL bytes;
- the same sentence 5,000 times, and a 1,500-slide `.pptx`;
- 20,000 nested `<div>`s, UTF-16 and Latin-1 files;
- a binary file renamed `.txt`, and a file name containing `../` and HTML.

Each one got a 200 or a clear refusal, and the chat kept working afterwards. Peak memory over all 32 cases was 810 MB, down from 3.8 GB.

The slowest accepted input is a 150,000-character paste of Chinese: 9,110 pieces, about 30 s on 4 cores. Nearly all of that time is MiniLM embedding.

## 2. Questions buried in long messages (fixed)

Each LoCoMo/REALTALK question was asked inside a long message: 6–14 turns of someone else's chat before it, and often more chat after it, which brought its own questions.

| Answer recall | LoCoMo | REALTALK |
|---|---|---|
| Same question asked on its own | 53.1% | 47.5% |
| Long message, whole text as the query | 30.3% | 26.2% |
| Long message, `focus()` (before) | 42.1% | 42.5% |
| Long message, **`queries()` (now)** | **49.8%** | **43.8%** |

`focus()` joined the message's questions from the closing paragraph into one query. When chat followed the real question, that closing paragraph was the wrong one: the real question was in the query only 56% of the time.

`ingest.queries()` now makes every question its own query, and `ThreeTierStore.retrieve` interleaves their rankings. With one query, retrieval works exactly as before, so no experiment number moves (`test_bench.py` passes).

## 3. Harder questions (no bug; where recall is lowest)

The setting is S = 4096 with a 2048-token context. On LoCoMo and REALTALK the shipped scorer is **in-sample**, because it was fitted on every turn of both. Read its column as a ceiling. The paper's leave-one-out numbers are the honest ones.

**LoCoMo**

| Questions | RAG | P1 | Learned | App |
|---|---|---|---|---|
| All | 28.1 | 37.5 | 53.3 | 53.1 |
| Multi-hop (98) | 19.4 | 29.6 | 48.0 | 48.0 |
| Needs 3+ turns (50) | 24.0 | 30.0 | 42.0 | 42.0 |
| Evidence in first third (262) | **9.2** | 33.2 | 51.9 | 51.9 |

**REALTALK**

| Questions | RAG | P1 | Learned | App |
|---|---|---|---|---|
| All | 16.9 | 32.5 | 45.6 | 47.5 |
| Multi-hop (93) | 8.6 | 15.1 | 32.3 | 32.3 |
| Needs 3+ turns (57) | 5.3 | 10.5 | 28.1 | 24.6 |
| Evidence in first third (86) | 8.1 | 19.8 | 33.7 | 33.7 |

- **Old evidence is where MemGate wins.** For questions about the first third of a conversation, RAG falls to 8–9% because it forgets oldest-first. MemGate stays at 34–52%.
- **Questions that need three or more turns are the hardest case for every system.** On REALTALK they reach only 25–28%. All the turns have to survive, and they have to fit the context together.

## 4. Big documents with planted facts

Eight facts were planted at depths from 5% to 95% of a long document built from BEAM's long-form answers. The document was uploaded, and each fact was then asked about. The table shows facts recalled out of 8 (RAG in brackets).

| Document | S = 4,096 (default) | S = 32,768 | S = 200,000 |
|---|---|---|---|
| 5k tokens | 8 (6) | 8 (8) | 8 (8) |
| 40k tokens | 4 (0) | 8 (3) | 8 (8) |
| 60k tokens | 1 (1) | 6 (4) | 8 (8) |
| 93k tokens (upload limit) | 1 (0) | 6 (3) | 8 (8) |

The same 40k document as PDF, Word, PowerPoint and HTML recalled 4–5 of 8 at S = 4,096 and 8 of 8 at S = 200,000.

When the store is large enough, nothing is lost. When the store is far smaller than the document, MemGate keeps about twice as many planted facts as RAG. The page already tells the person how many pieces were forgotten and suggests raising Store S.

Upload time is almost all embedding: about 1 s per 1,000 tokens on a busy 4-core machine. The 93k-token file took 55 s.

## 5. Load

| Test | Result |
|---|---|
| 600 messages in one chat | 0.07 s per message for the first 50, 0.09 s for the last 50, slowest 0.28 s |
| Reopening a chat with a 93k-token file and 100 messages | 0.5 s, identical memory |
| 40 visitors at once (public mode), 5 messages each | 0 errors, 0 leaks between visitors, p95 1.25 s |
| 30 messages sent into one chat at once | 30/30 kept, 60 transcript entries (each message and its reply) |

## 6. Retraining the scorer on a larger dataset

**What prompted it.** On BEAM, which it never saw in training, the shipped scorer (LoCoMo + REALTALK, PR #4) did **worse than plain retrieval** on the 500K chats: 21.8% vs RAG's 32.7%. On 100K it was 38.2% vs 32.4%.

**What changed.** The new model is trained on all 75 conversations: LoCoMo, REALTALK, BEAM 100K and BEAM 500K, 58,600 turns. Corpus families are weighted equally. It also adds 923 generated small-talk lines ("Hi!", "thanks!", "ok cool"), each with and without a speaker tag, as negatives at 5% weight.

PR #8's BEAM model was not shipped because it scored "Hi!" above a flight date. BEAM contains almost no small talk, so nothing taught it otherwise. The small-talk negatives are generated, not taken from any test conversation, so they leak nothing.


**Held-out test.** 5 folds by conversation. Every conversation is scored by a model that never saw it. Answer recall, with a 95% CI clustered by conversation, is compared with the shipped recipe fitted on the same folds.

| Corpus | RAG | P1 | Shipped recipe | **New** | New − shipped |
|---|---|---|---|---|---|
| LoCoMo | 28.1 | 37.5 | 49.7 | **50.3** | +0.7 [−3.2, +4.0] n.s. |
| REALTALK | 16.9 | 32.5 | 30.0 | **36.2** | +6.2 [−2.0, +12.3] n.s. |
| BEAM 100K | 32.4 | 35.3 | 32.4 | **73.5** | **+41.2 [+18.9, +65.5]** |
| BEAM 500K | 32.7 | 32.7 | 25.5 | **63.6** | **+38.2 [+23.1, +53.2]** |
| BEAM 100K, app path | | | 61.8 | **79.4** | **+17.6 [+6.9, +28.6]** |

The new model loses nowhere under P1-S, the app's default mode. It is significantly better on both BEAM scales, and BEAM answers are long, which is the kind of input a person pastes into the app.

The BEAM sample is small: 34 and 55 answerable questions, because most BEAM answers are paraphrases rather than words copied from the chat. The confidence intervals are wide, but they exclude zero.

**A document uploaded into a chat.** For each LoCoMo conversation, a document of BEAM text was inserted halfway through, and the LoCoMo questions were asked at S = 4,096 through the app path. Both models are in-sample on LoCoMo here.

| Document mid-chat | Old scorer | New scorer |
|---|---|---|
| none | 53.1% | 53.4% |
| 8,000 tokens | 26.7% | 47.3% |
| 20,000 tokens | **10.2%** | **43.6%** |

The old scorer rated long document pieces above the person's own short facts, so one upload pushed most of the chat out of memory. This was the largest gap the stress test found in the app.

**Trade-off: the threshold modes.** P1 (threshold) and P1-A read scores as absolute, and the new model is still lower in absolute terms on short facts: the flight-date line scores 0.07, against 0.79 before. Its logit was rescaled to the old model's mean and spread on LoCoMo + REALTALK, which changes no ranking. In-sample answer recall through the app path:

| Mode | LoCoMo | REALTALK |
|---|---|---|
| P1 (threshold) | 74.8% → 72.4% | 64.4% → 59.4% |
| P1-A | 35.9% → 33.7% | 29.4% → 33.8% |

The app's default, P1-S, only orders by score.

**What it does not change.** The paper's numbers come from `train_transfer.py`'s leave-one-out models, not from the shipped file, so no paper table moves. `run_live_check.py` pins the heuristic scorer, so the live check still matches.

## 7. What this does not test

- **The chat model's answers.** The cloud cannot reach a model, so this tests the context MemGate hands it. Run the jury script on the laptop for real replies.
- **LongMemEval.** It is unreachable from the cloud.
- **BEAM 500K through the app's chunking.** It means about 470,000 pieces to embed, which takes hours on a CPU. BEAM 100K (64,000 pieces) was run instead.
