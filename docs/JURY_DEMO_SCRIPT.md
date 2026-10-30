# MemGate: jury demo script

About 6 minutes, plus questions. It assumes all four PRs are merged: the chat
app, the scorer fix, the Review 2 docs, and the learned scorer. Every score
and decision below was checked by replaying these exact messages through the
merged code.

The app scores messages with the **learned scorer** by default. It was
trained on LoCoMo plus REALTALK and is the one the app reports in the sidebar
as `P3-learned`. To demo the hand-written P1 rules instead, start the app with
`--scorer heuristic`.

## Before the jury walks in

1. Start the app and leave it running (`docs/RUNNING.md` has the setup):
   ```bash
   cd MemGate && ./start.sh        # Windows: start.bat
   ```
   Open `http://localhost:8765` and make sure the terminal shows
   `embedder: all-MiniLM-L6-v2` (not `hash-bow`) and `scorer: P3-learned`.
   The learned scorer needs MiniLM; without it the app falls back to P1.
2. In the left sidebar set **Sizes → Compact (forgets quickly)** and
   **Memory → P1-S · forget lowest score first**. Changing either starts a
   fresh chat, which is what you want.
3. Open `MemGate/frontend/index.html` in a second browser tab (the results
   dashboard).
4. Run **Benchmark check** once now, so you know it ends in two MATCH lines
   on this laptop.
5. Close other heavy apps. Keep the 96-second recorded walkthrough
   (`memgate_demo.mp4`, made with `docs/demo/record_demo.js`) open in a player
   as the backup.

## 1. What MemGate is (30 s, no clicking)

**Say:** "A language model can only read a fixed window. MemGate sits in front
of a frozen model and decides, for every message, whether to keep it,
compress it, forget it, or recall it later, under a token budget. The model
never sees the chat history, only what MemGate hands it."

## 2. Teach it something live (1 min)

In the empty chat, click the three suggestion buttons in order, or type:

1. `My name is Simar and my capstone final review is on 6 November.`
2. `Actually the review moved to 8 November.`
3. `When is my final review?`

The model should answer **8 November**.

**Click** the "notes from memory" chip under the reply, then the
**Sent to model** tab.

**Say:** "This is everything the model was given. It never saw the chat,
only these notes, each tagged with the tier it came from. Both the old date
and the correction are there, and the correction is the newer note, so the
model takes 8 November."

## 3. Watch it decide (2 min)

**Click** **＋ New chat** (still Compact sizes, P1-S), then **Load sample chat**.
This is 18 messages: greetings, small talk, a deadline, a correction, and a
preference.

Open **Memory**, then **Decisions**. Point at three things:

* **Small talk was forgotten first.** "You're welcome!" (u = 0.01), "ok
  thanks" (0.09), "Nice to meet you" (0.14), "Hi!" (0.19) and "Hello! How
  can I help you today?" (0.28) were the five forgotten. Once the store went
  over its budget (S = 150 tokens), the lowest score went first each time:
  *store over budget: lowest-score fact forgotten*.
* **The facts stayed.** The review date (u = 0.99), the deadline correction
  (0.97), the name (0.96) and the NMIMS-template preference (0.76) are all
  still in **Tiers → Tier 3**.
* **The old deadline was retired.** The last decision on the list says the
  assistant's "Got it: report due 2 November" was *superseded: a newer fact
  about the same thing replaced it, and its space was reclaimed*. It is no
  longer in Tier 3.

Then ask: `When is my report due?` The answer should be **4 November**.

**Say:** "This is the headline policy, P1-S. When storage is full it forgets
the lowest-scored messages first, instead of the oldest. The scores come from
a small model trained on 20 real and generated conversations. With our
hand-written rules, this policy beats forgetting oldest-first by **+9.4
answer-recall points** on LoCoMo. With the learned scorer, on conversations
it never trained on, it is **+14.6 points better again**: 52.0% against
37.5%."

## 3b. Same question, three memories (1 min)

Still in the sample chat, ask: `What is my name?`

**Click** the second chip under the reply, **"2 the baselines forgot"**. It
opens **Memory → Compare**: the same question answered from three memories
that each hold at most S = 150 tokens and may send 512.

* **MemGate** sent 11 notes, 140 tokens. Two are green, *only MemGate*: the
  name ("My name is Simar…", message 2) and the review date (message 4).
* **Sliding window (P0)** and **RAG store-all** each sent 14 notes, 137
  tokens: "lol yes", "ok thanks", "You're welcome!"… and no name. Both had
  already forgotten messages 0 to 4 because they were the oldest.
* Click **Answer from this memory** under either baseline. The model answers
  from that memory only, and should say it doesn't remember. Nothing is
  saved, so the chat is unchanged.

**Say:** "Same model, same budget, same question. The only difference is what
each memory chose to keep. Forgetting by age loses the name; forgetting by
score keeps it."

Then **click** any Tier 3 note in **Tiers**. It opens *why this score*: the
starting point, then each part that pushed the score up (green) or down
(red), largest first, adding up exactly to the u shown. For "My name is
Simar…" the largest part is what the message is about (MiniLM meaning),
then that it names someone. **Decisions** has a box to score any message:
`My exam is on 3 May` gives u = 0.98, `lol ok` gives 0.08.

**Say:** "The learned scorer is a logistic regression, so every score breaks
down exactly into its parts. Nothing here is a black box."

## 4. Why we don't compress (1 min)

**Change** **Memory → P1-A · demote to a gist** (this starts a new chat), then
**Load sample chat** again. Open **Tiers → Tier 2**.

Point at the gists it made under pressure:

* "My name is Simar and I'm a final-year student at NMIMS Indore." became
  **"Simar I'm NMIMS Indore."**
* "Nice to meet you, Simar. What are you working on?" became
  **"meet Simar. What working"**

**Say:** "Compressing keeps more items, but each one loses words. On the
benchmark, compressing instead of dropping loses **7.1 answer points**, even
though it scores better on evidence recall. Keeping a few things whole beats
keeping everything half-legible."

## 5. The numbers are real (1 min)

**Click** **Benchmark check → Run** (S = 4096). It replays all 10 LoCoMo
conversations, 1,527 questions, in about 20 seconds to 2 minutes depending
on the laptop.

Expected result:

| | Live | Committed |
|---|---|---|
| RAG store-all | 28.1% | 28.1% · **MATCH** |
| P1-S MemGate select | 37.5% | 37.5% · **MATCH** |

"Selection over FIFO: **+9.4** answer-recall points."

**Say:** "These are the slide 11 and 12 numbers, recomputed live from the raw
benchmark. The 95% interval, clustered by conversation, is +5.4 to +14.9,
p = 2.8 × 10⁻⁵. This check uses the hand-written P1 rules on purpose,
because that is what the slides report."

If asked why it isn't the learned scorer: its 52.0% comes from a different
protocol (each conversation scored by a model that never saw it), in
`docs/SCORER_TRANSFER.md`.

Switch to the dashboard tab if they want the other experiments.

## 6. Hand it over

**Say:** "Please type anything you like." Let a jury member chat. This is the
strongest moment, so leave time for it. Keep the **Memory** panel open so
they see their own messages being scored.

## Likely questions

* **"Your Review 2 slides said +8.67."** "We found and fixed a bug: the scorer
  was reading the speaker's name as part of every message, so every LoCoMo
  turn looked like it named a person. With the fix the gain is +9.4, and
  every experiment was re-run. The learned scorer's lead over the
  hand-written one shrank from +5.2 to +2.7 points, because part of that lead
  was the bug."
* **"Does it work on other data?"** "Partly. On LongMemEval, which has long
  prose turns, the hand-written scorer ranks evidence *below* other turns, so
  selection lost to FIFO by 21 points (measured before the scorer fix). So we
  trained a scorer on LoCoMo plus REALTALK, which is real people chatting.
  On LoCoMo conversations it never saw, it beats the rules by +14.6 points.
  On REALTALK it is +3.8 ahead, but with only 160 questions that isn't
  significant. LongMemEval with the learned scorer is still to be run."
* **"Why not just store everything?"** "With storage capped, which is the
  realistic case, storing everything means forgetting oldest-first. That is
  the RAG row in the benchmark check."
* **"The replies are plain."** "The model is a 1.5B model running on this
  laptop. The demo is about what reaches the model, which the *Sent to
  model* tab shows exactly."
* **"How accurate is it on my chat?"** "A live chat has no answer key, which
  is why the benchmark check exists."

## If something breaks

* **The model doesn't reply:** restart with `./start.sh --no-llm`.
  Steps 2–5 still work: the reply then shows what the model *would* see.
* **Benchmark says DIFFERS:** check the embedder in the top-right info line.
  If it says `hash-bow`, MiniLM didn't load; restart.
* **The page shows an error:** press **＋ New chat** and carry on. Past
  chats stay in the sidebar.
