# MemGate: jury demo script

About 6 minutes, plus questions. It assumes all three PRs (chat app, scorer
fix, Review 2 docs) are merged. Every score and decision below was checked by
replaying these exact messages through the merged code.

## Before the jury walks in

1. Start the app and leave it running (`docs/RUN_FOR_JURY.md` has the setup):
   ```bash
   cd MemGate/memgate && source .venv/bin/activate
   python3 live_demo.py
   ```
   Open `http://localhost:8765` and make sure the terminal shows the
   embedder as `all-MiniLM-L6-v2`, not `hash-bow`.
2. In the left sidebar set **Sizes → Demo (evicts quickly)** and
   **Memory → P1-S · forget lowest score first**. Changing either starts a
   fresh chat, which is what you want.
3. Open `MemGate/frontend/index.html` in a second browser tab (the results
   dashboard).
4. Run **Benchmark check** once now, so you know it ends in two MATCH lines
   on this laptop.
5. Close other heavy apps. Keep a screen recording of one clean run as a
   backup.

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

**Click** **＋ New chat** (still Demo sizes, P1-S), then **Load sample chat**.
This is 18 messages: greetings, small talk, a deadline, a correction, and a
preference.

Open **Memory**, then **Decisions**. Point at three things:

* **Small talk was forgotten first.** "Hi!", "ok thanks", "You're welcome!"
  and "Sure." all scored about **u = 0.02**. Once the store went over its
  budget (S = 150 tokens), they were the first to go: *store over budget:
  lowest-score fact forgotten*.
* **The facts stayed.** The review date (u = 0.80), the deadline correction
  (u = 0.90), the name (u = 0.45) and the NMIMS-template preference
  (u = 0.15) are all still in **Tiers → Tier 3**.
* **The old deadline was retired.** The last decision on the list says the
  assistant's "Got it: report due 2 November" was *superseded: a newer fact
  about the same thing replaced it, and its space was reclaimed*. It is no
  longer in Tier 3.

Then ask: `When is my report due?` The answer should be **4 November**.

**Say:** "This is the headline policy, P1-S. When storage is full it forgets
the lowest-scored messages first, instead of the oldest. On the LoCoMo
benchmark, at the same storage, that is worth **+9.4 answer-recall points**
over forgetting oldest-first."

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
p = 2.8 × 10⁻⁵."

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
* **"Does it work on other data?"** "Not yet. On LongMemEval, which has long
  prose turns, the hand-written scorer ranks evidence *below* other turns, so
  selection loses to FIFO by 21 points (measured before the scorer fix; that
  run is still to be repeated). The mechanism is fine; the scorer doesn't
  transfer. A scorer that transfers is our remaining work."
* **"Why not just store everything?"** "With storage capped, which is the
  realistic case, storing everything means forgetting oldest-first. That is
  the RAG row in the benchmark check."
* **"The replies are plain."** "The model is a 1.5B model running on this
  laptop. The demo is about what reaches the model, which the *Sent to
  model* tab shows exactly."
* **"How accurate is it on my chat?"** "A live chat has no answer key, which
  is why the benchmark check exists."

## If something breaks

* **The model doesn't reply:** restart with `python3 live_demo.py --no-llm`.
  Steps 2–5 still work: the reply then shows what the model *would* see.
* **Benchmark says DIFFERS:** check the embedder in the top-right info line.
  If it says `hash-bow`, MiniLM didn't load; restart.
* **The page shows an error:** press **＋ New chat** and carry on. Past
  chats stay in the sidebar.
