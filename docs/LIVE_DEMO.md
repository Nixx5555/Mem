# MemGate — live demo guide

There are two live pieces, and both run on one page:

1. **Chat through MemGate.** You type, the local model answers, and the three
   tiers update on screen. Each card shows the message's score, what was kept,
   compressed or forgotten, and the exact context the model was given.
2. **Benchmark check.** This re-runs the slide 11/12 comparison on all of
   LoCoMo in about 20 seconds and checks the result against the committed CSV.
   The expected output is RAG 28.1%, P1-S 36.8%, both **MATCH**, +8.7 points.

Everything runs offline on the laptop. No internet and no API key are needed.

## Starting it

```bash
cd memgate
python3 live_demo.py          # loads MiniLM + Qwen2.5-1.5B (~10 s), then:
# open http://localhost:8765
```

* If the model won't load, the app says so in the terminal and carries on
  without it. `python3 live_demo.py --no-llm` does the same on purpose. The
  memory still works and the page shows what the model *would* have been sent.
* `python3 run_live_check.py` runs the benchmark check from a terminal on its
  own, without the page.
* Start it **before** the review and leave it running. Then close other heavy
  apps. The machine has 8 GB of memory and the demo uses about 2–3 GB.

## What's on screen

The main page (`/`) is a chat app; the memory sits in a side panel behind the
**Memory** button. For the script below use the **memory view** at
`http://localhost:8765/inspect`, which lays everything out at once:

| Area | What it shows |
|---|---|
| **Chat** (left) | The conversation. The model sees *only* what MemGate sends, never the history. |
| **Tier 1** | The last 4 messages, verbatim. Every message enters here. |
| **Tier 2** | Compressed gists. |
| **Tier 3** | Kept facts with their vectors, retrieved by meaning. A struck-through card has been superseded by a newer fact. |
| **Decisions** | Every routing decision: the score *u*, its label, and where the message went and why. |
| **Context sent** | The exact notes put in the prompt for the last message, tagged by tier. |
| **Mode** | **P1**: threshold routing, exactly slide 6. **P1-S**: the headline policy, which forgets the lowest score first once the store is full. **P1-A**: demotes to a gist instead of forgetting. |

The app opens at **full scale**, the sizes the experiments ran with: Tier 1
holds 6 messages, the context is 2048 tokens, and P1-S/P1-A use S = 4096. At
that scale nothing is evicted until a long conversation. For the jury, pick
**Demo scale** in the top bar: Tier 1 holds 4 messages, the context is 512
tokens and S = 150, so decisions happen within a few messages. Those sizes are
the only settings that differ from the experiments.

Every conversation is saved (in `memgate/chats/`) and comes back after a
restart. *New chat* starts a fresh memory; the dropdown next to it switches
between saved chats. Every decision on screen comes from the same `observe()` and
`build_context()` code the results came from.

## The script

Follow `docs/JURY_DEMO_SCRIPT.md`: what to click, what to type and what to say
at each step, with the numbers after the scorer fix. It uses the chat page
(`/`) and its **Memory** panel.

## What it can't show, and how to answer if asked

* **Accuracy on the panel's own chat.** There's no answer key for a live
  conversation, so there's nothing to score against. That is what the
  benchmark check is for.
* **The model is small (1.5B).** Its replies are plain, and it can be wrong
  even when the right note is in its context. That is the model reading
  badly, not the memory failing, and the context panel shows the
  difference. The demo is about the memory decisions, not the chatbot.
* **End-task accuracy (slide 17) is not re-run live.** It takes hours. The
  benchmark check reproduces context recall, which is the ceiling it is
  measured against.

## If something breaks

* **The page shows an error:** press *New chat* and continue.
* **The model hangs:** restart with `--no-llm`. The tiers, decisions and
  context panel all still work.
* **The benchmark check says DIFFERS:** check the info line at the top right.
  If the embedder is `hash-bow` rather than `all-MiniLM-L6-v2`, MiniLM failed
  to load and the numbers are not expected to match. Restart the demo.
* **Keep a screen recording** of one clean run of the script above as a
  backup.
