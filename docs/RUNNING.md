# Running MemGate

MemGate is a chat app with a memory layer in front of the model. Every
message goes through MemGate, which decides whether to keep, compress or
forget it. When you write, MemGate picks the notes the model needs. The
model never sees the raw history, only those notes. The **Memory** panel shows
exactly what was kept, what was forgotten, and what was sent to the model.
Chats are saved on your laptop and come back after a restart.

## What you need

* Python 3.10 or newer (`python3 --version`; on Windows, `py --version`).
* About 4 GB of free disk space for the models, and internet for the first
  setup only.
* **No API keys.** See [Keys](#keys) for the one optional key.

## 1. Get the code

```bash
git clone -b endtask-full-run https://github.com/Nixx5555/Mem.git MemGate
cd MemGate
```

## 2. Set up (once)

**macOS or Linux**

```bash
./setup.sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script does three things:

1. It creates a virtual environment in `memgate/.venv`.
2. It installs the packages from `memgate/requirements.txt`.
3. It downloads the models and the benchmark.

What it downloads depends on your laptop:

| Laptop | Chat model | Download |
|---|---|---|
| Mac with M1/M2/M3/M4 | Qwen2.5-1.5B-Instruct, 4-bit, via mlx | about 1 GB |
| Windows, Linux, Intel Mac | Qwen2.5-1.5B-Instruct via transformers | about 3 GB |

Plus the MiniLM embedder (90 MB) and the LoCoMo benchmark (3 MB). It's safe
to run again: finished downloads are kept. To skip the local chat model
because you'll use Claude instead, run `./setup.sh --no-chat`.

## 3. Start it

```bash
./start.sh          # macOS / Linux
start.bat           # Windows
```

Your browser opens at `http://localhost:8765`. The terminal shows the
embedder, scorer and chat model that loaded. After setup, no internet is
needed: everything runs on the laptop.

To stop it, press Ctrl+C in the terminal.

## Using it

* **Chat:** type in the box at the bottom. Enter sends; Shift+Enter starts a
  new line.
* **Chats:** past chats are in the left sidebar. **＋ New chat** starts a
  fresh memory, and ✕ deletes a chat. They're stored as files in
  `memgate/chats/`.
* **Memory** (top right) has three tabs:
  * **Tiers**: what MemGate is holding right now.
  * **Decisions**: every keep, compress or forget decision, with its score.
  * **Sent to model**: the exact notes behind the last reply.
  * **Compare**: the last question answered from three memories with the
    same budgets: MemGate, a sliding window and RAG store-all. Notes from
    messages both baselines had forgotten are green. **Answer from this
    memory** asks the model again from a baseline's memory (nothing is
    saved).
  * Click any Tier 2 or Tier 3 note to see **why it got its score**: each
    part that raised or lowered it, adding up exactly to the number.
    **Decisions** also has a box to score any message you type.
* **Settings** (bottom left) apply to new chats:
  * **Memory**: the policy. P1-S, which forgets the lowest-scored message
    first, is the default and the best one.
  * **Sizes**: *Full* is the sizes the experiments used. *Compact* makes it
    forget within a few messages, which is useful for showing how it works.
* **Benchmark check** re-runs the LoCoMo comparison (1,527 questions) and
  checks it against the committed results: RAG 28.1%, P1-S 37.5%.
* **Results dashboard:** open `frontend/index.html` in the browser. It needs
  no server.
* **All tiers side by side:** `http://localhost:8765/inspect`.

## Keys

You need **no keys** to run MemGate.

| Key | Needed? | What for |
|---|---|---|
| `ANTHROPIC_API_KEY` | Optional | Lets Claude write the replies instead of the local Qwen model. Replies are better, but it needs internet and is billed to your Anthropic account. |
| `GEMINI_API_KEY` | Optional | Lets Google Gemini write the replies. The free key from https://aistudio.google.com/apikey works (starts with `AQ.`, or `AIza` for older keys); it needs internet. Used when no Anthropic key is saved. |

The models and the benchmark are public, so you don't need a Hugging Face
token or a GitHub login.

**Adding the key.** At the end of setup you're asked for it once. Paste an
Anthropic or a Gemini key (the kind is told from the key itself)
(the typing is hidden) or press Enter to skip. It's saved in `.env` at the top
of the project, and `start.sh` / `start.bat` read it on every start. `.env` is
in `.gitignore`, so the key stays on your laptop and never goes to GitHub.
Never paste the key into a chat or into a file you commit.

To add, change or remove it later, inside `memgate/` with the environment
active (or via `.venv/bin/python` / `.venv\Scripts\python.exe`):

```bash
python set_key.py            # add or replace
python set_key.py --remove   # forget it
```

You can also edit `.env` by hand; `.env.example` shows the format.

**If Google rejects an `AQ.` key** ("API key not valid"), check it with
`python set_key.py --check`. Some of AI Studio's new `AQ.` keys are refused by
the Gemini API, a Google bug it has acknowledged. The app already tries every
way of sending the key. The fix is an older-style `AIza...` key: in the
Google Cloud console for the same project (https://console.cloud.google.com/apis/credentials),
Create credentials → API key, then Edit → API restrictions → Restrict key →
tick **Generative Language API** → Save. Save it with `python set_key.py`.

**Gemini's free tier** allows only a few replies a minute. When it says
"limit reached", the app waits a few seconds and tries again; if the limit
lasts, the downloaded Qwen answers that one message and the next one goes back
to Gemini. Without Qwen, the reply asks you to wait a minute. Memory is saved
either way.

**With a key saved, the app answers with Claude (or Gemini).** If Claude or Gemini can't be reached
(no Wi-Fi at the venue, a revoked key), it switches to the local model on its
own and carries on; the info line in the sidebar then says *(fallback)*. To
force the local model even with a key saved, start with `--llm mlx` (Mac) or
`--llm hf`.

## Options

Pass these to `start.sh` / `start.bat` (or `python app.py` inside `memgate/`):

| Option | |
|---|---|
| `--llm auto\|mlx\|hf\|claude\|gemini\|none` | Which model answers. `auto` picks Claude when an Anthropic key is saved, else Gemini when a Gemini key is saved, else mlx on an Apple-silicon Mac, else the downloaded Qwen. Also settable as `MEMGATE_LLM`. |
| `--no-llm` | No model. Memory still works, and replies show what the model would see. |
| `--scorer learned\|heuristic` | What scores each message. `learned` (the default) is the scorer trained on LoCoMo, REALTALK, BEAM (100K to 10M), MemBench and everyday chat from PersonaChat, 856,876 turns (docs/SCALE_TEST.md §4b). `heuristic` is the hand-written P1 rules. Also settable as `MEMGATE_SCORER`. |
| `--port 9000` | Use another port if 8765 is taken. |
| `--host 0.0.0.0` | Let other devices on the same network open it, e.g. a projector PC. |
| `--no-browser` | Don't open the browser on start. |
| `MEMGATE_CLAUDE_MODEL` | The Claude model id (default `claude-opus-5-5`). |
| `MEMGATE_GEMINI_MODEL` | The Gemini model id (default `gemini-flash-latest`, on the free tier; `gemini-2.5-flash-lite` has higher free limits). |
| `MEMGATE_HF_MODEL` | Another Hugging Face model for `--llm hf`. |

## If something goes wrong

* **`setup` stops during a download:** run it again. It picks up where it
  stopped.
* **The model won't load:** the terminal says why, and the app carries on
  without it. Tiers, decisions and the context panel all still work.
  `python setup_models.py --check` (inside `memgate/`) lists what's missing.
* **The scorer says `P1-heuristic` instead of `P3-learned`:** the learned
  scorer needs the MiniLM embedder. Run setup again.
* **Benchmark check says DIFFERS:** the embedder line in the bottom-left
  corner says `hash-bow`, which means MiniLM did not load. Run setup again.
* **Port already in use:** `./start.sh --port 9000`.

For the jury review, follow `docs/JURY_DEMO_SCRIPT.md`.
