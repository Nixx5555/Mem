# Running MemGate

MemGate runs as a local web app: you chat with a model, and every message goes
through the memory layer (keep, compress, forget, recall) before the model
sees anything. The page shows the three tiers, every routing decision, and the
exact context sent to the model. Conversations are saved and survive a
restart. A second page, the results dashboard, shows the experiment numbers.

## 1. One-time setup (needs internet)

```bash
git clone https://github.com/Nixx5555/Mem.git MemGate
cd MemGate/memgate

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install sentence-transformers tiktoken
```

Then install **one** chat model backend:

| Laptop | Install | Notes |
|---|---|---|
| Mac with M1/M2/M3/M4 | `pip install mlx-lm` | The model the experiments used. Fast. |
| Windows, Linux, Intel Mac | `pip install transformers torch` | Same Qwen model, runs on the CPU. Slower, a few seconds per reply. |
| Any, with internet | `pip install anthropic` and set `ANTHROPIC_API_KEY` | Claude answers. Best replies, needs internet at the review. |

Download the benchmark and the models (about 1–3 GB):

```bash
mkdir -p data/locomo
curl -L -o data/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# the embedder (everyone)
python3 -c "from huggingface_hub import snapshot_download as d; \
d('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2', \
  allow_patterns=['*.json', 'vocab.txt', 'model.safetensors'])"

# the chat model: Mac (mlx-lm)
python3 -c "from huggingface_hub import snapshot_download as d; \
d('mlx-community/Qwen2.5-1.5B-Instruct-4bit', local_dir='models/Qwen2.5-1.5B-Instruct-4bit')"

# the chat model: Windows / Linux / Intel Mac (transformers)
python3 -c "from huggingface_hub import snapshot_download as d; \
d('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
```

Check it once: `python3 run_live_check.py` should end with two **MATCH** lines
(RAG 28.1%, P1-S 36.8%; P1-S reads 37.5% once the scorer fix is merged).

## 2. Running it

```bash
cd MemGate/memgate
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 live_demo.py               # picks the backend you installed
# or: python3 live_demo.py --llm claude   (uses ANTHROPIC_API_KEY)
```

Wait for `open http://localhost:8765`, then open that address in a browser.
The terminal also prints which model and embedder loaded; check both say what
you expect.

The page is a normal chat: past chats on the left, the conversation in the
middle, the message box at the bottom. **Memory** (top right) opens a side
panel with what MemGate is holding (Tiers), every keep/compress/forget
decision (Decisions), and the exact notes the model was given for the last
reply (Sent to model). The small "notes from memory" chip under each reply
opens the same panel. `http://localhost:8765/inspect` shows all three tiers
side by side, which is the clearest view for the jury script.

The dashboard needs no server: open `MemGate/frontend/index.html` in the
browser.

For the review, follow `docs/JURY_DEMO_SCRIPT.md`. Pick
**Demo scale** in the top bar so evictions show within a few messages.

## Options

| | |
|---|---|
| `--llm auto\|mlx\|hf\|claude\|none` | Which model answers. `auto` tries mlx, then transformers. Also settable as `MEMGATE_LLM`. |
| `--no-llm` | No model. Memory still works; replies show what the model would see. |
| `--port 9000` | Use another port if 8765 is taken. |
| `--host 0.0.0.0` | Let other devices on the same network open it (e.g. a projector PC). |
| `MEMGATE_CLAUDE_MODEL` | Claude model id (default `claude-opus-5-5`). |
| `MEMGATE_HF_MODEL` | Hugging Face model for `--llm hf` (default Qwen2.5-1.5B-Instruct). |

## If something goes wrong

* **The model won't load:** the app says why in the terminal and carries on
  without it. Tiers, decisions and the context panel all still work.
* **Benchmark check says DIFFERS:** the top-right corner shows the embedder.
  If it says `hash-bow`, MiniLM did not load; redo the download step.
* **Start over:** *New chat*, or *Delete* to remove a saved conversation.
