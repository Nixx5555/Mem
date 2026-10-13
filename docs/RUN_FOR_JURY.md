# Running MemGate for the jury

Two things to show: the **live chat demo** (memory tiers updating as you type)
and the **results dashboard**. Do the setup once at home, with internet.
On the day, nothing needs the internet.

## 1. One-time setup (with internet)

```bash
git clone https://github.com/ruthme1ster/MemGate.git
cd MemGate
git checkout endtask-full-run
cd memgate

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install sentence-transformers tiktoken
pip install mlx-lm                 # Mac with Apple silicon (M1/M2/M3) only
```

Download the benchmark and the two models (about 1 GB in total):

```bash
mkdir -p data/locomo
curl -L -o data/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

python3 -c "from huggingface_hub import snapshot_download as d; \
d('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2', \
  allow_patterns=['*.json', 'vocab.txt', 'model.safetensors']); \
d('mlx-community/Qwen2.5-1.5B-Instruct-4bit', local_dir='models/Qwen2.5-1.5B-Instruct-4bit')"
```

Check it once: `python3 run_live_check.py` should end with two **MATCH** lines
(RAG 28.1%, P1-S 36.8%).

## 2. On the day

```bash
cd MemGate/memgate
source .venv/bin/activate
python3 live_demo.py
```

Wait for `open http://localhost:8765`, then open that address in the browser.
Start it before the review and leave it running.

The dashboard needs no server: open `MemGate/frontend/index.html` in the
browser (double-click it).

The 3-minute script for the demo is in `docs/LIVE_DEMO.md`.

## If something goes wrong

* **The model won't load** (or the laptop is not an Apple-silicon Mac): the
  demo now carries on without it and says so in the terminal. Tiers,
  decisions and the context panel all still work; replies show
  "(no model loaded ...)". You can also start it that way on purpose with
  `python3 live_demo.py --no-llm`.
* **Benchmark check says DIFFERS:** the top-right corner shows which embedder
  ran. If it says `hash-bow`, MiniLM did not load; redo the download step.
* **Port in use:** `python3 live_demo.py --port 9000`, then open
  http://localhost:9000.
