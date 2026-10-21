# MemGate

[![tests](https://github.com/Nixx5555/Mem/actions/workflows/tests.yml/badge.svg?branch=endtask-full-run)](https://github.com/Nixx5555/Mem/actions/workflows/tests.yml)

An adaptive memory layer for long-running LLM chats. For every message,
MemGate decides whether to keep it, compress it, forget it, or recall it
later, under a fixed token budget. The model only ever sees the notes MemGate
hands it. NMIMS Indore capstone, Simar Singh Khanuja and Yash Ramchandani.

## Run the app

```bash
git clone -b endtask-full-run https://github.com/Nixx5555/Mem.git MemGate
cd MemGate
./setup.sh        # once; Windows: powershell -ExecutionPolicy Bypass -File setup.ps1
./start.sh        # Windows: start.bat
```

The browser opens at `http://localhost:8765`. You need no API keys; the chat
model runs on your laptop. Setup offers to save an optional Anthropic key in a
private `.env` file, and then Claude writes the replies. [docs/RUNNING.md](docs/RUNNING.md) has the full
guide: options, the one optional key (`ANTHROPIC_API_KEY`, for Claude
replies), and troubleshooting.

## The results

On LoCoMo, with storage capped, forgetting the lowest-scored message first
beats forgetting the oldest by **+9.4 answer-recall points**: 37.5% against
28.1%, 95% CI [+5.4, +14.9], p = 2.8×10⁻⁵. A scorer trained on LoCoMo plus
REALTALK adds **+14.6** more on conversations it never saw (at the same
S = 4096; the gain is significant at every budget from 2,048 to 16,384 tokens).

| | |
|---|---|
| `REPORT.md` | The paper-ready write-up |
| `paper/` | The IEEE paper (`main.tex`) |
| `docs/SCORER_FIX_RESULTS.md`, `docs/SCORER_TRANSFER.md` | The scorer fix and the learned scorer |
| `frontend/index.html` | The interactive results dashboard (opens offline) |
| `memgate/` | The library, the app (`app.py`), 17 experiment scripts and the tests |

Run all the tests inside `memgate/` with `python run_tests.py` (220 tests in
six suites; add `--full` to also re-run the LoCoMo benchmark check). They
need the downloaded models.
