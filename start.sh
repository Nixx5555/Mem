#!/usr/bin/env bash
# Start MemGate and open it in the browser. Extra options go to app.py,
# e.g. ./start.sh --llm claude   or   ./start.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")/memgate"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
. .venv/bin/activate
exec python app.py "$@"
