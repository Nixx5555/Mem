#!/usr/bin/env bash
# One-time setup on macOS or Linux: a virtualenv, the packages, the models.
#   ./setup.sh             everything (about 1-3 GB of downloads)
#   ./setup.sh --no-chat   skip the local chat model (use Claude, or memory only)
# At the end it offers to save an Anthropic key in .env (optional, private).
set -euo pipefail
cd "$(dirname "$0")/memgate"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "MemGate needs Python 3.10 or newer (found $("$PY" --version))"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
python setup_models.py "$@"
python set_key.py --if-missing
echo
echo "Setup finished. Start MemGate with ./start.sh"
