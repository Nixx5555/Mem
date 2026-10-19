#!/usr/bin/env python3
"""Download everything the MemGate app needs, once, into models/ and data/.

    python3 setup_models.py              # embedder + chat model + LoCoMo
    python3 setup_models.py --no-chat    # skip the chat model (Claude or memory-only)
    python3 setup_models.py --check      # only report what is present

Needs internet the first time. After that the app runs offline: memgate pins
the Hugging Face libraries offline on import whenever models/ exists.

What it fetches (all public, no token needed):
    models/all-MiniLM-L6-v2               the embedder, ~90 MB (required)
    models/Qwen2.5-1.5B-Instruct-4bit     the chat model on an Apple-silicon Mac (mlx), ~900 MB
    models/Qwen2.5-1.5B-Instruct          the chat model elsewhere (transformers), ~3 GB
    data/locomo/locomo10.json             the benchmark, 2.7 MB (for Benchmark check)
"""
import argparse
import os
import platform
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
LOCOMO = os.path.join(HERE, "data", "locomo", "locomo10.json")
LOCOMO_URL = ("https://raw.githubusercontent.com/snap-research/locomo/main/"
              "data/locomo10.json")

EMBEDDER = ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2",
            ["*.json", "vocab.txt", "model.safetensors"])
CHAT_MLX = ("mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "Qwen2.5-1.5B-Instruct-4bit", None)
CHAT_HF = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct",
           ["*.json", "*.safetensors", "merges.txt", "vocab.json"])


def apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def present(local):
    return os.path.isfile(os.path.join(MODELS, local, "config.json"))


def fetch_model(repo, local, patterns):
    if present(local):
        print(f"  ok       models/{local}")
        return
    print(f"  fetching {repo} -> models/{local} ...", flush=True)
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=os.path.join(MODELS, local),
                      allow_patterns=patterns)
    print(f"  ok       models/{local}")


def fetch_locomo():
    if os.path.isfile(LOCOMO):
        print("  ok       data/locomo/locomo10.json")
        return
    print("  fetching LoCoMo -> data/locomo/locomo10.json ...", flush=True)
    os.makedirs(os.path.dirname(LOCOMO), exist_ok=True)
    tmp = LOCOMO + ".part"
    urllib.request.urlretrieve(LOCOMO_URL, tmp)
    os.replace(tmp, LOCOMO)
    print("  ok       data/locomo/locomo10.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-chat", action="store_true",
                    help="skip the local chat model")
    ap.add_argument("--chat", choices=("auto", "mlx", "hf"), default="auto",
                    help="which local chat model to fetch (default: mlx on an "
                         "Apple-silicon Mac, else transformers)")
    ap.add_argument("--check", action="store_true",
                    help="report what is present and download nothing")
    args = ap.parse_args()

    chat = None
    if not args.no_chat:
        kind = args.chat
        if kind == "auto":
            kind = "mlx" if apple_silicon() else "hf"
        chat = CHAT_MLX if kind == "mlx" else CHAT_HF

    wanted = [EMBEDDER] + ([chat] if chat else [])
    if args.check:
        for repo, local, _ in wanted:
            print(f"  {'ok     ' if present(local) else 'MISSING'}  models/{local}")
        print(f"  {'ok     ' if os.path.isfile(LOCOMO) else 'MISSING'}  "
              "data/locomo/locomo10.json")
        return 0

    print("MemGate setup: downloading models and data (first run only)")
    try:
        for repo, local, patterns in wanted:
            fetch_model(repo, local, patterns)
        fetch_locomo()
    except Exception as exc:
        print(f"\n  setup failed: {type(exc).__name__}: {exc}\n"
              "  Check the internet connection and run this again; finished "
              "downloads are kept.", file=sys.stderr)
        return 1
    print("\n  done. Start the app with:  python3 app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
