#!/usr/bin/env python3
"""Save an API key (Anthropic or Google Gemini) in the project's private .env
file, once.

    python set_key.py            # asks for it (typing is hidden)
    python set_key.py --remove   # forget the saved keys
    python set_key.py --if-missing   # ask only when none is saved (setup does this)

The key goes into .env at the top of the repo, which git ignores, so it stays
on this laptop. An Anthropic key (sk-ant-...) makes Claude write the replies;
a Gemini key (AIza..., free from https://aistudio.google.com/apikey) makes
Gemini write them. The kind is told from the key itself.
"""
import argparse
import getpass
import sys

from memgate.envfile import (ENV_PATH, GEMINI_KEY, KEY, key_name, read_env,
                             write_setting)

LABEL = {KEY: "Claude", GEMINI_KEY: "Gemini"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remove", action="store_true", help="forget the saved keys")
    ap.add_argument("--if-missing", action="store_true",
                    help="do nothing when a key is already saved")
    args = ap.parse_args()

    if args.remove:
        for name in (KEY, GEMINI_KEY):
            write_setting(name, "")
        print(f"  removed the keys from {ENV_PATH}")
        return 0
    saved = [n for n in (KEY, GEMINI_KEY) if read_env().get(n)]
    if args.if_missing and saved:
        print(f"  {LABEL[saved[0]]} key: saved in {ENV_PATH}")
        return 0
    if not sys.stdin.isatty():
        print("  API key: skipped (no terminal to type it in). "
              "Run  python set_key.py  later to add one.")
        return 0

    print("\n  Optional: an API key lets Claude or Gemini write the replies.")
    print("  Paste an Anthropic key (sk-ant-...) or a Gemini key (AIza...)")
    print("  and press Enter (it won't show), or just press Enter to skip.")
    for _ in range(3):
        key = getpass.getpass("  API key: ").strip()
        if not key:
            print("  skipped: the app will use the local model. "
                  "Run  python set_key.py  later to add a key.")
            return 0
        name = key_name(key)
        if name:
            write_setting(name, key)
            print(f"  saved as {name} in {ENV_PATH} (private to this laptop, "
                  f"never committed). The app will answer with {LABEL[name]}.")
            return 0
        print("  That doesn't look like an Anthropic key (sk-ant-...) or a "
              "Gemini key (AIza...). Try again, or press Enter to skip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
