#!/usr/bin/env python3
"""Save the Anthropic API key in the project's private .env file, once.

    python set_key.py            # asks for it (typing is hidden)
    python set_key.py --remove   # forget the saved key
    python set_key.py --if-missing   # ask only when none is saved (setup does this)

The key goes into .env at the top of the repo, which git ignores, so it stays
on this laptop. With a key saved, the app answers with Claude.
"""
import argparse
import getpass
import sys

from memgate.envfile import ENV_PATH, KEY, looks_like_key, read_env, write_setting


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remove", action="store_true", help="forget the saved key")
    ap.add_argument("--if-missing", action="store_true",
                    help="do nothing when a key is already saved")
    args = ap.parse_args()

    if args.remove:
        write_setting(KEY, "")
        print(f"  removed the key from {ENV_PATH}")
        return 0
    if args.if_missing and read_env().get(KEY):
        print(f"  Anthropic key: saved in {ENV_PATH}")
        return 0
    if not sys.stdin.isatty():
        print("  Anthropic key: skipped (no terminal to type it in). "
              "Run  python set_key.py  later to add one.")
        return 0

    print("\n  Optional: an Anthropic API key lets Claude write the replies.")
    print("  Paste it and press Enter (it won't show), or just press Enter to skip.")
    for _ in range(3):
        key = getpass.getpass("  Anthropic API key: ").strip()
        if not key:
            print("  skipped: the app will use the local model. "
                  "Run  python set_key.py  later to add a key.")
            return 0
        if looks_like_key(key):
            write_setting(KEY, key)
            print(f"  saved in {ENV_PATH} (private to this laptop, never "
                  "committed). The app will answer with Claude.")
            return 0
        print("  That doesn't look like an Anthropic key (they start with "
              "sk-ant-). Try again, or press Enter to skip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
