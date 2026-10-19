"""The project's private settings file, `.env` at the top of the repo.

Setup writes the Anthropic key there once, and the app reads it on every
start, so nobody has to export it in each terminal. `.env` is gitignored: the
key stays on the laptop that typed it and never reaches GitHub.

    ANTHROPIC_API_KEY=sk-ant-...
    MEMGATE_LLM=auto

Values already set in the environment win over the file.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(ROOT, ".env")
KEY = "ANTHROPIC_API_KEY"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_env(path=ENV_PATH):
    """{name: value} from a .env file; {} when there is none."""
    out = {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[name] = value
    return out


def load_env(path=ENV_PATH):
    """Copy the file's settings into os.environ without overriding any.
    Returns the names it set."""
    applied = []
    for name, value in read_env(path).items():
        if value and not os.environ.get(name):
            os.environ[name] = value
            applied.append(name)
    return applied


def looks_like_key(key):
    return bool(re.fullmatch(r"sk-ant-[A-Za-z0-9_\-]{20,}", key or ""))


def write_setting(name, value, path=ENV_PATH):
    """Set one name in the file, keeping every other line, readable only by
    this user."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = ["# MemGate private settings. Never commit this file."]
    new = f"{name}={value}"
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if m and m.group(1) == name:
            lines[i] = new
            break
    else:
        lines.append(new)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
