"""The project's private settings file, `.env` at the top of the repo.

Setup writes the API key (Anthropic or Gemini) there once, and the app reads it on every
start, so nobody has to export it in each terminal. `.env` is gitignored: the
key stays on the laptop that typed it and never reaches GitHub.

    ANTHROPIC_API_KEY=sk-ant-...      or      GEMINI_API_KEY=AIza...
    MEMGATE_LLM=auto

Values already set in the environment win over the file.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(ROOT, ".env")
# A .env made next to app.py by mistake is read too, after the main one.
ALT_PATH = os.path.join(ROOT, "memgate", ".env")
KEY = "ANTHROPIC_API_KEY"
GEMINI_KEY = "GEMINI_API_KEY"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_env(path=ENV_PATH):
    """{name: value} from a .env file; {} when there is none."""
    out = {}
    try:
        lines = _read_text(path).splitlines()
    except (OSError, UnicodeDecodeError):
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


def _read_text(path):
    """The file's text. Windows PowerShell's `echo X=1 > .env` writes UTF-16,
    and Notepad may add a BOM; both are read like plain UTF-8."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_env(path=None):
    """Copy the file's settings into os.environ without overriding any.
    With no path, reads .env at the top of the repo, then memgate/.env.
    Returns the names it set."""
    applied = []
    for p in ([path] if path else [ENV_PATH, ALT_PATH]):
        for name, value in read_env(p).items():
            if value and not os.environ.get(name):
                os.environ[name] = value
                applied.append(name)
    return applied


def looks_like_key(key):
    return bool(re.fullmatch(r"sk-ant-[A-Za-z0-9_\-]{20,}", key or ""))


def looks_like_gemini_key(key):
    """Google AI Studio keys: the older AIza... ones, and the authentication
    keys (AQ....) it has made by default since May 2026."""
    return bool(re.fullmatch(r"AIza[A-Za-z0-9_\-]{30,}", key or "")
                or re.fullmatch(r"AQ\.[A-Za-z0-9_\-.]{20,}", key or ""))


def key_name(key):
    """The setting a pasted key belongs in, or None if it fits neither."""
    if looks_like_key(key):
        return KEY
    if looks_like_gemini_key(key):
        return GEMINI_KEY
    return None


def write_setting(name, value, path=ENV_PATH):
    """Set one name in the file, keeping every other line, readable only by
    this user."""
    try:
        lines = _read_text(path).splitlines()
    except (OSError, UnicodeDecodeError):
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
