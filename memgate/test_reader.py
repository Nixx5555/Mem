#!/usr/bin/env python3
"""Reader tests: which model answers, what it is sent, and the saved key.
No model is loaded and nothing goes to the internet: the backends are faked.
Run: python test_reader.py"""
import os
import sys
import tempfile
import types

import memgate.reader as R
from memgate import envfile as E

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
saved_env = {k: os.environ.pop(k, None) for k in KEYS}
real = (R._importable, R._apple_silicon, R._hf_ready, R._mlx_ready)


def world(mac=False, mlx=False, torch=False, anthropic=True, hf_ready=False,
          mlx_ready=False, key=False):
    """Fake the laptop: what is installed, what is downloaded, is a key set."""
    have = {"mlx_lm": mlx, "transformers": torch, "torch": torch,
            "anthropic": anthropic}
    R._importable = lambda n: have.get(n, False)
    R._apple_silicon = lambda: mac
    R._hf_ready = lambda: hf_ready
    R._mlx_ready = lambda: mlx_ready
    if key:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-" + "x" * 30
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


print("\n-- which model answers (auto) --")
world(key=True, mac=True, mlx=True, mlx_ready=True)
check("a saved key picks Claude, even on a Mac with mlx", R.resolve("auto") == "claude")
world(mac=True, mlx=True)
check("no key on an Apple-silicon Mac picks mlx", R.resolve("auto") == "mlx")
world(torch=True, hf_ready=True)
check("no key on Windows/Linux picks transformers", R.resolve("auto") == "hf")
world()
check("nothing installed and no key gives none", R.resolve("auto") == "none")
world(key=True, anthropic=False, torch=True)
check("a key without the anthropic package does not pick Claude",
      R.resolve("auto") == "hf")
world(key=False)
os.environ["ANTHROPIC_AUTH_TOKEN"] = "tok"
check("ANTHROPIC_AUTH_TOKEN counts as a key", R.resolve("auto") == "claude")
del os.environ["ANTHROPIC_AUTH_TOKEN"]
check("an explicit backend is kept", R.resolve("hf") == "hf" and R.resolve("NONE") == "none")
try:
    R.resolve("gpt")
    check("an unknown backend is refused", False)
except ValueError:
    check("an unknown backend is refused", True)

print("\n-- the none backend --")
m = R.ChatModel("none")
check("none is disabled and named so", not m.enabled and "none" in m.name)
check("none replies without a model", "no model" in m.reply("notes", "hi"))

print("\n-- the prompt --")
p = R._prompt("[user] My exam is on 8 November.", "When is my exam?")
check("the system prompt says the model sees only notes",
      p[0]["role"] == "system" and "only see the memory notes" in p[0]["content"])
check("the notes and the message are both sent",
      "8 November" in p[1]["content"] and "When is my exam?" in p[1]["content"])
check("empty notes are marked, not blank", "(empty)" in R._prompt("", "hi")[1]["content"])


class FakeClient:
    """Stands in for anthropic.Anthropic; records what it was sent."""
    def __init__(self, reply="  The exam is on 8 November.\n", stop="end_turn",
                 fail=None):
        self.sent, self.reply, self.stop, self.fail = [], reply, stop, fail
        self.beta = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.sent.append(kw)
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(
            stop_reason=self.stop,
            content=[types.SimpleNamespace(type="thinking", text="hidden"),
                     types.SimpleNamespace(type="text", text=self.reply)])


print("\n-- the Claude backend (faked) --")
world(key=True)
m = R.ChatModel("claude")
check("Claude uses the configured model id", m.path == R.CLAUDE_MODEL)
m._client = FakeClient()
out = m.reply("[user] My exam is on 8 November.", "When is my exam?")
kw = m._client.sent[0]
check("only the text blocks come back, trimmed", out == "The exam is on 8 November.")
check("the system prompt goes in system, not in messages",
      kw["system"] == R.SYSTEM and [x["role"] for x in kw["messages"]] == ["user"])
check("server-side fallback is requested",
      kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"])
m._client = FakeClient(stop="refusal", reply="")
check("a refusal is shown as such", "declined" in m.reply("", "x"))
os.environ["MEMGATE_CLAUDE_MODEL"] = "claude-sonnet-5-5"
check("MEMGATE_CLAUDE_MODEL overrides the model", R.ChatModel("claude").path == "claude-sonnet-5-5")
del os.environ["MEMGATE_CLAUDE_MODEL"]
world(key=False)
try:
    R.ChatModel("claude").load()
    check("Claude without a key refuses to load", False)
except RuntimeError as exc:
    check("Claude without a key refuses to load", "ANTHROPIC_API_KEY" in str(exc))

print("\n-- falling back to the local model --")
world(key=True, torch=True, hf_ready=True)
m = R.ChatModel("claude")
check("with a downloaded local model, Claude has a fallback", m.fallback == "hf")
world(key=True, torch=True, hf_ready=False)
check("without one, there is no fallback", R.ChatModel("claude").fallback is None)
world(key=True, torch=True, hf_ready=True)
m = R.ChatModel("claude")
m._client = FakeClient(fail=ConnectionError("no internet"))
m.load = lambda: None
calls = []


def fake_reply(notes, message, _orig=m._reply):
    if m.backend == "hf":
        calls.append(message)
        return "local answer"
    return _orig(notes, message)


m._reply = fake_reply
out = m.reply("notes", "hello")
check("Claude failing mid-chat answers with the local model",
      out == "local answer" and m.backend == "hf" and calls == ["hello"])
check("the switch is explained", m.note and "no internet" in m.note)
world(key=True)
m = R.ChatModel("claude")
m._client = FakeClient(fail=ConnectionError("no internet"))
try:
    m.reply("notes", "hello")
    check("with no fallback the error reaches the page", False)
except ConnectionError:
    check("with no fallback the error reaches the page", True)

class AuthenticationError(Exception):
    pass


m = R.ChatModel("claude")
m._client = FakeClient(fail=AuthenticationError("invalid x-api-key"))
try:
    m.reply("notes", "hello")
    check("a rejected key says how to replace it", False)
except RuntimeError as exc:
    check("a rejected key says how to replace it", "set_key.py" in str(exc))

print("\n-- the saved key (.env) --")
d = tempfile.mkdtemp()
env = os.path.join(d, ".env")
check("no file reads as empty", E.read_env(env) == {})
with open(env, "w") as fh:
    fh.write("# comment\n\nexport MEMGATE_SCORER=heuristic\nQUOTED='a b'\n"
             'DQ="x"\nnot a line\nEMPTY=\n')
got = E.read_env(env)
check("comments, export, quotes and blanks are handled",
      got == {"MEMGATE_SCORER": "heuristic", "QUOTED": "a b", "DQ": "x", "EMPTY": ""},
      str(got))
key = "sk-ant-api03-" + "A1b2_C3-" * 6
E.write_setting("ANTHROPIC_API_KEY", key, env)
check("the key is written and other lines kept",
      E.read_env(env)["ANTHROPIC_API_KEY"] == key and "QUOTED" in E.read_env(env))
E.write_setting("ANTHROPIC_API_KEY", key + "Z", env)
check("writing again replaces, not duplicates",
      open(env).read().count("ANTHROPIC_API_KEY") == 1)
if os.name == "posix":
    check("the file is readable by this user only",
          (os.stat(env).st_mode & 0o077) == 0, oct(os.stat(env).st_mode))
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["MEMGATE_SCORER"] = "learned"
applied = E.load_env(env)
check("load_env sets the key but never overrides the terminal",
      os.environ["ANTHROPIC_API_KEY"] == key + "Z"
      and os.environ["MEMGATE_SCORER"] == "learned"
      and "MEMGATE_SCORER" not in applied and "EMPTY" not in applied)
check("keys are recognised by shape",
      E.looks_like_key(key) and not E.looks_like_key("hello")
      and not E.looks_like_key("sk-ant-short"))
check(".env sits at the top of the repo", E.ENV_PATH == os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import subprocess
ign = subprocess.run(["git", "check-ignore", "-q", E.ENV_PATH],
                     cwd=os.path.dirname(E.ENV_PATH))
check(".env is gitignored, so the key can't be committed", ign.returncode == 0)

R._importable, R._apple_silicon, R._hf_ready, R._mlx_ready = real
for k, v in saved_env.items():
    os.environ.pop(k, None)
    if v is not None:
        os.environ[k] = v

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
