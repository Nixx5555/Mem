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


KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY",
        "GOOGLE_API_KEY", "MEMGATE_GEMINI_MODEL")
saved_env = {k: os.environ.pop(k, None) for k in KEYS}
real = (R._importable, R._apple_silicon, R._hf_ready, R._mlx_ready,
        R._gemini_post, R.time.sleep)


def world(mac=False, mlx=False, torch=False, anthropic=True, hf_ready=False,
          mlx_ready=False, key=False, gemini=False):
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
    if gemini:
        os.environ["GEMINI_API_KEY"] = "AIza" + "y" * 35
    else:
        os.environ.pop("GEMINI_API_KEY", None)


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

print("\n-- the Gemini backend (faked) --")
world(gemini=True, mac=True, mlx=True, mlx_ready=True)
check("a Gemini key picks Gemini over the local model", R.resolve("auto") == "gemini")
world(key=True, gemini=True)
check("with both keys, Claude still comes first", R.resolve("auto") == "claude")
world(gemini=True, anthropic=False)
check("Gemini needs no extra package", R.resolve("auto") == "gemini")
world()
os.environ["GOOGLE_API_KEY"] = "AIza" + "z" * 35
check("GOOGLE_API_KEY counts as a Gemini key", R.resolve("auto") == "gemini")
del os.environ["GOOGLE_API_KEY"]
world(gemini=True)
m = R.ChatModel("gemini")
check("Gemini defaults to the free-tier Flash model",
      m.path == R.GEMINI_MODEL and "flash" in m.path)
os.environ["MEMGATE_GEMINI_MODEL"] = "gemini-2.5-flash-lite"
check("MEMGATE_GEMINI_MODEL overrides the model",
      R.ChatModel("gemini").path == "gemini-2.5-flash-lite")
del os.environ["MEMGATE_GEMINI_MODEL"]
sent = []


def gemini_says(*replies):
    """Fake _gemini_post: each call takes the next reply (a dict or an error)."""
    queue = list(replies)

    def post(model, key, payload, timeout=60.0):
        sent.append((model, key, payload))
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(r, Exception):
            raise r
        return r
    R._gemini_post = post


def ok(text):
    return {"candidates": [{"content": {"parts": [
        {"text": "plan", "thought": True}, {"text": text}]},
        "finishReason": "STOP"}]}


def limited(wait=1.0):
    e = R.RateLimited(429, "Resource has been exhausted")
    e.wait = wait
    return e


slept = []
R.time.sleep = slept.append
gemini_says(ok("  The exam is on 8 November.\n"))
m = R.ChatModel("gemini")
out = m.reply("[user] My exam is on 8 November.", "When is my exam?")
model, key, payload = sent[-1]
check("Gemini's reply comes back trimmed, thoughts left out",
      out == "The exam is on 8 November.", repr(out))
check("the key and model are sent", key.startswith("AIza") and model == R.GEMINI_MODEL)
check("the system prompt goes in systemInstruction",
      payload["systemInstruction"]["parts"][0]["text"] == R.SYSTEM
      and [c["role"] for c in payload["contents"]] == ["user"]
      and "8 November" in payload["contents"][0]["parts"][0]["text"])
gemini_says({"promptFeedback": {"blockReason": "SAFETY"}})
check("a blocked prompt is shown as declined", "declined" in m.reply("", "x"))
gemini_says(limited(3), ok("after the wait"))
slept.clear()
check("a 429 is retried after Google's suggested wait",
      m.reply("", "x") == "after the wait" and slept == [3])
gemini_says(limited(8))
slept.clear()
out = m.reply("", "x")
check("a lasting 429 with no local model says to wait a minute",
      "free-tier limit" in out and m.backend == "gemini", out)
check("the total wait is capped", sum(slept) <= R.GEMINI_WAIT, str(slept))
world(gemini=True, torch=True, hf_ready=True)
m = R.ChatModel("gemini")
check("with a downloaded Qwen, Gemini has a local fallback", m.fallback == "hf")
gemini_says(limited(30))
local = []
real_hf_reply = R.ChatModel._reply


def fake_local(self, notes, message):
    if self.backend == "hf":
        local.append(message)
        return "local answer"
    return real_hf_reply(self, notes, message)


R.ChatModel._reply = fake_local
out = m.reply("notes", "hello")
check("a lasting 429 answers this message with Qwen but keeps Gemini",
      out == "local answer" and m.backend == "gemini" and local == ["hello"]
      and "free-tier" in (m.note or ""))
gemini_says(ok("gemini again"))
check("the next message goes back to Gemini and clears the note",
      m.reply("", "y") == "gemini again" and m.note is None)
gemini_says(OSError("no internet"))
out = m.reply("notes", "offline?")
check("no internet switches to Qwen for good",
      out == "local answer" and m.backend == "hf" and "Gemini failed" in m.note)
R.ChatModel._reply = real_hf_reply
world(gemini=True)
m = R.ChatModel("gemini")
gemini_says(R.GeminiError(400, "API key not valid. Please pass a valid API key."))
try:
    m.reply("notes", "hello")
    check("a rejected Gemini key says how to replace it", False)
except RuntimeError as exc:
    check("a rejected Gemini key says how to replace it, with Google's reason",
          "Gemini" in str(exc) and "set_key.py" in str(exc)
          and "API key not valid" in str(exc))
world()
try:
    R.ChatModel("gemini").load()
    check("Gemini without a key refuses to load", False)
except RuntimeError as exc:
    check("Gemini without a key refuses to load", "GEMINI_API_KEY" in str(exc))
check("Google's retryDelay is read from a 429 body",
      R._retry_delay('{"error":{"details":[{"retryDelay":"17s"}]}}', {}) == 17.0
      and R._retry_delay("not json", {"Retry-After": "4"}) == 4.0
      and R._retry_delay("", {}) == 5.0)
R._gemini_post, R.time.sleep = real[4], real[5]

print("\n-- the local Qwen backend, transformers 4 and 5 (faked) --")
import contextlib
import numpy as np


class Enc(dict):
    """transformers 5's BatchEncoding: a dict whose missing attributes raise a
    bare AttributeError, the 'Error: AttributeError:' seen on Windows."""
    def __getattr__(self, k):
        raise AttributeError


class Tok:
    def __init__(self):
        self.kw = None

    def apply_chat_template(self, msgs, **kw):
        self.kw = kw
        ids = np.array([[5, 6, 7]])
        if kw.get("return_dict"):
            return Enc(input_ids=ids, attention_mask=np.ones_like(ids))
        return Enc(input_ids=ids)          # v5 ignores the missing flag too

    def decode(self, ids, skip_special_tokens=True):
        return " " + " ".join(str(int(i)) for i in ids) + "\n"


class Model:
    def generate(self, ids, attention_mask=None, **kw):
        self.mask = attention_mask
        return np.concatenate([ids, np.array([[42, 43]])], axis=1)


sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=contextlib.nullcontext))
m = R.ChatModel("hf")
m._tok, m._model = Tok(), Model()
out = m.reply("[user] My birthday is on 19 October.", "hello")
check("Qwen replies under transformers 5 (no AttributeError)", out == "42 43", repr(out))
check("the attention mask is passed on", m._model.mask is not None
      and m._tok.kw.get("return_dict") is True)

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
gk = "AIza" + "Sy-_9" * 7
check("Gemini keys are recognised and routed to GEMINI_API_KEY",
      E.key_name(gk) == "GEMINI_API_KEY" and E.key_name(key) == "ANTHROPIC_API_KEY"
      and E.key_name("AIza-short") is None and E.key_name("hello") is None)
aq = "AQ.Ab8RN6L" + "x_Y-z.9" * 8
check("new AI Studio keys (AQ.) are Gemini keys too",
      E.key_name(aq) == "GEMINI_API_KEY" and E.key_name("AQ.short") is None)
u16 = os.path.join(d, "u16.env")
with open(u16, "w", encoding="utf-16") as fh:   # PowerShell's  echo X=1 > .env
    fh.write("GEMINI_API_KEY=" + aq + "\n")
check("a UTF-16 .env (made by PowerShell echo) is read",
      E.read_env(u16) == {"GEMINI_API_KEY": aq}, str(E.read_env(u16)))
check("memgate/.env is read too, after the main one",
      E.ALT_PATH == os.path.join(os.path.dirname(E.ENV_PATH), "memgate", ".env"))
check(".env sits at the top of the repo", E.ENV_PATH == os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import subprocess
ign = subprocess.run(["git", "check-ignore", "-q", E.ENV_PATH],
                     cwd=os.path.dirname(E.ENV_PATH))
check(".env is gitignored, so the key can't be committed", ign.returncode == 0)

R._importable, R._apple_silicon, R._hf_ready, R._mlx_ready = real[:4]
for k, v in saved_env.items():
    os.environ.pop(k, None)
    if v is not None:
        os.environ[k] = v

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
