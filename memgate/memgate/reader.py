"""The chat model that answers from MemGate's context.

The model only ever sees the memory notes MemGate assembled plus the current
message, never the raw history. That is the point of the system, so every
backend here takes the same two strings.

Backends, chosen with MEMGATE_LLM or `app.py --llm`:

    auto    Claude when ANTHROPIC_API_KEY is set (setup can save it in .env),
            else Gemini when GEMINI_API_KEY is set, else mlx on an
            Apple-silicon Mac, else transformers, else none
    mlx     mlx_lm, the reader the end-task run used (Apple silicon only)
    hf      transformers on CPU/GPU, any OS
    claude  the Anthropic API; needs ANTHROPIC_API_KEY and internet
    gemini  the Google Gemini API (free tier works); needs GEMINI_API_KEY
            and internet, and no extra package
    none    no model: replies say what the model would have been sent

`load()` raises if the chosen backend cannot start; the caller decides whether
that is fatal.
"""
import json
import os
import platform
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SYSTEM = ("You are a helpful assistant in a long-running chat. You do not see "
          "the conversation history. You only see the memory notes below, "
          "which a memory system selected for this message. Use them to "
          "answer, and reply naturally to everything else. If the user asks "
          "about something the notes do not contain, say you don't remember "
          "it rather than guessing.")

BACKENDS = ("auto", "mlx", "hf", "claude", "gemini", "none")

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HF_LOCAL = ("Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct")
HF_HUB = "Qwen/Qwen2.5-1.5B-Instruct"
CLAUDE_MODEL = "claude-opus-5-5"
# Google's alias for its current Flash model, which the free tier includes.
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")
# The free tier allows only a few requests a minute. A 429 is retried after
# Google's suggested wait, up to this many seconds in total.
GEMINI_WAIT = 20
# Other free-tier models to ask when the main one is overloaded (503) or
# rate limited: each has its own quota and its own load.
GEMINI_BACKUPS = ("gemini-flash-lite-latest", "gemini-2.5-flash",
                  "gemini-2.5-flash-lite")
REMOTE = ("claude", "gemini")


def _importable(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def resolve(backend):
    backend = (backend or "auto").lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown model backend {backend!r}; one of {BACKENDS}")
    if backend != "auto":
        return backend
    if _importable("anthropic") and _has_key():
        return "claude"
    if _gemini_key():
        return "gemini"
    return _local() or "none"


def _has_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _gemini_key():
    from memgate.envfile import clean_key
    return clean_key(os.environ.get("GEMINI_API_KEY")
                     or os.environ.get("GOOGLE_API_KEY") or "")


class GeminiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"Gemini HTTP {status}: {message}")
        self.status = status


class KeyRejected(RuntimeError):
    """The provider refused the API key: the one failure that lasts."""


class Temporary(GeminiError):
    """Google is busy or the quota is used up: try again soon, or elsewhere."""
    wait = 2.0


class RateLimited(Temporary):
    pass


class Busy(Temporary):
    """A 5xx, such as 503 "the model is experiencing high demand"."""


def _retry_delay(body, headers):
    """Seconds Google asks us to wait after a 429, or a short default."""
    try:
        for d in json.loads(body)["error"].get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return 5.0


# Ways to send the key. Google documents the header; the newer AQ. keys
# have been refused by some endpoints that way (a Google bug it has
# acknowledged), so the others are tried before a key is called invalid.
AUTH_MODES = ("header", "query", "bearer")
_auth_ok = {}                   # key -> the way that worked


def _gemini_post(model, key, payload, timeout=60.0):
    """POST to generateContent (or, with payload None, GET the model's info,
    which checks the key without using any of the free quota). Returns the
    JSON reply; raises RateLimited on 429 (with .wait) and GeminiError on any
    other HTTP error, after trying each way of sending the key."""
    last = None
    for mode in ([_auth_ok[key]] if key in _auth_ok else AUTH_MODES):
        try:
            r = _gemini_once(model, key, payload, timeout, mode)
        except GeminiError as exc:
            if isinstance(exc, Temporary) or not _key_rejected(exc):
                raise
            last = exc
            continue
        _auth_ok[key] = mode
        return r
    raise last


def _gemini_once(model, key, payload, timeout, mode):
    url = GEMINI_URL.format(model=model)
    if payload is None:
        url = url.rsplit(":", 1)[0]
    headers = {"Content-Type": "application/json"}
    if mode == "header":
        headers["x-goog-api-key"] = key
    elif mode == "query":
        url += "?" + urllib.parse.urlencode({"key": key})
    else:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            msg = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = body[:200]
        if exc.code == 429:
            err = RateLimited(429, msg)
            err.wait = _retry_delay(body, exc.headers)
            raise err from None
        if exc.code >= 500:
            raise Busy(exc.code, msg) from None
        raise GeminiError(exc.code, msg) from None


def _key_rejected(exc):
    if isinstance(exc, KeyRejected) or type(exc).__name__ in (
            "AuthenticationError", "PermissionDeniedError"):
        return True
    return (isinstance(exc, GeminiError) and not isinstance(exc, Temporary)
            and (exc.status in (401, 403) or "API key" in str(exc)))


def _local(ready_only=False):
    """The local backend this laptop can run, or None. With ready_only, only
    one whose weights are already downloaded (so it needs no internet)."""
    if _apple_silicon() and _importable("mlx_lm"):
        if not ready_only or _mlx_ready():
            return "mlx"
    if _importable("transformers") and _importable("torch"):
        if not ready_only or _hf_ready():
            return "hf"
    return None


def _mlx_ready():
    from memgate.llm_judge import LOCAL_CANDIDATES
    return any(os.path.isfile(os.path.join(_MODELS_DIR, n, "config.json"))
               for n in LOCAL_CANDIDATES)


def _hf_ready():
    """A local Qwen copy is downloaded (so `hf` won't need the internet)."""
    return any(os.path.isfile(os.path.join(_MODELS_DIR, n, "config.json"))
               for n in HF_LOCAL)


def _hf_path():
    for name in HF_LOCAL:
        d = os.path.join(_MODELS_DIR, name)
        if os.path.isfile(os.path.join(d, "config.json")):
            return d
    return os.environ.get("MEMGATE_HF_MODEL", HF_HUB)


def _path_for(backend):
    if backend == "mlx":
        from memgate.llm_judge import default_model
        return default_model()
    if backend == "hf":
        return os.environ.get("MEMGATE_HF_MODEL") or _hf_path()
    if backend == "claude":
        return os.environ.get("MEMGATE_CLAUDE_MODEL", CLAUDE_MODEL)
    if backend == "gemini":
        return os.environ.get("MEMGATE_GEMINI_MODEL", GEMINI_MODEL)
    return None


def _prompt(notes, message):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Memory notes (said earlier in this chat):\n{notes or '(empty)'}"
             f"\n\nThe user's new message. Reply to this one:\n{message}"}]


class ChatModel:
    def __init__(self, backend="auto"):
        self.backend = resolve(backend)
        self.lock = threading.Lock()
        self._model = self._tok = self._client = None
        self.path = _path_for(self.backend)
        # Claude and Gemini need the internet. If one fails mid-chat (no
        # Wi-Fi at the venue, an expired key), answer with the downloaded
        # local model instead of showing an error.
        self.fallback = (_local(ready_only=True)
                         if self.backend in REMOTE else None)
        self.note = None        # why the backend changed, for the page
        self._spare = None      # the local model, while Gemini is rate limited

    def _fall_back(self, exc):
        why = f"{type(exc).__name__}: {str(exc)[:120]}"
        who = self.backend.capitalize()
        self.backend, self.path = self.fallback, _path_for(self.fallback)
        self.fallback, self._client = None, None
        self.note = f"{who} failed ({why}); switched to the local model"
        print(f"  [MemGate] {self.note}", flush=True)

    @property
    def enabled(self):
        return self.backend != "none"

    @property
    def name(self):
        if not self.enabled:
            return "none (memory only)"
        return f"{os.path.basename(self.path.rstrip('/'))} ({self.backend})"

    def disable(self):
        self.backend, self.path = "none", None

    def load(self):
        if not self.enabled or self._model is not None or self._client is not None:
            return
        if self.backend == "mlx":
            from mlx_lm import load
            self._model, self._tok = load(self.path)
        elif self.backend == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.path, torch_dtype="auto")
            self._model.eval()
        elif self.backend == "claude":
            import anthropic
            if not (os.environ.get("ANTHROPIC_API_KEY")
                    or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(max_retries=2, timeout=60.0)
        elif self.backend == "gemini":
            key = _gemini_key()
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            # Check the key and the internet now, at start, so a bad key or
            # no Wi-Fi switches to the local model before the first message
            try:
                try:
                    _gemini_post(self.path, key, None, timeout=15.0)
                except GeminiError as exc:
                    if isinstance(exc, Temporary) or not _key_rejected(exc):
                        raise
                    # the model-info call refused it; a one-token reply is
                    # the final word before giving up on the key
                    _gemini_post(self.path, key, {
                        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                        "generationConfig": {"maxOutputTokens": 1}},
                        timeout=20.0)
            except Temporary:
                pass                      # the key works; Google is just busy
            except GeminiError as exc:
                if _key_rejected(exc):
                    raise KeyRejected(
                        f"Gemini rejected the key ({exc}). Check it at "
                        "https://aistudio.google.com/apikey and save it with  "
                        "python set_key.py") from None
                raise
            self._client = key

    def reply(self, notes, message):
        if not self.enabled:
            return "(no model loaded: the context panel shows what it would see)"
        with self.lock:
            try:
                out = self._reply(notes, message)
                if self.note and self.note.startswith("Gemini"):
                    self.note = None
                return out
            except Exception as exc:
                if self.backend == "gemini" and not _key_rejected(exc):
                    return self._gemini_failed(exc, notes, message)
                if self.backend not in REMOTE:
                    raise
                if not self.fallback:
                    if _key_rejected(exc):
                        raise RuntimeError(
                            f"{self.backend.capitalize()} rejected the saved "
                            "API key. Replace it with  python set_key.py  "
                            "(inside memgate/) and restart. "
                            f"({str(exc)[:200]})") from exc
                    raise
                self._fall_back(exc)
                return self._reply(notes, message)

    def _reply(self, notes, message):
        self.load()
        msgs = _prompt(notes, message)
        if self.backend == "mlx":
            from mlx_lm import generate
            prompt = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            out = generate(self._model, self._tok, prompt=prompt,
                           max_tokens=512, verbose=False)
        elif self.backend == "hf":
            import torch
            # transformers 5 returns a dict-like BatchEncoding here, 4.x a
            # bare tensor; return_dict=True gives the dict on both
            enc = self._tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
            ids = enc["input_ids"]
            with torch.no_grad():
                gen = self._model.generate(
                    ids, attention_mask=enc.get("attention_mask"),
                    max_new_tokens=512,
                    # Qwen's own recommended sampling. Greedy decoding made
                    # the 1.5B model copy a line from the notes (its own
                    # earlier "Hello! How can I assist you today?") as the
                    # reply to every message; the penalty covers the prompt
                    # too, so copying the notes word for word costs more.
                    do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                    repetition_penalty=1.1)
            out = self._tok.decode(gen[0, ids.shape[-1]:],
                                   skip_special_tokens=True)
        elif self.backend == "gemini":
            out = self._gemini(msgs)
        else:
            # fallbacks="default": a policy decline is retried server-side
            # on a fallback model instead of returning an empty reply
            r = self._client.beta.messages.create(
                model=self.path, max_tokens=1024, system=SYSTEM,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=msgs[1:])
            if r.stop_reason == "refusal":
                return "(the model declined to answer this message)"
            out = "".join(b.text for b in r.content if b.type == "text")
        return out.strip()

    def _gemini_failed(self, exc, notes, message):
        """Gemini couldn't answer this message (busy, quota, no internet).
        That passes, so Gemini stays the main model: this one message is
        answered locally when Qwen is downloaded, else the page says to
        send it again. Only a rejected key switches models for good."""
        why = (f"Gemini {exc.status}" if isinstance(exc, GeminiError)
               else f"no connection to Gemini ({type(exc).__name__})")
        print(f"  [MemGate] {why}: {str(exc)[:200]}", flush=True)
        if self.fallback:
            if self._spare is None:
                self._spare = ChatModel(self.fallback)
            self.note = (f"Gemini was busy or unreachable ({why}); the local "
                         f"model answered this message")
            return self._spare._reply(notes, message)
        if isinstance(exc, RateLimited):
            return ("(Gemini's free-tier limit was reached. Wait about a "
                    "minute and send the message again. MemGate has "
                    "already saved it to memory.)")
        return (f"(Gemini couldn't answer just now: {why}. Send the message "
                "again in a moment. MemGate has already saved it to memory.)")

    def _gemini(self, msgs):
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user",
                          "parts": [{"text": msgs[1]["content"]}]}],
            "generationConfig": {"maxOutputTokens": 4096},
        }
        r = self._gemini_any(payload)
        if (r.get("promptFeedback") or {}).get("blockReason"):
            return "(the model declined to answer this message)"
        cands = r.get("candidates") or []
        if not cands:
            return "(the model returned no reply)"
        parts = (cands[0].get("content") or {}).get("parts") or []
        out = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not out.strip() and cands[0].get("finishReason") == "SAFETY":
            return "(the model declined to answer this message)"
        return out

    def _gemini_any(self, payload):
        """Ask the main model, waiting briefly when it is busy or limited,
        then the other free-tier models in turn. Raises the last Temporary
        error when none of them could answer."""
        # With Qwen downloaded, hand over sooner rather than keep the page
        # waiting; without it, wait a little longer.
        budget = 6 if self.fallback else GEMINI_WAIT
        waited, last = 0.0, None
        models = [self.path] + [m for m in GEMINI_BACKUPS if m != self.path]
        for i, model in enumerate(models):
            for attempt in range(2 if i == 0 else 1):
                try:
                    r = _gemini_post(model, self._client, payload)
                    if model != self.path:
                        print(f"  [MemGate] {model} answered while "
                              f"{self.path} was busy", flush=True)
                    return r
                except Temporary as exc:
                    last = exc
                    wait = min(max(exc.wait, 1.0), budget - waited)
                    if attempt or i or wait <= 0:
                        break             # on to the next model
                    print(f"  [MemGate] Gemini {exc.status} (busy or "
                          f"limited); retrying in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    waited += wait
                except GeminiError as exc:
                    if exc.status == 404 and i:
                        break             # that backup model is retired
                    raise
        raise last
