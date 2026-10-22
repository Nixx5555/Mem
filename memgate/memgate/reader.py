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
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"Gemini HTTP {status}: {message}")
        self.status = status


class RateLimited(GeminiError):
    pass


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


def _gemini_post(model, key, payload, timeout=60.0):
    """POST to generateContent (or, with payload None, GET the model's info,
    which checks the key without using any of the free quota). Returns the
    JSON reply; raises RateLimited on 429 (with .wait) and GeminiError on any
    other HTTP error."""
    url = GEMINI_URL.format(model=model)
    if payload is None:
        url = url.rsplit(":", 1)[0]
    req = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
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
        raise GeminiError(exc.code, msg) from None


def _key_rejected(exc):
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    return (isinstance(exc, GeminiError) and not isinstance(exc, RateLimited)
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
             f"Memory notes:\n{notes or '(empty)'}\n\nMessage: {message}"}]


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
                _gemini_post(self.path, key, None, timeout=15.0)
            except RateLimited:
                pass                      # the key works; the limit passes
            except GeminiError as exc:
                if _key_rejected(exc):
                    raise RuntimeError(
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
                if self.note and self.note.startswith("Gemini's free-tier"):
                    self.note = None
                return out
            except RateLimited:
                # Free-tier limit: temporary, so Gemini stays the main model.
                # This one message is answered locally when Qwen is
                # downloaded, else the page says to wait.
                if self.fallback:
                    if self._spare is None:
                        self._spare = ChatModel(self.fallback)
                    self.note = ("Gemini's free-tier limit was reached; the "
                                 "local model answered this message")
                    print(f"  [MemGate] {self.note}", flush=True)
                    return self._spare._reply(notes, message)
                return ("(Gemini's free-tier limit was reached. Wait about a "
                        "minute and send the message again. MemGate has "
                        "already saved it to memory.)")
            except Exception as exc:
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
                    max_new_tokens=512, do_sample=False)
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

    def _gemini(self, msgs):
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user",
                          "parts": [{"text": msgs[1]["content"]}]}],
            "generationConfig": {"maxOutputTokens": 4096},
        }
        # With Qwen downloaded, hand over sooner rather than keep the page
        # waiting; without it, wait out the limit a little longer.
        budget = 5 if self.fallback else GEMINI_WAIT
        waited = 0.0
        while True:
            try:
                r = _gemini_post(self.path, self._client, payload)
                break
            except RateLimited as exc:
                wait = min(max(exc.wait, 1.0), budget - waited)
                if wait <= 0:
                    raise
                print(f"  [MemGate] Gemini rate limit; retrying in "
                      f"{wait:.0f}s", flush=True)
                time.sleep(wait)
                waited += wait
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
