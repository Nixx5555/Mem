"""The chat model that answers from MemGate's context.

The model only ever sees the memory notes MemGate assembled plus the current
message, never the raw history. That is the point of the system, so every
backend here takes the same two strings.

Backends, chosen with MEMGATE_LLM or `app.py --llm`:

    auto    Claude when ANTHROPIC_API_KEY is set (setup can save it in .env),
            else mlx on an Apple-silicon Mac, else transformers, else none
    mlx     mlx_lm, the reader the end-task run used (Apple silicon only)
    hf      transformers on CPU/GPU, any OS
    claude  the Anthropic API; needs ANTHROPIC_API_KEY and internet
    none    no model: replies say what the model would have been sent

`load()` raises if the chosen backend cannot start; the caller decides whether
that is fatal.
"""
import os
import platform
import threading

SYSTEM = ("You are a helpful assistant in a long-running chat. You do not see "
          "the conversation history. You only see the memory notes below, "
          "which a memory system selected for this message. Use them to "
          "answer, and reply naturally to everything else. If the user asks "
          "about something the notes do not contain, say you don't remember "
          "it rather than guessing.")

BACKENDS = ("auto", "mlx", "hf", "claude", "none")

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HF_LOCAL = ("Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct")
HF_HUB = "Qwen/Qwen2.5-1.5B-Instruct"
CLAUDE_MODEL = "claude-opus-5-5"


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
    return _local() or "none"


def _has_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


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
        # Claude needs the internet. If it fails mid-chat (no Wi-Fi at the
        # venue, an expired key), answer with the downloaded local model
        # instead of showing an error.
        self.fallback = (_local(ready_only=True)
                         if self.backend == "claude" else None)
        self.note = None        # why the backend changed, for the page

    def _fall_back(self, exc):
        why = f"{type(exc).__name__}: {str(exc)[:120]}"
        self.backend, self.path = self.fallback, _path_for(self.fallback)
        self.fallback, self._client = None, None
        self.note = f"Claude failed ({why}); switched to the local model"
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

    def reply(self, notes, message):
        if not self.enabled:
            return "(no model loaded: the context panel shows what it would see)"
        with self.lock:
            try:
                return self._reply(notes, message)
            except Exception as exc:
                if self.backend != "claude":
                    raise
                if not self.fallback:
                    if type(exc).__name__ in ("AuthenticationError",
                                              "PermissionDeniedError"):
                        raise RuntimeError(
                            "Claude rejected the saved API key. Replace it "
                            "with  python set_key.py  (inside memgate/) and "
                            "restart.") from exc
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
            ids = self._tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                gen = self._model.generate(ids, max_new_tokens=512,
                                           do_sample=False)
            out = self._tok.decode(gen[0, ids.shape[-1]:],
                                   skip_special_tokens=True)
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
