# Putting MemGate online

The app runs as a free Hugging Face Space, so the jury can open a link on any
laptop or phone. Replies come from Claude (an Anthropic API key) or Gemini (a
free Google key): a free Space has no GPU, and the local Qwen would take
minutes per reply on its CPU.

What the Space does, from `deploy/huggingface/`:

- `Dockerfile` installs the app's packages (MiniLM runs on onnxruntime, so no
  torch), downloads the code of the `endtask-full-run` branch from GitHub, the
  ONNX copy of all-MiniLM-L6-v2 (checksummed) and LoCoMo, and starts `app.py`.
- `MEMGATE_PUBLIC=1`: every visitor gets their own chats, so two people on the
  link never see or change each other's conversations.
- The build fails if MiniLM would not load, rather than serving a demo that
  quietly fell back to the hashing embedder.

## Steps (about 10 minutes, once)

1. **A key.** Either Gemini, free: https://aistudio.google.com/apikey →
   Create API key (it starts with `AQ.`, or `AIza` for older keys). Or Anthropic: at
   https://console.anthropic.com, add a little credit under Billing, then
   create a key under API keys (it starts with `sk-ant-`).
2. **Hugging Face account.** Sign up at https://huggingface.co/join (free).
3. **New Space.** https://huggingface.co/new-space: name it `memgate`, pick
   **Docker** → **Blank**, hardware **CPU basic (free)**, visibility **Public**.
4. **Upload the three files.** In the Space, Files → Add file → Upload files,
   and upload `Dockerfile`, `README.md` and `fetch.py` from
   `deploy/huggingface/`. Commit. (The README replaces the Space's default one;
   its header tells Hugging Face this is a Docker app on port 7860.)
5. **Secrets.** Space Settings → Variables and secrets → New secret:
   - `GEMINI_API_KEY` = your Gemini key, or `ANTHROPIC_API_KEY` = your
     Anthropic key (one is required for replies)
   - `MEMGATE_PASSCODE` = a word you give the jury (recommended, so strangers
     who find the link can't spend your credit)
6. **Wait for the build** (about 3 minutes; the Space shows "Building", then
   "Running"). Open the direct link:

       https://<your-hf-username>-memgate.hf.space

   Use this link rather than the huggingface.co/spaces page: the app gets the
   whole screen, and the passcode is remembered.

## Good to know

- **Before the review, open the link once.** A free Space sleeps after 48
  hours without visitors and takes about a minute to wake.
- **Chats reset when the Space restarts** (a rebuild, or after sleeping). Use
  "Load sample chat" to get a ready conversation in one click.
- **New code:** after a change is merged into `endtask-full-run`, Space
  Settings → Factory rebuild picks it up.
- **Gemini free tier:** a few replies a minute. If several jury members type
  at once, a reply may ask them to wait a minute; memory is still saved.
- **Cost (Claude):** each reply is one Claude call with at most about 2,000 tokens of
  memory in and 1,024 out. To spend less, add the variable
  `MEMGATE_CLAUDE_MODEL` = `claude-sonnet-5-5` (or `claude-haiku-5-5`).
- **No key, or the key fails:** the app still runs. Memory, tiers, context and
  the Benchmark check all work; replies say no model is loaded.

## Run the same image on your own machine

    docker build -t memgate deploy/huggingface
    docker run -p 7860:7860 -e GEMINI_API_KEY=AQ.... -e MEMGATE_PASSCODE=jury memgate

then open http://localhost:7860.
