# MemGate as a Hugging Face Space (Docker SDK). The Space holds only this file,
# fetch.py and README.md; the code is downloaded from GitHub at build time, so a
# rebuild picks up whatever is on the branch.
#
# The chat model is Claude (the Space secret ANTHROPIC_API_KEY): a free Space
# has no GPU, and the local Qwen would take minutes per reply on its CPU.
# Without the key the app still runs and shows memory and context only.
FROM python:3.11-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/home/user/tiktoken
WORKDIR /home/user

# What the app needs online. MiniLM runs on onnxruntime instead of torch:
# the same weights (memgate/utils.py, _OnnxMiniLM), a ~1 GB smaller image.
RUN pip install --no-cache-dir --user numpy tiktoken onnxruntime tokenizers anthropic

# The exact token counter the experiments used, fetched once at build.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

ARG REPO=Nixx5555/Mem
ARG BRANCH=endtask-full-run
# Changes whenever the branch moves, so a rebuild never reuses a stale clone.
ADD --chown=user https://api.github.com/repos/${REPO}/git/refs/heads/${BRANCH} /home/user/branch.json
COPY --chown=user fetch.py /home/user/fetch.py
# The code, all-MiniLM-L6-v2 exported to ONNX (checked byte for byte), and
# LoCoMo for the "Benchmark check" button.
RUN python /home/user/fetch.py ${REPO} ${BRANCH}

WORKDIR /home/user/mem/memgate

# Fail the build, not the demo, if the embedder would fall back to hashing.
RUN python -c "from memgate.utils import embed, backend_info; embed('x'); \
b = backend_info()['embedder']; print(b); assert b.startswith('all-MiniLM'), b"

# Public link: every visitor gets their own chats. Add the Space secret
# MEMGATE_PASSCODE to ask for a passcode first.
ENV HOST=0.0.0.0 PORT=7860 MEMGATE_PUBLIC=1 MEMGATE_LLM=claude
EXPOSE 7860
CMD ["python", "app.py", "--no-browser"]
