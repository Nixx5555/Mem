"""A MiniLM cache for the 10x retrain that survives a restart.

retrain_scorer.py kept its 40 minutes of embeddings in one pickle written at
the end. The 10x set (retrain_10x.py) is about 600,000 turns, a third of them
BEAM's long markdown answers, which MiniLM embeds at ~35 a second on 4 cores:
hours, not minutes. So the work is cut into shards of SHARD strings, each
written as soon as it is done. A restarted run loads every finished shard and
embeds only what is left.

Strings are embedded in order of length, so a batch pads to its own longest
string rather than to 256 tokens. Mean pooling is over the attention mask, so
the vectors are the ones an unsorted run gives, up to float rounding.
"""
import glob
import os
import pickle
import time

import numpy as np

from memgate import utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "data", "cache", "minilm_10x")
SHARD = 20000


def load_shards(cache_dir=CACHE_DIR):
    n = 0
    for fn in sorted(glob.glob(os.path.join(cache_dir, "shard_*.pkl"))):
        try:
            with open(fn, "rb") as f:
                texts, vecs = pickle.load(f)
        except Exception:                      # a shard cut off mid-write
            os.remove(fn)
            continue
        for t, v in zip(texts, vecs):
            U._cache[t] = v
        n += len(texts)
    return n


def warm(texts, cache_dir=CACHE_DIR, log=print):
    """Embed every string not cached yet, writing a shard per SHARD strings."""
    os.makedirs(cache_dir, exist_ok=True)
    load_shards(cache_dir)
    todo = sorted({t for t in texts if t not in U._cache}, key=len)
    if not todo:
        return 0
    model = U._get_model()
    if model is None:
        raise SystemExit("needs MiniLM")
    done = len(glob.glob(os.path.join(cache_dir, "shard_*.pkl")))
    t0 = time.time()
    for i in range(0, len(todo), SHARD):
        part = todo[i:i + SHARD]
        vecs = np.asarray(model.encode(part, normalize_embeddings=True,
                                       batch_size=64), dtype=np.float32)
        fn = os.path.join(cache_dir, f"shard_{done:05d}.pkl")
        with open(fn + ".tmp", "wb") as f:
            pickle.dump((part, vecs), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(fn + ".tmp", fn)
        done += 1
        for t, v in zip(part, vecs):
            U._cache[t] = v
        k = i + len(part)
        rate = k / (time.time() - t0)
        log(f"  embedded {k:,}/{len(todo):,} ({rate:.0f}/s, "
            f"~{(len(todo) - k) / rate / 60:.0f} min left)")
    return len(todo)
