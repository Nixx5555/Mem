#!/usr/bin/env python3
"""Stress test 3: a document uploaded in the middle of a chat.

For each LoCoMo conversation, a document of BEAM long-form text (8,000 or
20,000 tokens) is inserted halfway, and the LoCoMo questions are asked at
S = 4096 through the app path, scored by the previous shipped scorer and the
current one. See docs/STRESS_TEST.md section 6.

    git show <commit>:memgate/memgate/scorer_pooled_lr.json > /tmp/old.json
    python3 stress_upload_midchat.py /tmp/old.json
"""
import random, sys
import numpy as np
from memgate.data import load_locomo
from memgate.harness import run_policy
from memgate.ingest import GuardedScorer, chunk
from memgate.policies import MemGateSelectPolicy
from memgate.pretrained import load_learned
from memgate.types import Turn
from memgate.utils import warm_cache
from stress_questions import AppPath
from stress_inputs import haystack_paragraphs
OLD = sys.argv[1]
models = {"old": load_learned(OLD), "new": load_learned()}
data = load_locomo()
paras = haystack_paragraphs(); rnd = random.Random(1)
for doc_tokens in (0, 8000, 20000):
    res = {k: [] for k in models}
    for sid, ts, qs in data:
        doc = []
        n = 0
        while n < doc_tokens:
            p = rnd.choice(paras); doc.append(p); n += len(p.split()) * 1.3
        mid = len(ts) // 2
        ins = [Turn(session=ts[mid].session, index=ts[mid].index, speaker="file report.txt",
                    text=p, fact_id=f"doc-{sid}-{i}") for i, p in enumerate(doc)]
        turns = ts[:mid] + ins + ts[mid:]
        warm_cache([f"[{t.speaker}] {x}" for t in turns for x in chunk(t.text)])
        for k, m in models.items():
            pol = AppPath(MemGateSelectPolicy(budget=2048, store_budget=4096, scorer=GuardedScorer(m)))
            r = run_policy(pol, turns, qs, 2048)
            res[k] += [d["answer"] for d in r["detail"] if d["answer"] is not None]
    print(f"doc {doc_tokens:>6} tok mid-chat:", {k: round(np.mean(v) * 100, 1) for k, v in res.items()}, flush=True)
