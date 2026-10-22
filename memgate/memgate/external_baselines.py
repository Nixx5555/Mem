"""The memory designs a jury will name: MemGPT and Mem0, at the same budget.

Every other baseline in this project is internal (window, RAG, compression),
so "how does this compare to MemGPT / Mem0?" had no number behind it. These
two policies rebuild each system's MEMORY-MANAGEMENT pattern inside the same
harness, under the same store budget S and context budget, so the comparison
is the policy and nothing else.

What is copied and what is not:

  MemGPT (Packer et al., 2023) keeps a FIFO queue of recent messages in main
  context, folds messages evicted from it into a recursive summary that also
  lives in main context, and pages old messages to archival storage that it
  searches on demand. Here: archival storage is the verbatim store (evicted
  FIFO when S binds), the summary takes a fixed share of the budget and is
  rebuilt from what the archive drops, oldest detail lost first, exactly as a
  recursive summary loses it.

  Mem0 (Chhikara et al., 2025) extracts salient facts from each exchange and
  runs ADD / UPDATE / NOOP against the nearest stored memories, so the store
  holds one current version of each fact. Here: a turn with no content is a
  NOOP, a turn close to a stored memory UPDATEs it (newest wins), the rest are
  ADDed, and the oldest-touched memory goes first when S binds.

  NOT copied: the LLM calls. Both systems ask a chat model to write the
  summary or the fact; this container cannot reach one, and an LLM writer
  would also change the reader-independent metric this project uses. The
  summary and the fact are written with MemGate's own extractive compressor
  (compress.py), so every policy here uses the same text tools. That makes
  these "-style" baselines: the structure of each system, not its prompts.
  Their knobs were tuned on LoCoMo itself (run_external_baselines.py --tune),
  which favours the baselines, not MemGate.
"""
from typing import List

from .types import Turn, MemoryItem, ContextBundle, WORKING, LONG_TERM
from .utils import count_tokens, embed, cosine
from .compress import compress
from .store import rank
from .scoring import FILLER_PAT, words
from .policies import _as_item, _pack, _item_cost


def _read(items, recent_items, query, budget, split, k, head=None):
    """The read path every retrieval policy here shares with RAGPolicy: a
    recent share, then retrieved items, with an optional pinned head first."""
    out, used = [], 0
    if head is not None and head.text:
        c = count_tokens(head.text)
        if c <= budget:
            out.append(head)
            used = c
    b_recent = int((budget - used) * split[0])
    recent = _pack(list(reversed(recent_items)), b_recent)
    seen = {id(i) for i in recent.items}
    hits = rank(query, [i for i in items if id(i) not in seen], k)
    long_ = _pack(hits, budget - used - recent.tokens)
    out += recent.items + long_.items
    text = "\n".join(i.text for i in out)
    return ContextBundle(text=text, items=out,
                         tokens=used + recent.tokens + long_.tokens,
                         budget=budget)


class MemGPTStylePolicy:
    """Recent queue + recursive summary + searchable archive, under one cap.

    The summary is part of the store (it costs storage like anything else)
    and is always placed in the context, as MemGPT keeps it in main context.
    Its size is a share of the smaller of the two budgets, because a summary
    that cannot fit in the context is not one MemGPT could hold.
    """
    name = "MemGPT-style"
    is_baseline = False

    def __init__(self, store_budget: int = None, budget: int = 2048,
                 summary_share: float = 0.25, gist_tokens: int = 16,
                 retrieve_k: int = 60, split=(0.25, 0.75),
                 cost_mode: str = "tokens", vector_bytes: int = 1536, **kw):
        self.store_budget = store_budget
        self.cost_mode = cost_mode
        self.vector_bytes = vector_bytes
        self.retrieve_k = retrieve_k
        self.split = split
        self.gist_tokens = gist_tokens
        cap = budget if store_budget is None else min(budget, store_budget)
        self.summary_budget = int(cap * summary_share)
        self.archive: List[MemoryItem] = []
        self._tokens = 0
        # (evidence id, gist) pairs, oldest first. A fragment is kept whole or
        # dropped with its id, never kept as an id whose text is gone.
        self.fragments: List[tuple] = []
        self._summary_tokens = 0
        self.summary = MemoryItem(text="", tier=WORKING, kind="summary",
                                  session=0, turn_index=0)
        self.dropped = 0

    def _cost(self, it):
        return _item_cost(it, self.cost_mode, self.vector_bytes)

    def observe(self, turn: Turn):
        it = _as_item(turn)
        it.embedding = embed(it.text)
        self.archive.append(it)
        self._tokens += self._cost(it)
        while self.store_budget is not None and len(self.archive) > 1 and \
                self._tokens + self._summary_tokens > self.store_budget:
            old = self.archive.pop(0)
            self._tokens -= self._cost(old)
            self._fold(old)

    def _fold(self, it: MemoryItem):
        """Fold an evicted message into the summary; the oldest detail goes
        first when the summary is over its share."""
        gist = compress(it.text, self.gist_tokens)
        if gist and self.summary_budget > 0:
            self.fragments.append((it.fact_id, gist))
        while self.fragments and \
                count_tokens("\n".join(g for _, g in self.fragments)) > self.summary_budget:
            self.fragments.pop(0)
            self.dropped += 1
        self.summary.text = "\n".join(g for _, g in self.fragments)
        self.summary.covered_ids = [f for f, _ in self.fragments if f]
        self._summary_tokens = count_tokens(self.summary.text) if self.fragments else 0

    def build_context(self, query, budget: int) -> ContextBundle:
        return _read(self.archive, self.archive, query, budget, self.split,
                     self.retrieve_k, head=self.summary)

    def stored_tokens(self) -> int:
        return sum(count_tokens(i.text) for i in self.archive) + self._summary_tokens

    def stats(self):
        return {"archive": len(self.archive), "summary_fragments": len(self.fragments),
                "dropped": self.dropped, "stored_tokens": self.stored_tokens()}


class Mem0StylePolicy:
    """Extract a fact per turn; ADD, UPDATE the nearest one, or NOOP."""
    name = "Mem0-style"
    is_baseline = False

    def __init__(self, store_budget: int = None, budget: int = 2048,
                 fact_tokens: int = 48, update_threshold: float = 0.85,
                 retrieve_k: int = 60, split=(0.25, 0.75), recent_n: int = 6,
                 cost_mode: str = "tokens", vector_bytes: int = 1536, **kw):
        self.store_budget = store_budget
        self.cost_mode = cost_mode
        self.vector_bytes = vector_bytes
        self.fact_tokens = fact_tokens
        self.update_threshold = update_threshold
        self.retrieve_k = retrieve_k
        self.split = split
        self.memories: List[MemoryItem] = []   # oldest-touched first
        self.recent: List[MemoryItem] = []     # the raw recent messages
        self.recent_n = recent_n
        self._tokens = 0
        self.ops = {"add": 0, "update": 0, "noop": 0, "evict": 0}

    def _cost(self, it):
        return _item_cost(it, self.cost_mode, self.vector_bytes)

    def observe(self, turn: Turn):
        raw = _as_item(turn)
        # The last few raw messages, which Mem0's caller passes alongside the
        # retrieved memories. They are held, so they are charged to S.
        self.recent.append(raw)
        self._tokens += self._cost(raw)
        if len(self.recent) > self.recent_n:
            self._tokens -= self._cost(self.recent.pop(0))
        text = turn.text.strip()
        if FILLER_PAT.match(text) or len(words(text)) <= 2:
            self.ops["noop"] += 1
            return
        fact = MemoryItem(text=compress(raw.text, self.fact_tokens),
                          tier=LONG_TERM, kind="fact", session=turn.session,
                          turn_index=turn.index, fact_id=turn.fact_id,
                          speaker=turn.speaker)
        fact.embedding = embed(fact.text)
        best, sim = None, -1.0
        for m in self.memories:
            s = cosine(fact.embedding, m.embedding)
            if s > sim:
                best, sim = m, s
        if best is not None and sim >= self.update_threshold:
            # UPDATE: the newer statement replaces the older one, as Mem0
            # keeps one current version of a fact. The old text is gone, so
            # its evidence id goes with it.
            self.memories.remove(best)
            self._tokens -= self._cost(best)
            self.ops["update"] += 1
        else:
            self.ops["add"] += 1
        self.memories.append(fact)
        self._tokens += self._cost(fact)
        while self.store_budget is not None and len(self.memories) > 1 and \
                self._tokens > self.store_budget:
            self._tokens -= self._cost(self.memories.pop(0))
            self.ops["evict"] += 1

    def build_context(self, query, budget: int) -> ContextBundle:
        return _read(self.memories, self.recent, query, budget, self.split,
                     self.retrieve_k)

    def stored_tokens(self) -> int:
        return sum(count_tokens(i.text) for i in self.memories + self.recent)

    def stats(self):
        return dict(self.ops, held=len(self.memories),
                    stored_tokens=self.stored_tokens())
