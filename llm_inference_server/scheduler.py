"""Continuous batching: the batch is re-formed at every model step, not once per request.

Static batching waits for a whole batch to finish before admitting anyone; a short answer then
waits for the longest one. Here the scheduler decides **per iteration** which sequences run:

1. Every running sequence decodes one token (if the KV cache cannot give one a slot, the most
   recently admitted sequence is *preempted*: its blocks are freed and it goes back to the front of
   the queue to be recomputed later, prompt plus what it had generated).
2. With the remaining token budget (``max_num_batched_tokens``) and free blocks, waiting
   sequences are admitted in arrival order and prefilled, in chunks if their prompt is longer than
   the budget left (*chunked prefill*), so a long prompt never stalls the decodes of others.

A finished sequence frees its blocks at once and its place goes to the next one in the queue on
the very next step.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence

from .kv_cache import PagedKVCacheManager


class Status(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False


_ids = itertools.count()


@dataclass
class Sequence_:
    prompt_ids: List[int]
    params: SamplingParams
    id: str = field(default_factory=lambda: f"seq-{next(_ids)}")
    output_ids: List[int] = field(default_factory=list)
    status: Status = Status.WAITING
    # Tokens whose K/V are in the cache (prefill progress, then every decoded token).
    num_computed: int = 0
    arrival: float = field(default_factory=time.perf_counter)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    finish_reason: Optional[str] = None
    preemptions: int = 0
    cached_prefix: int = 0

    @property
    def all_ids(self) -> List[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def pending(self) -> int:
        """Tokens whose K/V are not cached yet: the prompt (or its rest) while prefilling, the
        last sampled token while decoding, everything after a preemption."""
        return self.num_tokens - self.num_computed

    @property
    def latency(self) -> Optional[float]:
        return None if self.finished_at is None else self.finished_at - self.arrival

    @property
    def ttft(self) -> Optional[float]:
        return None if self.first_token_at is None else self.first_token_at - self.arrival


@dataclass
class ScheduledChunk:
    """Tokens of one sequence to run through the model in this step, from ``start``."""

    seq: Sequence_
    start: int
    tokens: List[int]
    # Whether the step must return logits for this sequence (last prompt token or a decode).
    sample: bool


@dataclass
class SchedulerOutput:
    chunks: List[ScheduledChunk]
    preempted: List[Sequence_]

    @property
    def num_tokens(self) -> int:
        return sum(len(c.tokens) for c in self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Scheduler:
    def __init__(
        self,
        kv: PagedKVCacheManager,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 512,
        max_model_len: int = 4096,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.waiting: Deque[Sequence_] = deque()
        self.running: List[Sequence_] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence_) -> None:
        if len(seq.prompt_ids) + seq.params.max_tokens > self.max_model_len:
            seq.params.max_tokens = max(1, self.max_model_len - len(seq.prompt_ids))
        if len(seq.prompt_ids) >= self.max_model_len:
            raise ValueError(f"prompt of {len(seq.prompt_ids)} tokens exceeds max_model_len {self.max_model_len}")
        if self.kv.blocks_needed(len(seq.prompt_ids) + 1) > self.kv.num_blocks:
            raise ValueError("prompt does not fit in the KV cache")
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, seq_id: str) -> Optional[Sequence_]:
        for seq in list(self.running):
            if seq.id == seq_id:
                self._finish(seq, "abort")
                return seq
        for seq in list(self.waiting):
            if seq.id == seq_id:
                self.waiting.remove(seq)
                seq.status, seq.finish_reason = Status.FINISHED, "abort"
                return seq
        return None

    # ------------------------------------------------------------------ one step

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        chunks: List[ScheduledChunk] = []
        preempted: List[Sequence_] = []

        # 1. Running sequences, oldest first: one token each when decoding, the next piece of the
        #    prompt when a chunked prefill is still under way.
        for seq in sorted(self.running, key=lambda s: s.arrival):
            if seq not in self.running or budget <= 0:
                continue
            if self.kv.tables[seq.id].num_tokens < seq.num_tokens:  # the sampled token needs a slot
                while not self.kv.can_append(seq.id):
                    victim = self._newest_running(exclude=seq)
                    if victim is None:
                        break
                    self._preempt(victim)
                    preempted.append(victim)
                    chunks = [c for c in chunks if c.seq is not victim]
                if not self.kv.can_append(seq.id):
                    self._preempt(seq)
                    preempted.append(seq)
                    continue
                self.kv.append_slot(seq.id)
            n = min(budget, seq.pending)
            start = seq.num_computed
            chunks.append(ScheduledChunk(seq, start, seq.all_ids[start : start + n], start + n == seq.num_tokens))
            budget -= n

        # 2. Admit waiting sequences while there is budget, room and free blocks (not in a step
        #    that had to preempt: the cache is already short).
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs and not preempted:
            seq = self.waiting[0]
            ids = seq.all_ids  # a preempted sequence recomputes what it had generated too
            # keep one free block per running sequence for its next decode
            if not self.kv.can_allocate(len(ids), reserve=len(self.running)):
                break
            self.waiting.popleft()
            seq.cached_prefix = self.kv.allocate(seq.id, ids)
            seq.num_computed = seq.cached_prefix
            seq.status = Status.RUNNING
            self.running.append(seq)
            n = min(budget, seq.pending)
            start = seq.num_computed
            chunks.append(ScheduledChunk(seq, start, ids[start : start + n], start + n == seq.num_tokens))
            budget -= n
        return SchedulerOutput(chunks, preempted)

    def _newest_running(self, exclude: Sequence_) -> Optional[Sequence_]:
        candidates = [s for s in self.running if s is not exclude]
        return max(candidates, key=lambda s: s.arrival) if candidates else None

    def _preempt(self, seq: Sequence_) -> None:
        self.kv.free(seq.id)
        self.running.remove(seq)
        seq.status = Status.WAITING
        seq.num_computed = 0
        seq.preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    # ------------------------------------------------------------------ results

    def update(self, out: SchedulerOutput, sampled: Dict[str, int], eos_ids: Sequence[int]) -> List[Sequence_]:
        """Apply one step's results: advance prefills, append sampled tokens, finish sequences.
        Returns the sequences that received a new token this step."""
        now = time.perf_counter()
        advanced: List[Sequence_] = []
        for chunk in out.chunks:
            seq = chunk.seq
            if seq.status is not Status.RUNNING:
                continue
            seq.num_computed = max(seq.num_computed, chunk.start + len(chunk.tokens))
            self.kv.register_tokens(seq.id, seq.all_ids[: seq.num_computed])
            if not chunk.sample:
                continue
            token = sampled[seq.id]
            if seq.first_token_at is None:
                seq.first_token_at = now
            seq.output_ids.append(token)
            advanced.append(seq)
            if not seq.params.ignore_eos and token in eos_ids:
                self._finish(seq, "stop")
            elif len(seq.output_ids) >= seq.params.max_tokens or seq.num_tokens >= self.max_model_len:
                self._finish(seq, "length")
        return advanced

    def finish(self, seq: Sequence_, reason: str) -> None:
        """Called by the engine for stop strings."""
        if seq.status is Status.RUNNING:
            self._finish(seq, reason)

    def _finish(self, seq: Sequence_, reason: str) -> None:
        if seq.id in self.kv.tables:
            self.kv.register_tokens(seq.id, seq.all_ids[: seq.num_computed])
        self.kv.free(seq.id)
        if seq in self.running:
            self.running.remove(seq)
        seq.status = Status.FINISHED
        seq.finish_reason = reason
        seq.finished_at = time.perf_counter()
