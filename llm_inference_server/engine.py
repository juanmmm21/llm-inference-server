"""The engine: scheduler + paged KV cache + backend, stepped in a loop.

:class:`LLMEngine` is synchronous: ``step()`` runs one iteration of continuous batching and
returns what each sequence produced. :class:`AsyncLLMEngine` runs that loop in a background
thread and hands tokens to asyncio consumers as they appear, which is what the HTTP server uses.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

import numpy as np

from .backends.base import Backend, BatchItem
from .kv_cache import PagedKVCacheManager
from .sampling import sample
from .scheduler import SamplingParams, Scheduler, Sequence_, Status


@dataclass
class TokenEvent:
    """What one step produced for one sequence."""

    seq_id: str
    token_id: int
    text_delta: str
    finished: bool = False
    finish_reason: Optional[str] = None


@dataclass
class EngineStats:
    steps: int = 0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    prefill_tokens_computed: int = 0
    requests_finished: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    ttfts: List[float] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


class _TextState:
    """Incremental detokenization with stop strings: emits only text that is final."""

    def __init__(self, stops: List[str]) -> None:
        self.emitted = ""
        self.stops = [s for s in stops if s]
        self.holdback = max((len(s) for s in self.stops), default=1) - 1

    def push(self, full_text: str, final: bool) -> tuple[str, bool]:
        for stop in self.stops:
            idx = full_text.find(stop, max(0, len(self.emitted) - len(stop)))
            if idx != -1:
                delta = full_text[len(self.emitted) : idx]
                self.emitted += delta
                return delta, True
        if not final:
            # an incomplete UTF-8 character decodes to U+FFFD: wait for the next token
            safe = len(full_text) - (1 if full_text.endswith("�") else 0)
            safe = min(safe, len(full_text) - self.holdback) if self.stops else safe
        else:
            safe = len(full_text)
        delta = full_text[len(self.emitted) : max(len(self.emitted), safe)]
        self.emitted += delta
        return delta, False


class LLMEngine:
    def __init__(
        self,
        backend: Backend,
        block_size: int = 16,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 512,
        max_model_len: Optional[int] = None,
        enable_prefix_caching: Optional[bool] = None,
    ) -> None:
        self.backend = backend
        num_blocks = max(1, backend.kv_capacity_tokens() // block_size)
        prefix = backend.supports_prefix_cache if enable_prefix_caching is None else enable_prefix_caching and backend.supports_prefix_cache
        self.kv = PagedKVCacheManager(num_blocks, block_size, enable_prefix_caching=prefix)
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=max_num_seqs if backend.batched else 1,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=min(max_model_len or backend.max_model_len, backend.max_model_len),
        )
        self.stats = EngineStats()
        self._text: Dict[str, _TextState] = {}
        self._rng: Dict[str, np.random.Generator] = {}
        self._seqs: Dict[str, Sequence_] = {}

    def add_request(self, prompt_ids: List[int], params: Optional[SamplingParams] = None, seq_id: Optional[str] = None) -> Sequence_:
        params = params or SamplingParams()
        seq = Sequence_(list(prompt_ids), params) if seq_id is None else Sequence_(list(prompt_ids), params, id=seq_id)
        self.scheduler.add(seq)
        self._text[seq.id] = _TextState(params.stop)
        self._rng[seq.id] = np.random.default_rng(params.seed)
        self._seqs[seq.id] = seq
        self.stats.prompt_tokens += len(prompt_ids)
        return seq

    def abort(self, seq_id: str) -> None:
        seq = self.scheduler.abort(seq_id)
        if seq is not None:
            self._cleanup(seq)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> List[TokenEvent]:
        out = self.scheduler.schedule()
        for seq in out.preempted:
            self.backend.release(seq.id)
        if out.is_empty:
            return []
        items = []
        for c in out.chunks:
            blocks, _ = self.kv.block_table(c.seq.id)
            items.append(
                BatchItem(
                    seq_id=c.seq.id,
                    tokens=c.tokens,
                    start_pos=c.start,
                    want_logits=c.sample,
                    slots=self.kv.slots(c.seq.id, c.start, c.start + len(c.tokens)),
                    block_table=blocks,
                )
            )
            if c.start < len(c.seq.prompt_ids):
                self.stats.prefill_tokens_computed += min(len(c.tokens), len(c.seq.prompt_ids) - c.start)
        logits = self.backend.forward(items)
        sampled = {c.seq.id: sample(logits[c.seq.id], c.seq.params, self._rng[c.seq.id]) for c in out.chunks if c.sample}
        advanced = self.scheduler.update(out, sampled, tuple(self.backend.eos_token_ids))
        self.stats.steps += 1
        self.stats.batch_sizes.append(len(out.chunks))
        events = []
        for seq in advanced:
            self.stats.tokens_generated += 1
            if len(seq.output_ids) == 1 and seq.ttft is not None:
                self.stats.ttfts.append(seq.ttft)
            finished = seq.status is Status.FINISHED
            ids = seq.output_ids[:-1] if finished and seq.finish_reason == "stop" else seq.output_ids
            delta, hit_stop = self._text[seq.id].push(self.backend.decode(ids), final=finished)
            if hit_stop and not finished:
                self.scheduler.finish(seq, "stop")
                finished = True
            events.append(TokenEvent(seq.id, seq.output_ids[-1], delta, finished, seq.finish_reason if finished else None))
            if finished:
                self._cleanup(seq)
        return events

    def _cleanup(self, seq: Sequence_) -> None:
        self.backend.release(seq.id)
        self.stats.requests_finished += 1
        self._text.pop(seq.id, None)
        self._rng.pop(seq.id, None)
        self._seqs.pop(seq.id, None)

    def generate(self, prompts: List[List[int]], params: Optional[SamplingParams] = None) -> List[Sequence_]:
        """Run ``prompts`` to completion with continuous batching (offline use, tests)."""
        seqs = [self.add_request(p, SamplingParams(**vars(params)) if params else None) for p in prompts]
        while self.has_work():
            self.step()
        return seqs

    def metrics(self) -> Dict[str, float]:
        s = self.stats
        elapsed = time.perf_counter() - s.started
        return {
            "steps": s.steps,
            "tokens_generated": s.tokens_generated,
            "prompt_tokens": s.prompt_tokens,
            "prefill_tokens_computed": s.prefill_tokens_computed,
            "requests_finished": s.requests_finished,
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "preemptions": self.scheduler.num_preemptions,
            "mean_batch_size": round(float(np.mean(s.batch_sizes)), 3) if s.batch_sizes else 0.0,
            "mean_ttft_s": round(float(np.mean(s.ttfts)), 4) if s.ttfts else 0.0,
            "generation_tokens_per_s": round(s.tokens_generated / elapsed, 2) if elapsed > 0 else 0.0,
            **{f"kv_{k}": v for k, v in self.kv.stats().items()},
        }


class AsyncLLMEngine:
    """Runs :class:`LLMEngine` in its own loop; ``generate`` is an async token stream.

    With ``start_thread=True`` (default) the loop runs in a background thread. Servers should
    prefer ``start_thread=False`` and call :meth:`run` from the **main thread**, serving HTTP from
    another thread: llama.cpp built with OpenMP (the default wheel) runs about half as fast when
    ``llama_decode`` is called from a thread other than the one that first used OpenMP, because a
    second OpenMP team competes with the first for the same cores (measured: 41 vs 20 tokens/s,
    see README). ``python -m llm_inference_server`` does this.
    """

    def __init__(self, engine: LLMEngine, start_thread: bool = True) -> None:
        self.engine = engine
        self._cv = threading.Condition()
        self._incoming: List[tuple] = []
        self._aborts: List[str] = []
        self._queues: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._stop = False
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(target=self.run, name="llm-engine", daemon=True)
            self._thread.start()

    @property
    def backend(self) -> Backend:
        return self.engine.backend

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def generate(self, prompt_ids: List[int], params: SamplingParams) -> AsyncIterator[TokenEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        seq = Sequence_(list(prompt_ids), params)
        with self._cv:
            if self._error is not None:
                raise RuntimeError("engine loop failed") from self._error
            self._queues[seq.id] = (loop, queue)
            self._incoming.append((seq.id, list(prompt_ids), params))
            self._cv.notify()
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            with self._cv:
                if seq.id in self._queues:
                    self._queues.pop(seq.id, None)
                    self._aborts.append(seq.id)
                    self._cv.notify()

    def _deliver(self, seq_id: str, item: object) -> None:
        target = self._queues.get(seq_id)
        if target is not None:
            loop, queue = target
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def run(self) -> None:
        """The engine loop; blocks until :meth:`shutdown`."""
        eng = self.engine
        while True:
            with self._cv:
                while not self._stop and not self._incoming and not self._aborts and not eng.has_work():
                    self._cv.wait()
                if self._stop:
                    return
                incoming, self._incoming = self._incoming, []
                aborts, self._aborts = self._aborts, []
            for seq_id in aborts:
                eng.abort(seq_id)
            for seq_id, prompt_ids, params in incoming:
                try:
                    eng.add_request(prompt_ids, params, seq_id=seq_id)
                except ValueError as exc:
                    self._deliver(seq_id, exc)
            if not eng.has_work():
                continue
            try:
                events = eng.step()
            except BaseException as exc:  # a backend failure must not hang every client
                self._error = exc
                with self._cv:
                    for sid in list(self._queues):
                        self._deliver(sid, RuntimeError(f"engine step failed: {exc}"))
                    self._queues.clear()
                for seq in list(eng.scheduler.running) + list(eng.scheduler.waiting):
                    eng.abort(seq.id)
                self._error = None
                continue
            with self._cv:
                for ev in events:
                    self._deliver(ev.seq_id, ev)
                    if ev.finished:
                        self._queues.pop(ev.seq_id, None)
