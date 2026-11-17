"""Paged KV cache, continuous-batching scheduler, engine and OpenAI API, on the toy backend."""

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_inference_server import AsyncLLMEngine, LLMEngine, OutOfBlocksError, PagedKVCacheManager, SamplingParams, create_app
from llm_inference_server.backends.toy import EOS, ToyBackend
from llm_inference_server.engine import _TextState
from llm_inference_server.sampling import sample
from llm_inference_server.scheduler import Scheduler, Sequence_, Status

# --------------------------------------------------------------------------- paged KV cache


def test_blocks_are_allocated_on_demand_and_freed():
    kv = PagedKVCacheManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    assert kv.allocate("a", list(range(6))) == 0
    blocks, n = kv.block_table("a")
    assert len(blocks) == 2 and n == 6 and kv.num_free_blocks == 6
    kv.append_slot("a")
    kv.append_slot("a")  # 8 tokens: still 2 blocks
    assert len(kv.block_table("a")[0]) == 2
    kv.append_slot("a")  # 9th token opens a third block
    assert len(kv.block_table("a")[0]) == 3
    kv.free("a")
    assert kv.num_free_blocks == 8 and kv.usage == 0.0


def test_slot_mapping_follows_the_block_table():
    kv = PagedKVCacheManager(num_blocks=4, block_size=4, enable_prefix_caching=False)
    kv.allocate("x", [1, 2, 3])
    kv.allocate("y", [1, 2, 3, 4, 5])
    by, _ = kv.block_table("y")
    assert kv.slots("y", 0, 5) == [by[0] * 4, by[0] * 4 + 1, by[0] * 4 + 2, by[0] * 4 + 3, by[1] * 4]
    assert set(kv.block_table("x")[0]).isdisjoint(by)


def test_out_of_blocks():
    kv = PagedKVCacheManager(num_blocks=2, block_size=4)
    assert not kv.can_allocate(9)
    with pytest.raises(OutOfBlocksError):
        kv.allocate("big", list(range(9)))
    assert kv.num_free_blocks == 2


def test_prefix_caching_shares_full_blocks():
    kv = PagedKVCacheManager(num_blocks=16, block_size=4)
    system = list(range(100, 112))  # 3 full blocks
    assert kv.allocate("a", system + [1, 2]) == 0
    assert kv.allocate("early", system + [5]) == 0  # nothing computed yet: nothing to share
    kv.register_tokens("a", system + [1, 2])  # a's prefill ran
    cached = kv.allocate("b", system + [7, 8, 9])
    assert cached == 12
    a, _ = kv.block_table("a")
    b, _ = kv.block_table("b")
    assert a[:3] == b[:3] and a[3] != b[3]
    # a different prefix shares nothing
    assert kv.allocate("c", [5] + system) == 0
    # freeing keeps cached blocks reusable until the pool needs them
    kv.free("a")
    kv.free("b")
    assert kv.allocate("d", system + [3]) == 12


def test_prefix_cache_never_covers_the_last_prompt_token():
    kv = PagedKVCacheManager(num_blocks=8, block_size=4)
    kv.allocate("a", list(range(8)))
    kv.register_tokens("a", list(range(8)))
    assert kv.allocate("b", list(range(8))) == 4  # the block with the last token is recomputed


def test_cached_blocks_are_evicted_when_needed():
    kv = PagedKVCacheManager(num_blocks=3, block_size=2)
    kv.allocate("a", [1, 2, 3, 4, 5])
    kv.register_tokens("a", [1, 2, 3, 4, 5])
    kv.free("a")
    assert kv.num_free_blocks == 3
    kv.allocate("b", [9, 9, 9, 9, 9, 9])  # needs all 3 blocks: evicts the cached ones
    assert kv.num_free_blocks == 0


# --------------------------------------------------------------------------- scheduler


def make(num_blocks=64, block_size=4, **kw):
    kv = PagedKVCacheManager(num_blocks, block_size, enable_prefix_caching=False)
    return kv, Scheduler(kv, **kw)


def test_continuous_batching_admits_as_soon_as_a_slot_frees():
    eng = LLMEngine(ToyBackend(kv_tokens=4096), block_size=4, max_num_seqs=2, max_num_batched_tokens=64)
    short = eng.add_request(list(b"abc"), SamplingParams(max_tokens=2))
    long = eng.add_request(list(b"abc"), SamplingParams(max_tokens=10))
    late = eng.add_request(list(b"xyz"), SamplingParams(max_tokens=3))
    steps = []
    while eng.has_work():
        eng.step()
        steps.append({s.id for s in eng.scheduler.running})
    # the third request joined while the long one was still running (not after the whole batch)
    assert any(late.id in s and long.id in s for s in steps)
    assert short.finish_reason == long.finish_reason == late.finish_reason == "length"
    assert len(long.output_ids) == 10


def test_chunked_prefill_respects_the_token_budget():
    backend = ToyBackend()
    eng = LLMEngine(backend, block_size=4, max_num_seqs=4, max_num_batched_tokens=8)
    eng.add_request(list(b"abcdefghijklmnopqrst"), SamplingParams(max_tokens=1))  # 20-token prompt
    while eng.has_work():
        eng.step()
    sizes = [sum(len(i.tokens) for i in call) for call in backend.calls]
    assert max(sizes) <= 8 and sizes[:3] == [8, 8, 4]
    assert [i.want_logits for i in backend.calls[2]] == [True]


def test_decodes_are_scheduled_before_new_prefills():
    kv, sch = make(max_num_seqs=4, max_num_batched_tokens=6)
    a = Sequence_(list(range(10, 14)), SamplingParams(max_tokens=5))
    sch.add(a)
    out = sch.schedule()
    sch.update(out, {a.id: 1}, ())
    b = Sequence_(list(range(20, 30)), SamplingParams(max_tokens=5))
    sch.add(b)
    out = sch.schedule()
    assert out.chunks[0].seq is a and out.chunks[0].tokens == [1]
    assert out.chunks[1].seq is b and len(out.chunks[1].tokens) == 5 and not out.chunks[1].sample


def test_preemption_recomputes_without_losing_tokens():
    backend = ToyBackend(kv_tokens=24)  # 6 blocks of 4 slots
    eng = LLMEngine(backend, block_size=4, max_num_seqs=3, max_num_batched_tokens=64)
    seqs = [eng.add_request(list(b"aaaa"), SamplingParams(max_tokens=8, ignore_eos=True)) for _ in range(3)]
    while eng.has_work():
        eng.step()
    assert eng.scheduler.num_preemptions > 0
    expected = list(b"bcdefghi")
    for s in seqs:
        assert s.output_ids == expected and s.finish_reason == "length"
    assert eng.kv.num_free_blocks == eng.kv.num_blocks


def test_prompt_too_long_is_rejected():
    kv, sch = make(num_blocks=2, block_size=4, max_model_len=100)
    with pytest.raises(ValueError):
        sch.add(Sequence_(list(range(20)), SamplingParams()))


def test_abort_frees_blocks():
    eng = LLMEngine(ToyBackend(), block_size=4)
    s = eng.add_request(list(b"abc"), SamplingParams(max_tokens=50))
    eng.step()
    eng.abort(s.id)
    assert s.status is Status.FINISHED and s.finish_reason == "abort"
    assert eng.kv.num_free_blocks == eng.kv.num_blocks and not eng.has_work()


# --------------------------------------------------------------------------- engine


def test_batched_equals_sequential_and_prefix_cache_hits():
    prompts = [list(b"system prompt shared by all: ") + list(t.encode()) for t in ("abc", "xyz", "mmm")]
    params = SamplingParams(max_tokens=6)
    together = LLMEngine(ToyBackend(), block_size=4, max_num_seqs=3)
    first = together.generate(prompts[:1], params)[0].output_ids  # its blocks enter the cache
    out = [first] + [s.output_ids for s in together.generate(prompts[1:], params)]
    alone = [LLMEngine(ToyBackend(), block_size=4, max_num_seqs=1).generate([p], params)[0].output_ids for p in prompts]
    assert out == alone
    assert together.metrics()["kv_prefix_hit_rate"] > 0
    assert together.metrics()["prefill_tokens_computed"] < sum(len(p) for p in prompts)


def test_stop_strings_and_eos():
    eng = LLMEngine(ToyBackend())
    s = eng.add_request(list(b"a"), SamplingParams(max_tokens=20, stop=["de"]))
    text = ""
    while eng.has_work():
        for ev in eng.step():
            text += ev.text_delta
    assert text == "bc" and s.finish_reason == "stop"


def test_text_state_holds_back_partial_utf8_and_stop_prefixes():
    st = _TextState(["STOP"])
    assert st.push("hola ST", final=False) == ("hola", False)
    assert st.push("hola STOP y más", final=False) == (" ", True)
    st2 = _TextState([])
    assert st2.push("caf�", final=False) == ("caf", False)
    assert st2.push("café", final=True) == ("é", False)


def test_sampling():
    logits = np.array([0.0, 5.0, 1.0])
    assert sample(logits, SamplingParams(temperature=0)) == 1
    rng = np.random.default_rng(0)
    draws = {sample(logits, SamplingParams(temperature=1.0, top_k=1), rng) for _ in range(20)}
    assert draws == {1}
    draws = [sample(logits, SamplingParams(temperature=1.0, top_p=0.5), np.random.default_rng(i)) for i in range(20)]
    assert set(draws) == {1}


def test_async_engine_streams_concurrent_requests():
    engine = AsyncLLMEngine(LLMEngine(ToyBackend(), block_size=4, max_num_seqs=4))

    async def collect(prompt, n):
        text = ""
        async for ev in engine.generate(list(prompt.encode()), SamplingParams(max_tokens=n)):
            text += ev.text_delta
        return text

    async def main():
        return await asyncio.gather(collect("a", 3), collect("x", 4), collect("m", 2))

    try:
        assert asyncio.run(main()) == ["bcd", "yzab", "no"]
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------- OpenAI API


@pytest.fixture
def client():
    engine = AsyncLLMEngine(LLMEngine(ToyBackend(), block_size=4, max_num_seqs=4))
    with TestClient(create_app(engine, model_name="toy")) as c:
        yield c
    engine.shutdown()


def test_models_health_metrics(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "toy"
    assert client.get("/health").json()["status"] == "ok"
    assert "kv_usage" in client.get("/metrics").json()


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={"model": "toy", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0})
    body = r.json()
    assert r.status_code == 200 and body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "abc"  # the template ends in "assistant: "
    assert body["usage"]["completion_tokens"] == 3 and body["choices"][0]["finish_reason"] == "length"


def test_chat_completion_stream_with_usage(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 4, "stream": True, "stream_options": {"include_usage": True}},
    ) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "abcd"
    assert chunks[-1]["usage"]["completion_tokens"] == 4


def test_text_completion_and_errors(client):
    r = client.post("/v1/completions", json={"prompt": "k", "max_tokens": 2, "temperature": 0})
    assert r.json()["choices"][0]["text"] == "lm"
    assert client.post("/v1/completions", json={"model": "other", "prompt": "k"}).status_code == 404
    assert client.post("/v1/completions", json={"prompt": "k", "temperature": 5}).status_code == 422


def test_eos_stops_generation():
    class StopAfterTwo(ToyBackend):
        def forward(self, items):
            out = super().forward(items)
            for item in items:
                if item.want_logits and len(self.contexts[item.seq_id]) >= 3:
                    out[item.seq_id][:] = -10
                    out[item.seq_id][EOS] = 10
            return out

    eng = LLMEngine(StopAfterTwo())
    s = eng.generate([list(b"a")], SamplingParams(max_tokens=10))[0]
    assert s.finish_reason == "stop" and s.output_ids[-1] == EOS and len(s.output_ids) == 3


def test_blocks_reserved_but_not_computed_are_never_shared():
    """A sequence aborted before its prefill ran must not leave 'cached' garbage blocks."""
    eng = LLMEngine(ToyBackend(), block_size=4, max_num_seqs=2, max_num_batched_tokens=64)
    prompt = list(b"shared prefix of sixteen!") + [1]
    first = eng.add_request(prompt, SamplingParams(max_tokens=1))
    eng.scheduler.schedule()  # blocks reserved, forward never run
    eng.abort(first.id)
    second = eng.add_request(prompt, SamplingParams(max_tokens=1))
    eng.step()
    assert second.cached_prefix == 0


def test_prefix_cache_reuses_blocks_computed_in_earlier_steps():
    eng = LLMEngine(ToyBackend(), block_size=4, max_num_seqs=2, max_num_batched_tokens=64)
    prompt = list(b"shared prefix of sixteen!")
    eng.generate([prompt + list(b"a")], SamplingParams(max_tokens=1))
    later = eng.generate([prompt + list(b"b")], SamplingParams(max_tokens=1))[0]
    assert later.cached_prefix == 24
