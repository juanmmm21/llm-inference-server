"""The torch reference backend against Hugging Face transformers (slow: downloads ~1 GB).

Run with AICORE_SLOW_TESTS=1. Logits must match HF to float32 precision, and greedy generation
through the engine — several sequences batched, chunked prefill, prefix-cache hits — must
produce exactly HF's greedy tokens.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AICORE_SLOW_TESTS"), reason="set AICORE_SLOW_TESTS=1")

MODEL = os.environ.get("TORCH_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def models():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from llm_inference_server.backends.torch_ref import TorchPagedBackend

    be = TorchPagedBackend(MODEL, kv_tokens=4096, block_size=16)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return be, hf, torch


def prompts(be):
    system = "You are a terse assistant. " * 8
    return [
        be.tokenize(be.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}]))
        for q in ("What is the capital of France?", "Name three prime numbers.", "Write a haiku about rain.")
    ]


def test_logits_match_huggingface(models):
    from llm_inference_server.backends.base import BatchItem
    from llm_inference_server.engine import LLMEngine
    from llm_inference_server.scheduler import SamplingParams

    be, hf, torch = models
    ids = prompts(be)[0]
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0, -1].numpy()
    eng = LLMEngine(be, block_size=16, enable_prefix_caching=False)
    seq = eng.add_request(ids, SamplingParams(max_tokens=1))
    c = eng.scheduler.schedule().chunks[0]
    item = BatchItem(seq.id, c.tokens, c.start, True, eng.kv.slots(seq.id, c.start, c.start + len(c.tokens)), eng.kv.block_table(seq.id)[0])
    ours = be.forward([item])[seq.id]
    eng.abort(seq.id)
    assert np.abs(ours - ref).max() < 1e-3


def test_batched_chunked_prefix_cached_greedy_equals_huggingface(models):
    from llm_inference_server.engine import LLMEngine
    from llm_inference_server.scheduler import SamplingParams

    be, hf, torch = models
    ps = prompts(be)
    eng = LLMEngine(be, block_size=16, max_num_seqs=3, max_num_batched_tokens=48)
    params = SamplingParams(max_tokens=20, temperature=0)
    first = eng.generate(ps[:1], params)
    rest = eng.generate(ps[1:], params)
    m = eng.metrics()
    assert m["kv_prefix_hit_rate"] > 0 and m["mean_batch_size"] > 1
    for ids, seq in zip(ps, first + rest):
        with torch.no_grad():
            # plain greedy: Qwen's generation_config adds repetition_penalty=1.05 by default
            ref = hf.generate(
                torch.tensor([ids]), max_new_tokens=20, do_sample=False, repetition_penalty=1.0, top_p=None, top_k=None, temperature=None
            )[0, len(ids):].tolist()
        n = min(len(ref), len(seq.output_ids))
        assert seq.output_ids[:n] == ref[:n]
