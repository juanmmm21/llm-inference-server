"""The MLX backend (paged KV pool behind mlx-lm's cache interface) against mlx-lm itself.

Slow and needs MLX: run with AICORE_SLOW_TESTS=1. On Linux it uses MLX's CPU build, where 4-bit
checkpoints are very slow; the default model is therefore the bf16 one.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AICORE_SLOW_TESTS"), reason="set AICORE_SLOW_TESTS=1")
MODEL = os.environ.get("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-bf16")


def reference_greedy(model, ids, n):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)
    out = []
    for _ in range(n):
        tok = int(logits[0, -1].argmax())
        out.append(tok)
        logits = model(mx.array([tok])[None], cache=cache)
    return out


def test_paged_mlx_matches_mlx_lm_greedy():
    pytest.importorskip("mlx.core")
    from llm_inference_server.backends.mlx import MLXBackend
    from llm_inference_server.engine import LLMEngine
    from llm_inference_server.scheduler import SamplingParams

    be = MLXBackend(MODEL, kv_tokens=1024, block_size=16)
    system = "You are a terse assistant. " * 6
    prompts = [
        be.tokenize(be.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}]))
        for q in ("What is the capital of France?", "Name a prime number.")
    ]
    eng = LLMEngine(be, block_size=16, max_num_seqs=2, max_num_batched_tokens=32)
    params = SamplingParams(max_tokens=6, temperature=0)
    first = eng.generate(prompts[:1], params)
    second = eng.generate(prompts[1:], params)
    assert eng.metrics()["kv_prefix_hit_rate"] > 0
    for ids, seq in zip(prompts, first + second):
        assert seq.output_ids == reference_greedy(be.model, ids, 6)
