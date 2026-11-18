"""How the MLX backend packs several sequences into one forward call (no MLX needed)."""

import numpy as np
import pytest

from llm_inference_server.backends.base import BatchItem
from llm_inference_server.backends.mlx import build_layout, context_slots, plan_groups


def test_decodes_share_one_group_and_long_prefills_go_apart():
    groups = plan_groups([1, 1, 1, 64, 1, 60])
    assert sorted(map(sorted, groups)) == [[0, 1, 2, 4], [3, 5]]


def test_every_item_is_planned_exactly_once():
    lengths = [5, 1, 9, 1, 30, 2, 2, 7]
    groups = plan_groups(lengths)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        longest = max(lengths[i] for i in g)
        assert len(g) * longest <= 1.5 * sum(lengths[i] for i in g) or len(g) == 1


def test_pad_ratio_below_one_is_rejected():
    with pytest.raises(ValueError):
        plan_groups([1, 2], max_pad_ratio=0.9)


def test_context_slots_follow_the_block_table():
    item = BatchItem("a", [7], start_pos=5, want_logits=True, slots=[13], block_table=[3, 1])
    assert context_slots(item, 4) == [12, 13, 14, 15, 4, 5]
    with pytest.raises(ValueError):
        context_slots(BatchItem("b", [7], 8, True, [0], [3]), 4)


def test_layout_pads_rows_and_masks_each_context():
    bs = 4
    # a: decode at position 5 (context 6); b: prefill chunk of 3 tokens at positions 0..2
    a = BatchItem("a", [11], start_pos=5, want_logits=True, slots=[13], block_table=[3, 1])
    b = BatchItem("b", [21, 22, 23], start_pos=0, want_logits=False, slots=[8, 9, 10], block_table=[2])
    lay = build_layout([a, b], bs)

    assert lay.tokens.tolist() == [[11, 0, 0], [21, 22, 23]]
    assert lay.offsets.tolist() == [5, 0]
    assert lay.valid.tolist() == [0, 3, 4, 5]
    assert lay.write_slots.tolist() == [13, 8, 9, 10]
    assert lay.last.tolist() == [0, 5]
    assert lay.read_slots.tolist() == [[12, 13, 14, 15, 4, 5], [8, 9, 10, 0, 0, 0]]
    # a's real query sees its whole context; b is causal and never sees the padded keys
    assert lay.mask[0, 0, 0].tolist() == [True] * 6
    assert lay.mask[1, 0].tolist() == [
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
    ]
    # padded query rows keep at least one visible key (no empty softmax)
    assert np.all(lay.mask.any(axis=-1))


def test_layout_rejects_inconsistent_items():
    with pytest.raises(ValueError):
        build_layout([], 4)
    with pytest.raises(ValueError):
        build_layout([BatchItem("a", [1, 2], 0, True, [0], [0])], 4)


def _tiny_backend(block_size=4, max_pad_ratio=1.5):
    """A two-layer Qwen2 with random weights: exercises the real mlx-lm attention code in seconds."""
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import qwen2

    from llm_inference_server.backends.mlx import MLXBackend

    class _Tokenizer:
        eos_token_id = 255

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    mx.random.seed(0)
    args = qwen2.ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=2,
    )
    model = qwen2.Model(args)
    mx.eval(model.parameters())
    return MLXBackend.from_loaded(model, _Tokenizer(), "tiny", kv_tokens=512, block_size=block_size, max_pad_ratio=max_pad_ratio)


def _reference_logits(model, ids):
    """Logits of every position of ``ids`` from one forward without any cache."""
    import mlx.core as mx

    return np.array(model(mx.array(ids)[None])[0].astype(mx.float32))


def _run_and_check(be, prompts, max_num_batched_tokens, n=6, eng=None):
    """Generate with the engine and compare every step's logits with a cache-free forward."""
    from llm_inference_server.engine import LLMEngine
    from llm_inference_server.scheduler import SamplingParams

    seen = {}
    forward = be.forward

    def spy(items):
        out = forward(items)
        for sid, row in out.items():
            seen.setdefault(sid, []).append(row)
        return out

    be.forward = spy
    eng = eng or LLMEngine(be, block_size=4, max_num_seqs=4, max_num_batched_tokens=max_num_batched_tokens)
    seqs = eng.generate(prompts, SamplingParams(max_tokens=n, temperature=0, ignore_eos=True))
    for ids, seq in zip(prompts, seqs):
        ref = _reference_logits(be.model, ids + seq.output_ids[:-1])
        got = np.stack(seen[seq.id])
        np.testing.assert_allclose(got, ref[len(ids) - 1 :], atol=1e-4, rtol=1e-4)
    return eng


@pytest.mark.parametrize("max_pad_ratio", [1.0, 1.5, 100.0])
def test_batched_mlx_forward_matches_a_cache_free_forward(max_pad_ratio):
    """Mixed prefill chunks and decodes in one call give the logits of a plain mlx-lm forward.

    ``max_pad_ratio`` 1.0 runs only equal-length rows together, 100 puts everything in one call.
    """
    be = _tiny_backend(max_pad_ratio=max_pad_ratio)
    shared = list(range(10, 30))
    prompts = [shared + [40, 41, 42], list(range(50, 57)), shared + [60], [70, 71]]
    eng = _run_and_check(be, prompts, max_num_batched_tokens=24)
    assert eng.metrics()["mean_batch_size"] > 1


def test_prefix_cache_hits_read_blocks_written_by_a_batched_call():
    be = _tiny_backend()
    shared = list(range(100, 124))
    eng = _run_and_check(be, [shared + [1], shared + [2]], max_num_batched_tokens=64)
    _run_and_check(be, [shared + [3]], max_num_batched_tokens=64, eng=eng)
    assert eng.metrics()["kv_prefix_hit_rate"] > 0
