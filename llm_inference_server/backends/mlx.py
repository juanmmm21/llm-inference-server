"""MLX backend (Apple silicon, and MLX's CPU build on Linux) with the paged KV cache.

The model is any ``mlx-lm`` checkpoint (``mlx-community/...``). Its attention layers call
``cache.update_and_fetch(keys, values)``; here that cache is :class:`_PagedLayerCache`, which
writes the new keys/values into a per-layer **block pool** (``[num_slots, n_kv_heads, head_dim]``)
at the slots the :class:`~llm_inference_server.kv_cache.PagedKVCacheManager` assigned and reads
the sequence's context back through its block table. Prefix-cache hits are therefore real here
too.

Batching: mlx-lm models take one RoPE offset per forward call, so sequences at different
positions cannot share a call without changing the model code. Each scheduler step therefore runs
one forward per sequence — the batch is still re-formed every step (continuous batching: arrivals
join, finished sequences leave, prefills are chunked), but the matrix multiplications are not
fused across sequences as they are in the llama.cpp and torch backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .base import Backend, BatchItem


class MLXBackend(Backend):
    name = "mlx"
    supports_prefix_cache = True

    def __init__(
        self,
        model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        kv_tokens: int = 8192,
        block_size: int = 16,
        threads: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import KVCache

        self.mx = mx
        self._KVCache = KVCache
        self.model, self.tokenizer = load(model)
        self.model_name = model_name or model.rstrip("/").split("/")[-1]
        args = self.model.args
        self.n_layers = len(self.model.layers)
        self.n_kv = getattr(args, "num_key_value_heads", None) or args.num_attention_heads
        self.head_dim = getattr(args, "head_dim", None) or args.hidden_size // args.num_attention_heads
        self.block_size = block_size
        self.num_slots = (kv_tokens // block_size) * block_size
        dtype = self.model.model.embed_tokens.weight.dtype if hasattr(self.model.model.embed_tokens, "weight") else mx.float16
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            dtype = mx.float16  # quantized embeddings: keep the cache in half precision
        self.k_pool = [mx.zeros((self.num_slots, self.n_kv, self.head_dim), dtype=dtype) for _ in range(self.n_layers)]
        self.v_pool = [mx.zeros((self.num_slots, self.n_kv, self.head_dim), dtype=dtype) for _ in range(self.n_layers)]
        self._max_len = int(getattr(args, "max_position_embeddings", 4096) or 4096)
        eos = getattr(self.tokenizer, "eos_token_ids", None) or {self.tokenizer.eos_token_id}
        self._eos = {int(t) for t in eos if t is not None}

    @property
    def eos_token_ids(self) -> Set[int]:
        return self._eos

    @property
    def max_model_len(self) -> int:
        return self._max_len

    def kv_capacity_tokens(self) -> int:
        return self.num_slots

    def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def apply_chat_template(self, messages):  # type: ignore[override]
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    def forward(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        mx = self.mx
        out: Dict[str, np.ndarray] = {}
        for item in items:
            ctx_slots = _context_slots(item, self.block_size)
            caches = [
                _PagedLayerCache(self, layer, item.start_pos, mx.array(item.slots), mx.array(ctx_slots), self._KVCache)
                for layer in range(self.n_layers)
            ]
            logits = self.model(mx.array(item.tokens)[None], cache=caches)
            # the pools were updated lazily inside the graph: materialize them with the logits
            last = logits[0, -1].astype(mx.float32)
            mx.eval(last, *self.k_pool, *self.v_pool)
            if item.want_logits:
                out[item.seq_id] = np.array(last)
        return out


def _context_slots(item: BatchItem, bs: int) -> List[int]:
    ctx = item.context_len
    slots: List[int] = []
    for b in item.block_table[: -(-ctx // bs)]:
        slots.extend(range(b * bs, b * bs + bs))
    return slots[:ctx]


class _PagedLayerCache:
    """mlx-lm's cache interface backed by the shared block pool of one layer."""

    def __init__(self, backend: MLXBackend, layer: int, offset: int, write_slots, read_slots, kv_cls) -> None:
        self.backend = backend
        self.layer = layer
        self.offset = offset
        self._write = write_slots
        self._read = read_slots
        self._kv_cls = kv_cls

    def update_and_fetch(self, keys, values):
        # keys: [1, n_kv, L, d] -> [L, n_kv, d] at the assigned slots
        k = keys[0].transpose(1, 0, 2).astype(self.backend.k_pool[self.layer].dtype)
        v = values[0].transpose(1, 0, 2).astype(self.backend.v_pool[self.layer].dtype)
        kp, vp = self.backend.k_pool[self.layer], self.backend.v_pool[self.layer]
        kp[self._write] = k
        vp[self._write] = v
        self.backend.k_pool[self.layer], self.backend.v_pool[self.layer] = kp, vp
        self.offset += keys.shape[2]
        K = kp[self._read].transpose(1, 0, 2)[None].astype(keys.dtype)
        V = vp[self._read].transpose(1, 0, 2)[None].astype(values.dtype)
        return K, V

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def state(self):
        return ()

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return False
