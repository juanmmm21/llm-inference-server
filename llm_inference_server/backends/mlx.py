"""MLX backend (Apple silicon, and MLX's CPU build on Linux) with the paged KV cache.

The model is any ``mlx-lm`` checkpoint (``mlx-community/...``). Its attention layers call
``cache.update_and_fetch(keys, values)``; here that cache is :class:`_PagedLayerCache`, which
writes the new keys/values into a per-layer **block pool** (``[num_slots, n_kv_heads, head_dim]``)
at the slots the :class:`~llm_inference_server.kv_cache.PagedKVCacheManager` assigned and reads
each sequence's context back through its block table. Prefix-cache hits are therefore real here
too.

Batching: several sequences share one forward call. The step's chunks are right-padded to a
``[B, L]`` batch; RoPE takes one offset per row (``mx.fast.rope`` accepts a vector of ``B``
offsets, and mlx-lm passes ``cache.offset`` straight to it), and the attention mask, built here,
lets each row see only its own context and its own earlier tokens. Padding costs compute, so the
step is split into groups of similar length (:func:`plan_groups`): all decodes go together, and a
long prefill chunk does not drag a batch of one-token decodes up to its length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from .base import Backend, BatchItem

#: A group may hold at most this many padded tokens per real token.
DEFAULT_MAX_PAD_RATIO = 1.5


def plan_groups(lengths: Sequence[int], max_pad_ratio: float = DEFAULT_MAX_PAD_RATIO) -> List[List[int]]:
    """Split items (by their token counts) into forward calls with bounded padding.

    Items are taken shortest first; one joins the current group while ``rows × longest`` stays
    within ``max_pad_ratio`` times the real tokens. Returns lists of indices into ``lengths``.
    """
    if max_pad_ratio < 1.0:
        raise ValueError("max_pad_ratio must be >= 1")
    groups: List[List[int]] = []
    current: List[int] = []
    real = 0
    for idx in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        n = lengths[idx]
        if current and (len(current) + 1) * n > max_pad_ratio * (real + n):
            groups.append(current)
            current, real = [], 0
        current.append(idx)
        real += n
    if current:
        groups.append(current)
    return groups


@dataclass
class BatchLayout:
    """Index arrays of one batched forward call (numpy, turned into ``mx.array`` by the backend).

    ``tokens`` is ``[B, L]`` (right-padded with 0), ``offsets`` the first position of every row,
    ``valid`` the flat indices (into ``B·L``) of the real tokens and ``write_slots`` their KV
    slots, in the same order. ``read_slots`` is ``[B, C]``: the context slots of every row,
    padded with slot 0, and ``mask`` ``[B, 1, L, C]`` says which of them each query may see.
    ``last`` is the flat index of every row's last real token (where its logits are read).
    """

    tokens: np.ndarray
    offsets: np.ndarray
    valid: np.ndarray
    write_slots: np.ndarray
    read_slots: np.ndarray
    mask: np.ndarray
    last: np.ndarray


def context_slots(item: BatchItem, block_size: int) -> List[int]:
    """Physical slots of positions ``0 .. context_len - 1`` of ``item``, from its block table."""
    ctx = item.context_len
    slots: List[int] = []
    for b in item.block_table[: -(-ctx // block_size)]:
        slots.extend(range(b * block_size, b * block_size + block_size))
    if len(slots) < ctx:
        raise ValueError(f"block table of {item.seq_id} covers {len(slots)} tokens, needs {ctx}")
    return slots[:ctx]


def build_layout(items: Sequence[BatchItem], block_size: int) -> BatchLayout:
    if not items:
        raise ValueError("empty batch")
    for item in items:
        if not item.tokens:
            raise ValueError(f"{item.seq_id}: no tokens to run")
        if len(item.slots) != len(item.tokens):
            raise ValueError(f"{item.seq_id}: {len(item.slots)} slots for {len(item.tokens)} tokens")
    B = len(items)
    L = max(len(it.tokens) for it in items)
    C = max(it.context_len for it in items)
    tokens = np.zeros((B, L), dtype=np.int32)
    read = np.zeros((B, C), dtype=np.int32)
    mask = np.zeros((B, 1, L, C), dtype=bool)
    valid: List[int] = []
    write: List[int] = []
    last: List[int] = []
    key_pos = np.arange(C)[None, :]
    for b, it in enumerate(items):
        n = len(it.tokens)
        tokens[b, :n] = it.tokens
        ctx = context_slots(it, block_size)
        read[b, : len(ctx)] = ctx
        query_pos = it.start_pos + np.arange(L)[:, None]
        # padded query rows (i >= n) still see key 0, so softmax never runs over an empty row
        mask[b, 0] = (key_pos <= query_pos) & (key_pos < it.context_len)
        valid.extend(b * L + i for i in range(n))
        write.extend(it.slots)
        last.append(b * L + n - 1)
    return BatchLayout(
        tokens=tokens,
        offsets=np.array([it.start_pos for it in items], dtype=np.int32),
        valid=np.array(valid, dtype=np.int32),
        write_slots=np.array(write, dtype=np.int32),
        read_slots=read,
        mask=mask,
        last=np.array(last, dtype=np.int32),
    )


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
        max_pad_ratio: float = DEFAULT_MAX_PAD_RATIO,
    ) -> None:
        from mlx_lm import load

        loaded_model, tokenizer = load(model)
        self._setup(loaded_model, tokenizer, model_name or model.rstrip("/").split("/")[-1], kv_tokens, block_size, max_pad_ratio)

    @classmethod
    def from_loaded(
        cls,
        model: Any,
        tokenizer: Any,
        model_name: str = "model",
        kv_tokens: int = 8192,
        block_size: int = 16,
        max_pad_ratio: float = DEFAULT_MAX_PAD_RATIO,
    ) -> "MLXBackend":
        """Wrap an mlx-lm model already in memory (tests build small ones with random weights)."""
        backend = cls.__new__(cls)
        backend._setup(model, tokenizer, model_name, kv_tokens, block_size, max_pad_ratio)
        return backend

    def _setup(self, model: Any, tokenizer: Any, model_name: str, kv_tokens: int, block_size: int, max_pad_ratio: float) -> None:
        import mlx.core as mx

        if max_pad_ratio < 1.0:
            raise ValueError("max_pad_ratio must be >= 1")
        self.mx = mx
        self.model, self.tokenizer = model, tokenizer
        self.model_name = model_name
        args = self.model.args
        self.n_layers = len(self.model.layers)
        self.n_kv = getattr(args, "num_key_value_heads", None) or args.num_attention_heads
        self.head_dim = getattr(args, "head_dim", None) or args.hidden_size // args.num_attention_heads
        self.block_size = block_size
        self.max_pad_ratio = max_pad_ratio
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
        out: Dict[str, np.ndarray] = {}
        for group in plan_groups([len(it.tokens) for it in items], self.max_pad_ratio):
            out.update(self._forward_group([items[i] for i in group]))
        return out

    def _forward_group(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        mx = self.mx
        lay = build_layout(items, self.block_size)
        offsets = mx.array(lay.offsets)
        valid, write = mx.array(lay.valid), mx.array(lay.write_slots)
        read, mask = mx.array(lay.read_slots), mx.array(lay.mask)
        caches = [_PagedLayerCache(self, layer, offsets, valid, write, read, mask) for layer in range(self.n_layers)]
        logits = self.model(mx.array(lay.tokens), cache=caches)
        B, L, V = logits.shape
        last = logits.reshape(B * L, V)[mx.array(lay.last)].astype(mx.float32)
        # the pools were updated lazily inside the graph: materialize them with the logits
        mx.eval(last, *self.k_pool, *self.v_pool)
        rows = np.array(last)
        return {it.seq_id: rows[b] for b, it in enumerate(items) if it.want_logits}


class _PagedLayerCache:
    """mlx-lm's cache interface backed by the shared block pool of one layer, for a whole batch."""

    def __init__(self, backend: MLXBackend, layer: int, offsets, valid, write_slots, read_slots, mask) -> None:
        self.backend = backend
        self.layer = layer
        self.offset = offsets
        self._valid = valid
        self._write = write_slots
        self._read = read_slots
        self._mask = mask

    def update_and_fetch(self, keys, values):
        # keys: [B, n_kv, L, d] -> real tokens as [N, n_kv, d], written at their slots
        B, H, L, D = keys.shape
        kp, vp = self.backend.k_pool[self.layer], self.backend.v_pool[self.layer]
        k = keys.transpose(0, 2, 1, 3).reshape(B * L, H, D)[self._valid].astype(kp.dtype)
        v = values.transpose(0, 2, 1, 3).reshape(B * L, H, D)[self._valid].astype(vp.dtype)
        kp[self._write] = k
        vp[self._write] = v
        self.backend.k_pool[self.layer], self.backend.v_pool[self.layer] = kp, vp
        # context of every row: [B, C, n_kv, d] -> [B, n_kv, C, d]
        K = kp[self._read].transpose(0, 2, 1, 3).astype(keys.dtype)
        V = vp[self._read].transpose(0, 2, 1, 3).astype(values.dtype)
        return K, V

    def make_mask(self, *args, **kwargs):
        return self._mask

    @property
    def state(self):
        return ()

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return False
