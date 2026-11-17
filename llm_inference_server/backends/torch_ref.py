"""Reference backend in PyTorch with a real paged KV cache (Qwen2 / Llama architectures).

The forward pass is written here, not borrowed: embeddings, RMSNorm, rotary embeddings,
grouped-query attention and the SwiGLU MLP, with the weights of a Hugging Face checkpoint. What
makes it a serving backend:

* **Paged KV memory.** Each layer owns ``k_cache`` and ``v_cache`` tensors of shape
  ``[num_blocks * block_size, n_kv_heads, head_dim]`` — the block pool. New keys/values are
  written to the slots the :class:`~llm_inference_server.kv_cache.PagedKVCacheManager` assigned
  (``BatchItem.slots``) and each sequence's attention reads its own blocks back through its block
  table. No sequence owns a contiguous buffer, and prefix-cache hits are real (shared blocks are
  read, not recomputed).
* **One batch per step.** The tokens of every scheduled sequence — prefill chunks and decode
  tokens together — are flattened into one ``[T, hidden]`` tensor, so every linear layer runs once
  per step for the whole batch; only attention is computed per sequence (each has its own length).

It favours clarity over speed (no fused kernels, float32 by default) and is validated against the
Hugging Face implementation in the tests. Use the llama.cpp backend for throughput.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .base import Backend, BatchItem


class TorchPagedBackend(Backend):
    name = "torch"
    supports_prefix_cache = True

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        kv_tokens: int = 8192,
        block_size: int = 16,
        dtype: str = "float32",
        threads: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.model_name = model_name or os.path.basename(model.rstrip("/"))
        self.dtype = getattr(torch, dtype)
        cfg = AutoConfig.from_pretrained(model)
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=self.dtype)
        sd = {k: v.detach() for k, v in hf.state_dict().items()}
        del hf
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.eps = cfg.rms_norm_eps
        self.embed = sd["model.embed_tokens.weight"]
        self.lm_head = sd.get("lm_head.weight", self.embed)
        self.final_norm = sd["model.norm.weight"]
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append(
                {
                    name: sd.get(p + key)
                    for name, key in {
                        "ln1": "input_layernorm.weight",
                        "ln2": "post_attention_layernorm.weight",
                        "wq": "self_attn.q_proj.weight",
                        "bq": "self_attn.q_proj.bias",
                        "wk": "self_attn.k_proj.weight",
                        "bk": "self_attn.k_proj.bias",
                        "wv": "self_attn.v_proj.weight",
                        "bv": "self_attn.v_proj.bias",
                        "wo": "self_attn.o_proj.weight",
                        "gate": "mlp.gate_proj.weight",
                        "up": "mlp.up_proj.weight",
                        "down": "mlp.down_proj.weight",
                    }.items()
                }
            )
        theta = _rope_theta(cfg)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.block_size = block_size
        self.num_slots = (kv_tokens // block_size) * block_size
        shape = (self.num_slots, self.n_kv, self.head_dim)
        self.k_cache = [torch.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        self._kv_tokens = self.num_slots
        eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
        gen_eos = getattr(self.tokenizer, "eos_token_id", None)
        self._eos = {int(t) for t in eos + ([gen_eos] if gen_eos is not None else []) if t is not None}
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in self.tokenizer.get_vocab() else None
        if im_end is not None:
            self._eos.add(int(im_end))

    # ------------------------------------------------------------------ facts and text

    @property
    def eos_token_ids(self) -> Set[int]:
        return self._eos

    @property
    def max_model_len(self) -> int:
        return int(getattr(self.cfg, "max_position_embeddings", 4096))

    def kv_capacity_tokens(self) -> int:
        return self._kv_tokens

    def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def apply_chat_template(self, messages):  # type: ignore[override]
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ math

    def _rms(self, x, w):
        torch = self.torch
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * w

    def _rope(self, x, positions):
        torch = self.torch
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [T, d/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[:, None, :].to(x.dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[:, None, :].to(x.dtype)
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    def forward(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            tokens = torch.tensor([t for it in items for t in it.tokens], dtype=torch.long)
            positions = torch.tensor([it.start_pos + j for it in items for j in range(len(it.tokens))], dtype=torch.long)
            slots = torch.tensor([s for it in items for s in it.slots], dtype=torch.long)
            # Per sequence: where its tokens sit in the flat batch and which slots hold its context.
            spans = []
            offset = 0
            for it in items:
                ctx_slots = self._context_slots(it)
                spans.append((offset, len(it.tokens), it.start_pos, ctx_slots))
                offset += len(it.tokens)
            h = self.embed[tokens]
            scale = 1.0 / math.sqrt(self.head_dim)
            group = self.n_heads // self.n_kv
            for li, w in enumerate(self.layers):
                x = self._rms(h, w["ln1"])
                q = torch.nn.functional.linear(x, w["wq"], w["bq"]).view(-1, self.n_heads, self.head_dim)
                k = torch.nn.functional.linear(x, w["wk"], w["bk"]).view(-1, self.n_kv, self.head_dim)
                v = torch.nn.functional.linear(x, w["wv"], w["bv"]).view(-1, self.n_kv, self.head_dim)
                q, k = self._rope(q, positions), self._rope(k, positions)
                self.k_cache[li][slots] = k  # write this step's K/V into their paged slots
                self.v_cache[li][slots] = v
                attn = torch.empty_like(q)
                for start, n, pos0, ctx_slots in spans:
                    K = self.k_cache[li][ctx_slots].repeat_interleave(group, dim=1)  # read through the block table
                    V = self.v_cache[li][ctx_slots].repeat_interleave(group, dim=1)
                    qi = q[start : start + n]
                    scores = torch.einsum("qhd,khd->hqk", qi, K) * scale
                    ctx_len = K.shape[0]
                    qpos = torch.arange(pos0, pos0 + n)[:, None]
                    kpos = torch.arange(ctx_len)[None, :]
                    scores = scores.masked_fill((kpos > qpos)[None], float("-inf"))
                    attn[start : start + n] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), V)
                h = h + torch.nn.functional.linear(attn.reshape(attn.shape[0], -1), w["wo"])
                x = self._rms(h, w["ln2"])
                mlp = torch.nn.functional.silu(torch.nn.functional.linear(x, w["gate"])) * torch.nn.functional.linear(x, w["up"])
                h = h + torch.nn.functional.linear(mlp, w["down"])
            last_rows = [start + n - 1 for (start, n, _, _), it in zip(spans, items) if it.want_logits]
            out: Dict[str, np.ndarray] = {}
            if last_rows:
                hidden = self._rms(h[last_rows], self.final_norm)
                logits = torch.nn.functional.linear(hidden, self.lm_head).float().numpy()
                for row, it in zip(logits, [it for it in items if it.want_logits]):
                    out[it.seq_id] = row
            return out

    def _context_slots(self, item: BatchItem):
        """Slots of positions ``0 .. context_len-1`` from the block table."""
        torch = self.torch
        ctx = item.context_len
        bs = self.block_size
        blocks = torch.tensor(item.block_table[: -(-ctx // bs)], dtype=torch.long)
        slots = (blocks[:, None] * bs + torch.arange(bs)[None, :]).reshape(-1)
        return slots[:ctx]


def _rope_theta(cfg) -> float:
    """RoPE base: ``config.rope_theta`` in older transformers, ``rope_parameters`` /
    ``rope_scaling`` in newer ones. Getting it wrong still produces fluent text (the argmax often
    survives) but wrong logits, which is why the tests compare against Hugging Face numerically."""
    theta = getattr(cfg, "rope_theta", None)
    for attr in ("rope_parameters", "rope_scaling"):
        params = getattr(cfg, attr, None)
        if theta is None and isinstance(params, dict):
            theta = params.get("rope_theta")
    return float(theta) if theta is not None else 10000.0
