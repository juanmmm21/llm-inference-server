"""A deterministic stand-in model for tests: bytes in, "next byte" out.

The tokenizer is UTF-8 bytes (ids 0-255, EOS = 256). The "model" predicts the byte after the
last one of the context, cyclically within lowercase letters, and EOS after ``stop_after``
generated tokens if set. It records every forward call so tests can check what was batched.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .base import Backend, BatchItem

EOS = 256
VOCAB = 257


class ToyBackend(Backend):
    name = "toy"
    supports_prefix_cache = True

    def __init__(self, kv_tokens: int = 4096, max_len: int = 2048, model_name: str = "toy") -> None:
        self.model_name = model_name
        self._kv_tokens = kv_tokens
        self._max_len = max_len
        self.calls: List[List[BatchItem]] = []
        self.released: List[str] = []
        self.contexts: Dict[str, List[int]] = {}

    @property
    def eos_token_ids(self) -> Set[int]:
        return {EOS}

    @property
    def max_model_len(self) -> int:
        return self._max_len

    def kv_capacity_tokens(self) -> int:
        return self._kv_tokens

    def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def forward(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        self.calls.append(list(items))
        out: Dict[str, np.ndarray] = {}
        for item in items:
            ctx = self.contexts.setdefault(item.seq_id, [])
            del ctx[item.start_pos :]
            ctx.extend(item.tokens)
            if item.want_logits:
                logits = np.full(VOCAB, -10.0)
                last = ctx[-1]
                nxt = (last - 97 + 1) % 26 + 97 if 97 <= last <= 122 else 97
                logits[nxt] = 10.0
                out[item.seq_id] = logits
        return out

    def release(self, seq_id: str) -> None:
        self.released.append(seq_id)
        self.contexts.pop(seq_id, None)

    def chat_template(self) -> Optional[str]:
        return "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: "
