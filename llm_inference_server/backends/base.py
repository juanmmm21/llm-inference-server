"""The contract between the engine and a model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set

import numpy as np

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@dataclass
class BatchItem:
    """One sequence's share of a model step: ``tokens`` at positions ``start_pos...``.

    ``slots`` are the physical KV-cache slots of those tokens and ``block_table`` the sequence's
    blocks (from :class:`~llm_inference_server.kv_cache.PagedKVCacheManager`): backends that keep
    their own paged KV memory read and write through them; others ignore them.
    """

    seq_id: str
    tokens: List[int]
    start_pos: int
    want_logits: bool
    slots: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)

    @property
    def context_len(self) -> int:
        return self.start_pos + len(self.tokens)


class Backend(ABC):
    """A model behind the engine."""

    name: str = "backend"
    model_name: str = "model"
    #: Whether the backend reads K/V through the manager's slots, so prefix-cache hits are valid.
    supports_prefix_cache: bool = False
    #: Whether one forward call can mix several sequences (true batching).
    batched: bool = True

    @property
    @abstractmethod
    def eos_token_ids(self) -> Set[int]: ...

    @property
    @abstractmethod
    def max_model_len(self) -> int: ...

    @abstractmethod
    def kv_capacity_tokens(self) -> int:
        """Tokens of KV memory available to all sequences together."""

    @abstractmethod
    def tokenize(self, text: str, add_special: bool = False) -> List[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    @abstractmethod
    def forward(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        """Run one step; return next-token logits for every item with ``want_logits``."""

    def release(self, seq_id: str) -> None:
        """Forget everything about ``seq_id`` (finished or preempted)."""

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]]) -> str:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(self.chat_template() or CHATML)
        return template.render(messages=list(messages), add_generation_prompt=True, bos_token="", eos_token="")

    def chat_template(self) -> str:
        return CHATML
