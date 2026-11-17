"""Paged KV cache: fixed-size blocks from one shared pool, a block table per sequence.

As in vLLM's PagedAttention, the key/value cache is not one contiguous buffer per sequence (which
wastes memory reserving the maximum length) but a pool of ``num_blocks`` blocks of
``block_size`` token slots. A sequence owns a *block table*: the list of physical blocks that hold
its tokens in order, so token ``i`` lives in slot ``table[i // block_size] * block_size + i %
block_size``. Blocks are allocated as the sequence grows and returned when it finishes, so memory
in use tracks the tokens actually cached.

Blocks are reference-counted. Full blocks are also indexed by a hash of their tokens *and of every
token before them* (the prefix chain), so a new sequence whose prompt starts like an earlier one
(the same system prompt, the same few-shot examples) reuses those blocks instead of recomputing
them: **prefix caching**. A shared block is never written: only full blocks are shared, and a
sequence always writes into a block it owns alone.

This module only does the bookkeeping (which slot holds which token). Backends that own their
KV memory use the slot mapping to read and write it (the torch reference backend does); backends
that manage KV memory themselves (llama.cpp) use the manager for admission control, so the
scheduler never admits more tokens than the cache can hold.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation needs more free blocks than the pool has."""


@dataclass
class _Block:
    id: int
    ref: int = 0
    hash: Optional[str] = None  # set only for full blocks registered in the prefix cache


@dataclass
class BlockTable:
    """The physical blocks of one sequence and how many token slots of them are filled."""

    blocks: List[int] = field(default_factory=list)
    num_tokens: int = 0
    # Prefix-chain hash of each full block of this sequence (for registering in the cache).
    block_hashes: List[str] = field(default_factory=list)


def _chain(previous: str, tokens: Sequence[int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(previous.encode())
    h.update(b"|")
    h.update(",".join(map(str, tokens)).encode())
    return h.hexdigest()


class PagedKVCacheManager:
    """Allocates KV-cache blocks to sequences.

    Args:
        num_blocks: blocks in the pool.
        block_size: token slots per block.
        enable_prefix_caching: reuse full blocks whose prefix chain matches.
    """

    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._blocks = [_Block(i) for i in range(num_blocks)]
        self._free: List[int] = list(range(num_blocks - 1, -1, -1))  # stack: pop() gives low ids first
        # Prefix cache: hash -> block id. Blocks with ref == 0 stay here (evictable, LRU order) so
        # a later sequence can still hit them; they are reclaimed only when the free list is empty.
        self._cached: Dict[str, int] = {}
        self._evictable: "OrderedDict[int, None]" = OrderedDict()
        self.tables: Dict[str, BlockTable] = {}
        self.prefix_hits = 0
        self.prefix_queries = 0

    # ------------------------------------------------------------------ capacity

    @property
    def num_free_blocks(self) -> int:
        """Blocks that can be handed out now (free or cached but unreferenced)."""
        return len(self._free) + len(self._evictable)

    @property
    def usage(self) -> float:
        """Fraction of the pool referenced by live sequences."""
        return 1.0 - self.num_free_blocks / self.num_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def can_allocate(self, num_tokens: int, reserve: int = 0) -> bool:
        """Whether a new sequence of ``num_tokens`` fits, keeping ``reserve`` blocks free."""
        return self.blocks_needed(num_tokens) + reserve <= self.num_free_blocks

    def can_append(self, seq_id: str) -> bool:
        table = self.tables[seq_id]
        needs_block = table.num_tokens % self.block_size == 0
        return not needs_block or self.num_free_blocks > 0

    # ------------------------------------------------------------------ allocation

    def _take_free(self) -> int:
        if self._free:
            bid = self._free.pop()
        elif self._evictable:
            bid, _ = self._evictable.popitem(last=False)  # least recently released
            old = self._blocks[bid].hash
            if old is not None and self._cached.get(old) == bid:
                del self._cached[old]
            self._blocks[bid].hash = None
        else:
            raise OutOfBlocksError("no free KV-cache blocks")
        self._blocks[bid].ref = 1
        return bid

    def _incref(self, bid: int) -> None:
        block = self._blocks[bid]
        if block.ref == 0:
            self._evictable.pop(bid, None)
        block.ref += 1

    def _decref(self, bid: int) -> None:
        block = self._blocks[bid]
        block.ref -= 1
        if block.ref == 0:
            if block.hash is not None and self._cached.get(block.hash) == bid:
                self._evictable[bid] = None  # keep its contents for future prefix hits
            else:
                block.hash = None
                self._free.append(bid)

    def allocate(self, seq_id: str, tokens: Sequence[int]) -> int:
        """Give ``seq_id`` the blocks for its prompt ``tokens``.

        Returns how many leading tokens are already cached (their K/V need not be computed);
        always leaves at least the last prompt token to compute, so the backend has logits.
        """
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already has blocks")
        n = len(tokens)
        table = BlockTable()
        cached_tokens = 0
        prev = ""
        full = n // self.block_size
        if self.enable_prefix_caching:
            self.prefix_queries += 1
            for b in range(full):
                chunk = tokens[b * self.block_size : (b + 1) * self.block_size]
                h = _chain(prev, chunk)
                bid = self._cached.get(h)
                # never reuse the block holding the last prompt token: it must be computed
                if bid is None or (b + 1) * self.block_size >= n:
                    break
                self._incref(bid)
                table.blocks.append(bid)
                table.block_hashes.append(h)
                prev = h
                cached_tokens += self.block_size
            if cached_tokens:
                self.prefix_hits += 1
        needed = self.blocks_needed(n) - len(table.blocks)
        if needed > self.num_free_blocks:
            for bid in table.blocks:
                self._decref(bid)
            raise OutOfBlocksError(f"{needed} blocks needed, {self.num_free_blocks} free")
        for _ in range(needed):
            table.blocks.append(self._take_free())
        table.num_tokens = n
        self.tables[seq_id] = table
        # The new blocks enter the prefix cache only once their K/V are computed
        # (``register_tokens``): a block reserved but not yet written must never be shared.
        return cached_tokens

    def append_slot(self, seq_id: str) -> int:
        """Reserve the slot for one more token of ``seq_id``; returns its physical slot index."""
        table = self.tables[seq_id]
        if table.num_tokens % self.block_size == 0 and table.num_tokens // self.block_size == len(table.blocks):
            table.blocks.append(self._take_free())
        slot = self.slot(seq_id, table.num_tokens)
        table.num_tokens += 1
        return slot

    def register_tokens(self, seq_id: str, tokens: Sequence[int]) -> None:
        """Offer the full blocks covering ``tokens`` — which must already be computed — to the
        prefix cache. Called as prefill advances and when a sequence finishes, so generated text
        can also be a later prompt's prefix (a chat's next turn)."""
        self._register_full_blocks(seq_id, tokens)

    def _register_full_blocks(self, seq_id: str, tokens: Sequence[int]) -> None:
        if not self.enable_prefix_caching:
            return
        table = self.tables[seq_id]
        full = min(len(tokens), table.num_tokens) // self.block_size
        prev = table.block_hashes[-1] if table.block_hashes else ""
        for b in range(len(table.block_hashes), full):
            h = _chain(prev, tokens[b * self.block_size : (b + 1) * self.block_size])
            table.block_hashes.append(h)
            bid = table.blocks[b]
            if h not in self._cached:
                self._cached[h] = bid
                self._blocks[bid].hash = h
            prev = h

    def free(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        if table is None:
            return
        # Release in reverse, so the tail of a sequence is evicted before its shared prefix.
        for bid in reversed(table.blocks):
            self._decref(bid)

    # ------------------------------------------------------------------ addressing

    def slot(self, seq_id: str, position: int) -> int:
        table = self.tables[seq_id]
        block = table.blocks[position // self.block_size]
        return block * self.block_size + position % self.block_size

    def slots(self, seq_id: str, start: int, end: int) -> List[int]:
        return [self.slot(seq_id, p) for p in range(start, end)]

    def block_table(self, seq_id: str) -> Tuple[List[int], int]:
        table = self.tables[seq_id]
        return list(table.blocks), table.num_tokens

    def stats(self) -> Dict[str, float]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": self.num_free_blocks,
            "usage": round(self.usage, 4),
            "sequences": len(self.tables),
            "prefix_hit_rate": round(self.prefix_hits / self.prefix_queries, 4) if self.prefix_queries else 0.0,
        }
