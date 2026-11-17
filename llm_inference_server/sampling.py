"""Next-token sampling from logits: greedy, temperature, top-k and nucleus (top-p)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .scheduler import SamplingParams


def sample(logits: np.ndarray, params: SamplingParams, rng: Optional[np.random.Generator] = None) -> int:
    if params.temperature <= 0.0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    z = logits.astype(np.float64) / params.temperature
    if params.top_k and params.top_k < z.size:
        cut = np.partition(z, -params.top_k)[-params.top_k]
        z = np.where(z < cut, -np.inf, z)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if params.top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, params.top_p) + 1)]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
    return int(rng.choice(p.size, p=p))
