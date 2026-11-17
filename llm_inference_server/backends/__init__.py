"""Model backends. Each one runs a batch of sequences one step forward and returns logits."""

from .base import Backend, BatchItem
from .toy import ToyBackend

__all__ = ["Backend", "BatchItem", "ToyBackend", "load_backend"]


def load_backend(name: str, **kwargs) -> Backend:
    """``llamacpp``, ``mlx``, ``torch`` or ``toy``; heavy dependencies are imported only here."""
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(**kwargs)
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(**kwargs)
    if name == "torch":
        from .torch_ref import TorchPagedBackend

        return TorchPagedBackend(**kwargs)
    if name == "toy":
        return ToyBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
