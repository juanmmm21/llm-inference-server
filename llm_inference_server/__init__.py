"""LLM inference server with an OpenAI-compatible API."""

from .inference_engine import InferenceEngine, InferenceEngineError
from .server import app

__version__ = "0.1.0"

__all__ = ['InferenceEngine', 'InferenceEngineError', 'app']
