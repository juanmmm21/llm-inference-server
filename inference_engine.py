"""Compatibilidad: el código vive en el paquete `llm_inference_server`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `inference_engine`
directamente."""

from llm_inference_server.inference_engine import *  # noqa: F401,F403
from llm_inference_server.inference_engine import InferenceEngine, InferenceEngineError  # noqa: F401
