"""Compatibilidad: el código vive en el paquete `llm_inference_server`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `server` directamente."""

from llm_inference_server.server import *  # noqa: F401,F403
from llm_inference_server.server import create_app  # noqa: F401
