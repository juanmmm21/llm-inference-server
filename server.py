import asyncio
import logging
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from inference_engine import InferenceEngine

# Configuracion de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_inference_server")

app = FastAPI(
    title="LLM Inference Server",
    description="Servidor de inferencia de alto rendimiento con agrupamiento dinamico de peticiones y streaming."
)

# Inicializamos el motor de inferencia (por defecto simulado para portabilidad offline)
engine = InferenceEngine(use_local_model=False)

# Cola para el despachador de agrupamiento dinamico (Dynamic Batching)
_batch_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None
_active_loop: Optional[asyncio.AbstractEventLoop] = None

def get_queue() -> asyncio.Queue:
    global _batch_queue, _worker_task, _active_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _batch_queue is None or _active_loop != current_loop:
        if _worker_task is not None and not _worker_task.done():
            _worker_task.cancel()
        _batch_queue = asyncio.Queue()
        _active_loop = current_loop
        if current_loop is not None:
            _worker_task = asyncio.create_task(dynamic_batch_worker())
    elif _worker_task is None or _worker_task.done():
        if current_loop is not None:
            _worker_task = asyncio.create_task(dynamic_batch_worker())

    return _batch_queue

# Constantes del agrupamiento dinamico
BATCH_WINDOW_SECONDS = 0.05  # Ventana de 50ms para acumular peticiones
MAX_BATCH_SIZE = 16          # Capacidad maxima de agrupacion en una llamada al motor


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="El prompt de entrada para el modelo.")
    max_tokens: int = Field(50, ge=1, le=512, description="Numero maximo de tokens a generar.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperatura de muestreo.")


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    latency_ms: float
    batch_size_processed: int


class QueueItem:
    """Clase interna para encapsular los prompts y canalizar su resolucion asincrona."""
    def __init__(self, prompt: str, max_tokens: int) -> None:
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()


async def dynamic_batch_worker() -> None:
    """
    Worker en segundo plano encargado de agrupar peticiones concorrentes
    y enviarlas al modelo en lotes unificados (Dynamic Batching).
    """
    logger.info("Iniciando worker de dynamic batching...")
    queue = get_queue()
    while True:
        try:
            # Esperamos a que entre la primera peticion
            first_item = await queue.get()
            items = [first_item]
            
            # Esperamos una pequeña ventana de tiempo para capturar mas peticiones entrantes
            start_time = asyncio.get_event_loop().time()
            while len(items) < MAX_BATCH_SIZE:
                time_elapsed = asyncio.get_event_loop().time() - start_time
                time_remaining = BATCH_WINDOW_SECONDS - time_elapsed
                if time_remaining <= 0:
                    break
                    
                try:
                    # Intentamos recuperar elementos sin bloquear indefinidamente
                    next_item = await asyncio.wait_for(queue.get(), timeout=time_remaining)
                    items.append(next_item)
                except asyncio.TimeoutError:
                    break
            
            # Extraemos los prompts y procesamos el lote en el motor de inferencia
            prompts = [item.prompt for item in items]
            max_tokens_lote = max(item.max_tokens for item in items)
            
            logger.info(f"Procesando lote dinamico de tamaño: {len(items)}")
            
            # Ejecutamos la inferencia en un hilo aparte para no bloquear el bucle de eventos
            loop = asyncio.get_event_loop()
            start_inference = asyncio.get_event_loop().time()
            
            results = await loop.run_in_executor(
                None, 
                engine.generate_batch, 
                prompts, 
                max_tokens_lote
            )
            
            latency_ms = (asyncio.get_event_loop().time() - start_inference) * 1000.0
            
            # Resolvemos los futures de cada peticion individual con su respuesta
            for item, generated_text in zip(items, results):
                if not item.future.cancelled():
                    item.future.set_result((generated_text, len(items), latency_ms))
                    
            for _ in range(len(items)):
                queue.task_done()
                
        except Exception as e:
            logger.error(f"Error critico en worker de batching: {str(e)}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event() -> None:
    """Arranca el despachador dinamico al iniciar el servidor."""
    global _batch_queue
    _batch_queue = asyncio.Queue()  # Forzamos recreacion para enlazar al bucle de eventos activo
    asyncio.create_task(dynamic_batch_worker())


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    """
    Endpoint para generacion sincrona que aprovecha el Dynamic Batching.
    """
    start_time = asyncio.get_event_loop().time()
    
    # Creamos un elemento de cola y lo registramos
    queue = get_queue()
    item = QueueItem(request.prompt, request.max_tokens)
    await queue.put(item)
    
    # Esperamos a que el worker resuelva la llamada
    try:
        generated_text, batch_size, inference_latency = await item.future
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la generacion: {str(e)}")
        
    total_latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
    
    # Estimacion burda de tokens
    tokens = len(generated_text.split())
    
    return GenerationResponse(
        prompt=request.prompt,
        generated_text=generated_text,
        tokens_generated=tokens,
        latency_ms=total_latency_ms,
        batch_size_processed=batch_size
    )


@app.post("/generate_stream")
async def generate_stream(request: GenerationRequest):
    """
    Endpoint de streaming mediante Server-Sent Events (SSE).
    """
    async def sse_generator():
        # Ejecutamos el streaming de tokens directo del motor
        loop = asyncio.get_event_loop()
        
        def run_stream():
            return engine.generate_stream(request.prompt, request.max_tokens)
            
        generator = await loop.run_in_executor(None, run_stream)
        
        try:
            for token_chunk in generator:
                yield {
                    "event": "token",
                    "id": str(uuid.uuid4()),
                    "data": token_chunk
                }
            # Indicamos finalizacion del stream
            yield {
                "event": "done",
                "id": str(uuid.uuid4()),
                "data": "[DONE]"
            }
        except Exception as e:
            logger.error(f"Error en streaming de respuesta: {str(e)}")
            yield {
                "event": "error",
                "id": str(uuid.uuid4()),
                "data": f"Error: {str(e)}"
            }

    return EventSourceResponse(sse_generator())
