# LLM Inference Server

Un servidor de inferencia de alto rendimiento desarrollado en Python y optimizado para la ejecucion de modelos de lenguaje causal. Este modulo implementa tecnicas avanzadas de optimizacion de rendimiento como el agrupamiento dinamico de peticiones (Dynamic Request Batching) y la emision de respuestas mediante streaming con Server-Sent Events (SSE).

## Arquitectura y Componentes Tecnicos

El servidor consta de los siguientes componentes nucleares:

1.  **Motor de Inferencia (Inference Engine):** 
    *   Gestiona la inferencia local con soporte para modelos CausalLM de Hugging Face y ejecucion en dispositivos acelerados (MPS/CUDA).
    *   Implementa un sistema de contingencia (fallback) semantico offline determinista que emite texto formateado coherentemente segun las palabras clave encontradas en el prompt. Esto asegura que la infraestructura pueda ser desplegada y probada de forma offline.
2.  **Agrupamiento Dinamico de Peticiones (Dynamic Batching):**
    *   Un worker asincrono en segundo plano monitoriza una cola de tareas (`asyncio.Queue`).
    *   Cuando entra una peticion, se abre una ventana temporal configurable (50 milisegundos). Todas las peticiones concurrentes recibidas en esa ventana se agrupan en un unico lote de entrada (lote maximo de 16).
    *   El lote se procesa en paralelo usando el motor de inferencia ejecutandose en un ejecutor de hilos separado (`run_in_executor`) para evitar el bloqueo del bucle de eventos principal de FastAPI.
3.  **Transmision por Streaming (SSE):**
    *   Una API basada en flujos que utiliza Server-Sent Events (SSE) a traves de `sse-starlette` para entregar fragmentos de texto en tiempo real conforme son generados por el motor, mejorando drasticamente la latencia percibida por el usuario.

## Requisitos e Instalacion

El modulo requiere Python 3.8 o superior y las siguientes dependencias principales:

*   FastAPI
*   Uvicorn
*   SSE-Starlette
*   PyTorch
*   HTTPX (para pruebas y cliente de demostracion)

Para instalar las dependencias en su entorno local, ejecute:

```bash
pip install -r requirements.txt
```

## Estructura de Archivos

*   `inference_engine.py`: Clase de abstraccion para la ejecucion de modelos y simulacion offline.
*   `server.py`: Servidor FastAPI con la cola de tareas de dynamic batching y endpoints.
*   `test_server.py`: Suite de pruebas automatizadas que verifican la concurrencia y los endpoints.
*   `example.py`: Script demostrativo que arranca el servidor, ejecuta un streaming y lanza peticiones concurrentes para comprobar el agrupamiento en tiempo real.
