import time
import subprocess
import asyncio
import httpx


async def listen_stream(client: httpx.AsyncClient, prompt: str, request_id: int):
    """
    Se conecta al endpoint de streaming SSE e imprime los chunks recibidos.
    """
    print(f"[Cliente {request_id}] Iniciando stream para prompt: '{prompt}'")
    try:
        async with client.stream(
            "POST", 
            "/generate_stream", 
            json={"prompt": prompt, "max_tokens": 40}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    token = line[6:]
                    if token == "[DONE]":
                        print(f"\n[Cliente {request_id}] Stream completado.")
                    else:
                        print(f"{token}", end="", flush=True)
    except Exception as e:
        print(f"\n[Cliente {request_id}] Error en stream: {str(e)}")


async def make_sync_request(client: httpx.AsyncClient, prompt: str, request_id: int):
    """
    Realiza una consulta sincrona y muestra las metricas de Dynamic Batching.
    """
    print(f"[Batch Client {request_id}] Enviando prompt: '{prompt}'")
    try:
        response = await client.post(
            "/generate", 
            json={"prompt": prompt, "max_tokens": 40}
        )
        if response.status_code == 200:
            data = response.json()
            print(
                f"\n[Batch Client {request_id}] Respuesta recibida:\n"
                f"  -> Texto: '{data['generated_text']}'\n"
                f"  -> Tamaño Lote Procesado: {data['batch_size_processed']} peticiones simultaneas\n"
                f"  -> Latencia Total: {data['latency_ms']:.2f} ms\n"
            )
        else:
            print(f"[Batch Client {request_id}] Error: {response.status_code}")
    except Exception as e:
        print(f"[Batch Client {request_id}] Exception: {str(e)}")


async def main():
    print("=" * 60)
    print("      Demostracion del Servidor de Inferencia LLM      ")
    print("=" * 60)
    
    # 1. Arrancamos el servidor FastAPI localmente en segundo plano
    print("Arrancando el servidor FastAPI en el puerto 8000...")
    server_process = subprocess.Popen(
        ["./.venv/bin/uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"]
    )
    
    # Damos tiempo a que se inicialice
    time.sleep(2.0)
    
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
            # 2. Demostracion de streaming individual
            print("\n--- TEST 1: Streaming de un prompt individual ---")
            await listen_stream(client, "Explicar RAG y embeddings", 1)
            
            # 3. Demostracion de Dynamic Batching
            # Lanzamos 4 peticiones concurrentes simultaneamente para forzar el agrupamiento
            print("\n--- TEST 2: Multiples peticiones concurrentes (Dynamic Batching) ---")
            tasks = [
                make_sync_request(client, "RAG base de datos vectorial HNSW", 1),
                make_sync_request(client, "Tokenizacion BPE en procesamiento de lenguaje", 2),
                make_sync_request(client, "Agentes autonomos y bucles ReAct", 3),
                make_sync_request(client, "Que es una base de datos vectorial?", 4),
            ]
            
            # asyncio.gather lanzara las corrutinas en paralelo cayendo en la ventana de batching
            await asyncio.gather(*tasks)
            
    finally:
        # 4. Apagamos el servidor
        print("\nApagando el servidor de inferencia...")
        server_process.terminate()
        server_process.wait()
        print("Servidor apagado correctamente.")


if __name__ == "__main__":
    asyncio.run(main())
