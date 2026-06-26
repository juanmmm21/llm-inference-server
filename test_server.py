import asyncio
import unittest
import httpx
from server import app


class TestLLMInferenceServer(unittest.IsolatedAsyncioTestCase):
    """
    Suite de pruebas asincronas para validar los endpoints del servidor de inferencia,
    el streaming y la logica de agrupamiento dinamico.
    """

    async def asyncSetUp(self) -> None:
        # Reseteamos el estado de startup en FastAPI para permitir su reinicializacion en cada test
        app.router.startup_triggered = False

    async def test_sync_generate(self) -> None:
        """
        Verifica una peticion basica de generacion de texto.
        """
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"prompt": "Describir el flujo de RAG semantico", "max_tokens": 30}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("generated_text", data)
            self.assertIn("RAG", data["generated_text"])
            self.assertTrue(data["latency_ms"] > 0)

    async def test_stream_generate(self) -> None:
        """
        Verifica que el endpoint de streaming retorne eventos SSE validos.
        """
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            async with ac.stream(
                "POST",
                "/generate_stream",
                json={"prompt": "Explicar indexacion de vectores HNSW", "max_tokens": 20}
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers.get("content-type", ""))
                
                chunks = []
                async for line in response.aiter_lines():
                    if line:
                        chunks.append(line)
                        if len(chunks) >= 4:
                            break
                            
                self.assertTrue(len(chunks) > 0)
                self.assertTrue(any("event:" in c or "data:" in c for c in chunks))

    async def test_dynamic_batching_concurrency(self) -> None:
        """
        Prueba que multiples peticiones simultaneas sean agrupadas
        en un unico lote (batch) por el worker en segundo plano.
        """
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            # Lanzamos 3 peticiones de forma paralela en la ventana de 50ms
            tasks = [
                ac.post("/generate", json={"prompt": "rag tokens tutorial", "max_tokens": 10}),
                ac.post("/generate", json={"prompt": "vector embeddings HNSW", "max_tokens": 10}),
                ac.post("/generate", json={"prompt": "agent planning ReAct", "max_tokens": 10}),
            ]
            responses = await asyncio.gather(*tasks)
            
            for r in responses:
                self.assertEqual(r.status_code, 200)
                data = r.json()
                # Deberian haberse agrupado las 3 en el lote
                self.assertEqual(data["batch_size_processed"], 3)


if __name__ == "__main__":
    unittest.main()
