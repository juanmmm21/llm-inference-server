"""``python -m llm_inference_server``: serve a model with an OpenAI-compatible API.

    python -m llm_inference_server --backend llamacpp --model qwen2.5-0.5b-instruct-q4_k_m.gguf
    python -m llm_inference_server --backend mlx --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
    python -m llm_inference_server --backend torch --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import threading


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm_inference_server")
    parser.add_argument("--backend", choices=["llamacpp", "mlx", "torch", "toy"], default="llamacpp")
    parser.add_argument("--model", help="GGUF path (llamacpp), HF repo or path (mlx, torch)")
    parser.add_argument("--served-name", help="model id reported by /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-tokens", type=int, default=8192, help="KV-cache capacity in tokens")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--default-max-tokens", type=int, default=256)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import uvicorn

    from .backends import load_backend
    from .engine import AsyncLLMEngine, LLMEngine
    from .server import create_app

    if args.backend == "llamacpp":
        backend = load_backend(
            "llamacpp",
            model_path=args.model,
            n_ctx=args.kv_tokens,
            max_num_seqs=args.max_num_seqs,
            n_batch=args.max_num_batched_tokens,
            n_threads=args.threads,
            model_name=args.served_name,
        )
    elif args.backend == "toy":
        backend = load_backend("toy", kv_tokens=args.kv_tokens)
    else:
        backend = load_backend(args.backend, model=args.model, kv_tokens=args.kv_tokens, block_size=args.block_size, threads=args.threads)
    engine = LLMEngine(
        backend,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    # The engine loop owns the main thread (llama.cpp's OpenMP threads are fastest there);
    # uvicorn serves HTTP from a second thread with its own event loop.
    async_engine = AsyncLLMEngine(engine, start_thread=False)
    app = create_app(async_engine, model_name=args.served_name, default_max_tokens=args.default_max_tokens)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    http = threading.Thread(target=server.run, name="http", daemon=True)
    http.start()
    try:
        async_engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        async_engine.shutdown()
        http.join(timeout=5)


if __name__ == "__main__":
    main()
