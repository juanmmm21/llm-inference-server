# LLM Inference Server

An LLM serving engine written from scratch: **continuous batching**, a **paged KV cache** with
prefix caching, three backends (**llama.cpp**, **MLX** and a PyTorch reference) and an
**OpenAI-compatible API** with streaming. Any OpenAI client works against it by changing the base
URL.

```bash
pip install "llm-inference-server[llamacpp] @ git+https://github.com/juanmmm21/llm-inference-server.git@v0.1.0"
python -m llm_inference_server --backend llamacpp --model qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --served-name qwen2.5-0.5b --max-num-seqs 8 --kv-tokens 8192 --threads 4

curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen2.5-0.5b", "messages": [{"role": "user", "content": "Hi!"}], "stream": true}'
```

## Architecture

```
HTTP thread (uvicorn)                         main thread (engine loop)
 POST /v1/chat/completions ──┐               ┌──────────────────────────────────────────────┐
   chat template, tokenize   │  submit       │ Scheduler.schedule()  ── iteration-level     │
   async token stream  ◄─────┼── tokens ─────│   decodes first, then admit + chunked prefill│
                             │               │   preempt (recompute) when blocks run out    │
                             │               │ PagedKVCacheManager  ── blocks, tables,      │
                             │               │   refcounts, prefix cache                    │
                             │               │ Backend.forward(batch) ── one step, all seqs │
                             │               │ sample → detokenize → stop strings → events  │
                             └───────────────└──────────────────────────────────────────────┘
```

### Continuous batching (`scheduler.py`)

The batch is re-formed at **every model step**. A request joins on the next step after it
arrives and leaves the step after it finishes; nobody waits for the longest request of a batch.
Each step: (1) every running sequence gets its next token — or the next chunk of its prompt; (2)
with the token budget left (`--max-num-batched-tokens`) waiting requests are admitted in arrival
order and prefilled, **in chunks** if the prompt is longer than the budget, so a long prompt never
stalls everyone's decoding. If the KV cache cannot give a running sequence a slot, the most
recently admitted sequence is **preempted**: its blocks are freed and it is recomputed later
(prompt plus what it had generated) without losing tokens.

### Paged KV cache (`kv_cache.py`)

As in vLLM's PagedAttention, KV memory is a pool of fixed-size blocks (`--block-size`, 16 tokens)
and each sequence has a *block table*; token *i* lives in slot
`table[i // B] * B + i % B`. Blocks are allocated as sequences grow and returned when they end,
so memory tracks the tokens actually cached instead of reserving the maximum length. Blocks are
reference-counted and full blocks are indexed by a hash of their tokens and of every token before
them: a prompt that starts like a previous one (the same system prompt, the next turn of a chat)
**reuses those blocks instead of recomputing them** (prefix caching). Only blocks whose K/V have
actually been computed enter the prefix cache.

### Backends (`backends/`)

| Backend | Model format | Batching in one forward | KV memory | Prefix caching |
| --- | --- | --- | --- | --- |
| `llamacpp` | GGUF | yes: all sequences in one `llama_batch`, one llama.cpp sequence id each | llama.cpp's own (unified, `n_ctx` cells); the block manager does admission and preemption | no |
| `torch` | Hugging Face (Qwen2 / Llama architectures) | yes: every linear layer runs once for all tokens of the step; attention per sequence | **our block pool**, read and written through the block tables | yes |
| `mlx` | `mlx-lm` checkpoints | no: one forward per sequence per step (mlx-lm takes one RoPE offset per call) | **our block pool**, behind mlx-lm's cache interface | yes |
| `toy` | — | yes | — | yes |

The torch backend is the reference implementation of paged attention here: its forward pass
(RMSNorm, RoPE, grouped-query attention, SwiGLU) is written in this repository and validated
against Hugging Face — logits within 3·10⁻⁵ and identical greedy tokens with batching, chunked
prefill and prefix-cache hits (`test_torch_backend.py`). The MLX backend reproduces `mlx-lm`'s
own greedy generation token for token with its KV in our pool (`test_mlx_backend.py`).

### API (`server.py`)

`POST /v1/chat/completions` and `POST /v1/completions` (with `stream`, `stream_options.include_usage`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`ignore_eos`), `GET /v1/models`, `GET /health`, `GET /metrics` (steps, tokens, mean batch size,
TTFT, preemptions, KV usage and prefix hit rate). Chat templates come from the model (the GGUF's
`tokenizer.chat_template`, rendered with sandboxed Jinja, or the Hugging Face tokenizer).

## Benchmark: against Ollama, same model, same machine

`bench/benchmark.py` streams chat completions (`temperature 0`, `max_tokens 128`, eight fixed
prompts with an explicit system message so both servers see the same tokens) at 1, 4 and 8
concurrent clients, 16 requests per level, after a warm-up request.

### Desktop CPU, profiled under load (2026-11-18)

* **Machine**: AMD Ryzen 7 7700X under WSL2 (16 threads visible, 7.5 GB RAM), no GPU; both servers
  pinned to 4 threads.
* **Model**: the same `qwen2.5-0.5b-instruct-q4_k_m.gguf` file for both servers.
* **llm-inference-server 0.1.0**: llama.cpp backend through the prebuilt llama-cpp-python 0.3.36
  wheel from PyPI, `--threads 4 --max-num-seqs 8 --kv-tokens 8192`, Python 3.12, with
  `--max-num-batched-tokens` set to 64, 128, 256 and 512 (the default).
* **Ollama 0.40.6**: native Linux binary (CPU), `OLLAMA_NUM_PARALLEL=8`, the GGUF imported with
  `bench/Modelfile` (`num_thread 4`, `num_ctx 2048`, plus `num_gpu 0`).
* All runs back to back, nothing else running.

Eight clients, by token budget per step:

| Server | `max_num_batched_tokens` | TTFT p50 (s) | TTFT p95 (s) | Decode per request p50 (tok/s) | Aggregate throughput (tok/s) |
| --- | --- | --- | --- | --- | --- |
| llm-inference-server | 64 | 0.538 | 1.005 | 17.4 | 129.6 |
| llm-inference-server | 128 | 0.639 | 0.923 | 18.5 | 135.8 |
| llm-inference-server | 256 | 0.871 | 0.877 | 19.0 | 137.6 |
| llm-inference-server | 512 | 0.851 | 0.875 | 19.3 | 138.8 |
| Ollama | — | 0.069 | 0.397 | 28.0 | 215.1 |

Default budget (512) against Ollama at every load level:

| Server | Clients | TTFT p50 (s) | TTFT p95 (s) | Decode per request p50 (tok/s) | Aggregate throughput (tok/s) |
| --- | --- | --- | --- | --- | --- |
| llm-inference-server | 1 | 0.116 | 0.136 | 91.6 | 84.5 |
| Ollama | 1 | 0.052 | 0.070 | 98.1 | 94.2 |
| llm-inference-server | 4 | 0.429 | 0.462 | 32.8 | 122.6 |
| Ollama | 4 | 0.177 | 0.294 | 48.0 | 179.4 |
| llm-inference-server | 8 | 0.851 | 0.875 | 19.3 | 138.8 |
| Ollama | 8 | 0.069 | 0.397 | 28.0 | 215.1 |

What the numbers say:

* **The token budget moves the median first token, not the gap.** The benchmark prompts are
  31 to 36 tokens, so with 8 clients arriving together a budget of 512 prefills all of them in
  one step (256, all but one) and the first tokens come out together (p50 ≈ p95 ≈ 0.87 s). A budget of 128
  admits four per step and 64 admits two: the first arrivals answer sooner and the last ones
  later. Going from 512 to 128 lowers the TTFT p50 by 25 % (0.85 → 0.64 s) for 2 % less
  throughput; 64 lowers it by 37 % (0.54 s) but costs 7 % of throughput and pushes p95 past 1 s.
  The default stays at 512; `--max-num-batched-tokens 128` is the setting for interactive load.
* **Ollama is still well ahead on this machine**: 215 against 139 tokens/s with 8 clients (this
  engine reaches 65 %), a first token in 0.07 s against 0.64 s at best, and even with one client
  it decodes 7 % faster (98.1 against 91.6 tokens/s).
* **The time is spent inside llama.cpp, not in Python or HTTP.** A `py-spy` profile of an
  8-client run (200 Hz, `--threads`, 3,892 samples) puts 98.6 % of the engine thread's samples
  inside `llama_decode`; the scheduler, sampling and logit copies add up to about 1.4 %, and the
  HTTP thread was awake for 1 % of the samples. The remaining gap is therefore in how the batch
  runs inside llama.cpp — this run used the prebuilt wheel, while the 2026-11-17 run below used a
  native source build — rather than in the step loop or the HTTP thread.

### Cloud container (2026-11-17)

* **Machine**: cloud container, Intel Xeon @ 2.10 GHz, 4 vCPU (AVX-512, AMX), 15 GB RAM, no GPU.
* **Model**: `qwen2.5-0.5b-instruct-q4_k_m.gguf` from `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
  (sha256 `74a4da8c9fdbcd15…`), the **same file** for both servers.
* **llm-inference-server 0.1.0**: llama.cpp backend through llama-cpp-python 0.3.37 (built
  from source, `GGML_NATIVE=ON`), `--threads 4 --max-num-seqs 8 --kv-tokens 8192
  --max-num-batched-tokens 512`, Python 3.13.
* **Ollama 0.40.16**: official Docker image (CPU), the GGUF imported with a `Modelfile`
  (`num_thread 4`, `num_ctx 2048`), `OLLAMA_NUM_PARALLEL=8`.
* Both measured back to back on 2026-11-17, nothing else running.

| Server | Clients | TTFT p50 (s) | TTFT p95 (s) | Decode per request p50 (tok/s) | Aggregate throughput (tok/s) |
| --- | --- | --- | --- | --- | --- |
| llm-inference-server | 1 | 0.171 | 0.207 | 48.1 | 45.4 |
| Ollama | 1 | 0.096 | 0.132 | 47.9 | 46.3 |
| llm-inference-server | 4 | 0.522 | 0.813 | 24.6 | 90.3 |
| Ollama | 4 | 0.348 | 0.498 | 30.0 | 110.5 |
| llm-inference-server | 8 | 1.083 | 1.143 | 16.9 | 119.2 |
| Ollama | 8 | 0.156 | 0.760 | 18.8 | 144.2 |

What the numbers say:

* **One client: parity in decoding** (48.1 vs 47.9 tokens/s per request) — both run the same
  llama.cpp kernels; the per-step Python overhead of this engine is small. Time to first token
  is ~75 ms higher (HTTP, template and tokenizer in Python).
* **Under load, continuous batching works** — aggregate throughput goes from 45 to 119 tokens/s
  (2.6×) with 8 clients — **but Ollama scales further**: 144 tokens/s, so this engine reaches
  83 % of Ollama at 8 clients and its TTFT under load is worse (the median request waits ~1 s
  for the first token, against 0.16 s).
* A finding along the way: llama.cpp built with OpenMP (llama-cpp-python's default) **halves
  its speed when `llama_decode` is called from a thread other than the main one** (41 vs 20
  tokens/s measured here: a second OpenMP team competes for the same cores). The server
  therefore runs the engine loop on the main thread and HTTP on a second thread; before that
  change this table read 23.7 / 53.0 / 73.1 tokens/s.

Reproduce:

```bash
python -m llm_inference_server --backend llamacpp --model qwen2.5-0.5b-instruct-q4_k_m.gguf --served-name qwen2.5-0.5b --threads 4 --max-num-batched-tokens 512 &
py-spy record -o profile.txt -f raw -r 200 --threads --pid <server pid>   # optional profile
python bench/benchmark.py --url http://127.0.0.1:8000/v1 --model qwen2.5-0.5b --concurrency 1,4,8 --requests 16
docker run -d -e OLLAMA_NUM_PARALLEL=8 -v $PWD/models:/models -p 11434:11434 ollama/ollama
docker exec <container> ollama create qwen05 -f /models/Modelfile   # bench/Modelfile
python bench/benchmark.py --url http://127.0.0.1:11434/v1 --model qwen05 --concurrency 1,4,8 --requests 16
```

Raw results: `bench/results/*.json` (2026-11-17) and `bench/results/2026-11-18/*.json`.

**MLX** was exercised on MLX's Linux CPU build (0.32.4): correct, but that build is not meant
for speed (4-bit checkpoints take ~19 s per token even in `mlx-lm` itself), so it is not in the
table. Its real target is Apple silicon, where it has not been measured.

## Using it as a package

```bash
pip install "llm-inference-server[llamacpp] @ git+https://github.com/juanmmm21/llm-inference-server.git@v0.1.0"
pip install "llm-inference-server[mlx] @ ..."     # Apple silicon (or MLX's CPU build)
pip install "llm-inference-server[torch] @ ..."   # reference backend (transformers weights)
```

```python
from llm_inference_server import LLMEngine, SamplingParams
from llm_inference_server.backends import load_backend

backend = load_backend("llamacpp", model_path="model.gguf", n_ctx=8192, max_num_seqs=8)
engine = LLMEngine(backend, max_num_seqs=8)
prompts = [backend.tokenize(backend.apply_chat_template([{"role": "user", "content": q}])) for q in ("Hi", "2+2?")]
for seq in engine.generate(prompts, SamplingParams(max_tokens=64)):
    print(backend.decode(seq.output_ids))
```

This component imports no other ai-core component; the `aicore` SDK of
[ai-core-infra-hub](https://github.com/juanmmm21/ai-core-infra-hub) talks to it through its
OpenAI-compatible API (`aicore.ops.InferenceClient`).

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # scheduler, KV cache, engine, API (toy backend)
AICORE_SLOW_TESTS=1 pytest               # + torch vs Hugging Face, MLX vs mlx-lm (downloads models)
```

Version `0.1.0` replaces the previous design (a time-window "dynamic batching" queue in front of
`transformers.generate`, with a simulated fallback engine): the old `/generate` endpoints are
gone in favour of the OpenAI-compatible API.
