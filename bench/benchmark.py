"""Throughput and latency benchmark for any OpenAI-compatible server.

    python bench/benchmark.py --url http://127.0.0.1:8000/v1 --model qwen2.5-0.5b --concurrency 1,4,8 --requests 16

Every request streams (``stream: true``) a chat completion with a fixed prompt from
``PROMPTS``, ``temperature 0`` and ``max_tokens``. Per request it records the time to first
token (first non-empty content delta), the end-to-end latency and the number of output tokens
(the server's ``usage`` when it reports it, otherwise the number of content chunks). Per
concurrency level it reports:

* **TTFT** p50 / p95 (s);
* **decode speed per request** (output tokens - 1) / (latency - TTFT), p50, in tokens/s;
* **aggregate throughput**: all output tokens / wall-clock time of the level, in tokens/s.

Results are printed as a Markdown table and optionally written as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from typing import Any, Dict, List

import httpx

PROMPTS = [
    "Explain how a hash map works and when its lookups stop being constant time.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the water cycle step by step for a ten-year-old.",
    "List the pros and cons of remote work for a small software team.",
    "Explain the difference between TCP and UDP with two examples of each.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "Give a recipe for a vegetarian lasagna with the steps in order.",
    "Explain what gradient descent is and why the learning rate matters.",
]


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    async with client.stream("POST", "/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if data.get("usage"):
                usage_tokens = data["usage"].get("completion_tokens")
            for choice in data.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - start
    latency = time.perf_counter() - start
    tokens = usage_tokens if usage_tokens is not None else chunks
    return {"ttft": ttft if ttft is not None else latency, "latency": latency, "tokens": tokens}


async def level(url: str, model: str, concurrency: int, requests: int, max_tokens: int) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=600) as client:

        async def guarded(i: int) -> Dict[str, Any]:
            async with sem:
                return await one(client, model, PROMPTS[i % len(PROMPTS)], max_tokens)

        await one(client, model, "Say hi.", 4)  # warm-up (model load, graph build)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(requests)))
        wall = time.perf_counter() - t0
    ttfts = [r["ttft"] for r in results]
    decode = [(r["tokens"] - 1) / (r["latency"] - r["ttft"]) for r in results if r["tokens"] > 1 and r["latency"] > r["ttft"]]
    total = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "output_tokens": total,
        "wall_s": wall,
        "ttft_p50": pct(ttfts, 0.5),
        "ttft_p95": pct(ttfts, 0.95),
        "decode_tps_p50": pct(decode, 0.5),
        "throughput_tps": total / wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        rows.append(asyncio.run(level(args.url, args.model, c, max(args.requests, c), args.max_tokens)))
    label = args.label or args.url
    print(f"\n{label} · {platform.machine()} · Python {platform.python_version()}")
    print("| Server | Concurrency | TTFT p50 (s) | TTFT p95 (s) | Decode/request p50 (tok/s) | Throughput (tok/s) | Output tokens |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {label} | {r['concurrency']} | {r['ttft_p50']:.3f} | {r['ttft_p95']:.3f} | "
            f"{r['decode_tps_p50']:.1f} | {r['throughput_tps']:.1f} | {r['output_tokens']} |"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"label": label, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
