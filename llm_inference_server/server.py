"""OpenAI-compatible HTTP API over the engine.

Endpoints: ``POST /v1/chat/completions`` and ``POST /v1/completions`` (both with ``stream``),
``GET /v1/models``, ``GET /health`` and ``GET /metrics`` (engine, scheduler and KV-cache
counters). Any OpenAI client works by pointing its base URL at ``http://host:port/v1``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import AsyncLLMEngine
from .scheduler import SamplingParams


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = ""

    def text(self) -> str:
        if isinstance(self.content, list):
            return "".join(part.get("text", "") for part in self.content if isinstance(part, dict))
        return self.content or ""


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Common(BaseModel):
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    max_completion_tokens: Optional[int] = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    seed: Optional[int] = None
    stop: Union[str, List[str], None] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    ignore_eos: bool = False

    def params(self, default_max: int) -> SamplingParams:
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop or [])
        return SamplingParams(
            max_tokens=self.max_completion_tokens or self.max_tokens or default_max,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=stops,
            ignore_eos=self.ignore_eos,
        )


class ChatCompletionRequest(_Common):
    messages: List[ChatMessage]


class CompletionRequest(_Common):
    prompt: str


def create_app(engine: AsyncLLMEngine, model_name: Optional[str] = None, default_max_tokens: int = 256) -> FastAPI:
    backend = engine.backend
    served = model_name or backend.model_name
    app = FastAPI(title="LLM Inference Server", description="Continuous batching, paged KV cache, OpenAI-compatible API.")
    app.state.engine = engine

    def check_model(name: Optional[str]) -> None:
        if name and name != served:
            raise HTTPException(status_code=404, detail=f"model {name!r} not served (this server serves {served!r})")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "backend": backend.name, "model": served}

    @app.get("/metrics")
    async def metrics() -> Dict[str, Any]:
        return engine.engine.metrics()

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": served, "object": "model", "created": 0, "owned_by": backend.name}]}

    async def run(prompt_ids: List[int], req: _Common, kind: str) -> Any:
        params = req.params(default_max_tokens)
        rid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        if req.stream:
            return StreamingResponse(stream(prompt_ids, params, req, kind, rid, created), media_type="text/event-stream")
        text, n, reason = [], 0, None
        try:
            async for ev in engine.generate(prompt_ids, params):
                text.append(ev.text_delta)
                n += 1
                reason = ev.finish_reason
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": n, "total_tokens": len(prompt_ids) + n}
        content = "".join(text)
        if kind == "chat":
            choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": reason}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": content, "finish_reason": reason, "logprobs": None}
            obj = "text_completion"
        return JSONResponse({"id": rid, "object": obj, "created": created, "model": served, "choices": [choice], "usage": usage})

    async def stream(prompt_ids: List[int], params: SamplingParams, req: _Common, kind: str, rid: str, created: int) -> AsyncIterator[str]:
        obj = "chat.completion.chunk" if kind == "chat" else "text_completion"

        def chunk(choice: Dict[str, Any], usage: Optional[Dict[str, int]] = None) -> str:
            body: Dict[str, Any] = {"id": rid, "object": obj, "created": created, "model": served, "choices": [choice] if choice else []}
            if usage is not None:
                body["usage"] = usage
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        if kind == "chat":
            yield chunk({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None})
        n = 0
        try:
            async for ev in engine.generate(prompt_ids, params):
                n += 1
                if ev.text_delta or ev.finished:
                    if kind == "chat":
                        delta = {"content": ev.text_delta} if ev.text_delta else {}
                        yield chunk({"index": 0, "delta": delta, "finish_reason": ev.finish_reason})
                    else:
                        yield chunk({"index": 0, "text": ev.text_delta, "finish_reason": ev.finish_reason})
        except ValueError as exc:
            yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
        if req.stream_options and req.stream_options.include_usage:
            yield chunk({}, {"prompt_tokens": len(prompt_ids), "completion_tokens": n, "total_tokens": len(prompt_ids) + n})
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest) -> Any:
        check_model(req.model)
        prompt = backend.apply_chat_template([{"role": m.role, "content": m.text()} for m in req.messages])
        return await run(backend.tokenize(prompt, add_special=False), req, "chat")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        check_model(req.model)
        return await run(backend.tokenize(req.prompt, add_special=True), req, "completion")

    return app
