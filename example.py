"""Continuous batching in action, without a model download (the deterministic toy backend).

    python example.py

Three requests of different lengths share the engine: watch the third one join while the long
one is still running, and the KV-cache blocks being reused for the shared prompt prefix. For a
real model: ``python -m llm_inference_server --backend llamacpp --model model.gguf``.
"""

from llm_inference_server import LLMEngine, SamplingParams
from llm_inference_server.backends.toy import ToyBackend

backend = ToyBackend(kv_tokens=256)
engine = LLMEngine(backend, block_size=4, max_num_seqs=2, max_num_batched_tokens=16)
system = list(b"system prompt shared by every request. ")
requests = [
    engine.add_request(system + list(b"a"), SamplingParams(max_tokens=3)),
    engine.add_request(system + list(b"j"), SamplingParams(max_tokens=10)),
    engine.add_request(system + list(b"p"), SamplingParams(max_tokens=4)),
]
step = 0
while engine.has_work():
    events = engine.step()
    step += 1
    running = [s.id for s in engine.scheduler.running]
    print(f"step {step:2d}: batch={[i.seq_id for i in backend.calls[-1]]} tokens={[e.text_delta for e in events]} running={running}")
for seq in requests:
    print(seq.id, repr(backend.decode(seq.output_ids)), seq.finish_reason, f"cached prefix={seq.cached_prefix}")
print(engine.metrics())
