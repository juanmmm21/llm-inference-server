"""llama.cpp backend (GGUF models) through llama-cpp-python's low-level C API.

One ``llama_context`` serves every sequence: each running sequence gets a llama.cpp *sequence
id* (``0..max_num_seqs-1``) and every step puts the tokens of all of them — prefill chunks and
single decode tokens alike — into **one** ``llama_batch``, so the matrix multiplications run once
for the whole batch. That is continuous batching on llama.cpp's kernels.

KV memory: llama.cpp keeps its own cache (one unified buffer of ``n_ctx`` cells shared by all
sequence ids); it is not addressed through our block tables. The engine's
:class:`~llm_inference_server.kv_cache.PagedKVCacheManager` is sized to the same ``n_ctx`` and
used for admission and preemption, so the scheduler never admits more tokens than llama.cpp can
hold, and a finished or preempted sequence's cells are removed with ``llama_memory_seq_rm``.
Prefix caching is therefore off for this backend.
"""

from __future__ import annotations

import ctypes
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .base import Backend, BatchItem


class LlamaCppBackend(Backend):
    name = "llamacpp"
    supports_prefix_cache = False

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        max_num_seqs: int = 8,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        model_name: Optional[str] = None,
        flash_attn: bool = False,
    ) -> None:
        import llama_cpp as lc

        self._lc = lc
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_name = model_name or os.path.splitext(os.path.basename(model_path))[0]
        lc.llama_backend_init()
        mparams = lc.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        self._model = lc.llama_model_load_from_file(model_path.encode(), mparams)
        if not self._model:
            raise RuntimeError(f"llama.cpp could not load {model_path}")
        threads = n_threads or os.cpu_count() or 4
        cparams = lc.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = n_batch
        cparams.n_seq_max = max_num_seqs
        cparams.n_threads = threads
        cparams.n_threads_batch = threads
        cparams.kv_unified = True
        cparams.no_perf = True
        if hasattr(cparams, "flash_attn_type"):
            cparams.flash_attn_type = 1 if flash_attn else 0
        self._ctx = lc.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            raise RuntimeError("llama.cpp could not create a context")
        self._mem = lc.llama_get_memory(self._ctx)
        self._vocab = lc.llama_model_get_vocab(self._model)
        self.n_vocab = lc.llama_vocab_n_tokens(self._vocab)
        self.n_ctx = lc.llama_n_ctx(self._ctx)
        self.n_batch = n_batch
        self.max_num_seqs = max_num_seqs
        self.threads = threads
        self._batch = lc.llama_batch_init(n_batch, 0, 1)
        self._free_ids: List[int] = list(range(max_num_seqs - 1, -1, -1))
        self._seq_ids: Dict[str, int] = {}
        self._pieces: Dict[int, bytes] = {}
        self._eos = {t for t in range(self.n_vocab) if lc.llama_vocab_is_eog(self._vocab, t)}
        self._train_ctx = lc.llama_model_n_ctx_train(self._model)
        self._template = self._meta("tokenizer.chat_template")
        self._compiled = None
        self._bos = self._piece(lc.llama_vocab_bos(self._vocab)).decode("utf-8", "replace") if lc.llama_vocab_bos(self._vocab) >= 0 else ""

    # ------------------------------------------------------------------ model facts

    @property
    def eos_token_ids(self) -> Set[int]:
        return self._eos

    @property
    def max_model_len(self) -> int:
        return min(self.n_ctx, self._train_ctx)

    def kv_capacity_tokens(self) -> int:
        return self.n_ctx

    def _meta(self, key: str) -> str:
        buf = ctypes.create_string_buffer(1 << 16)
        n = self._lc.llama_model_meta_val_str(self._model, key.encode(), buf, len(buf))
        return buf.value.decode("utf-8", "replace") if n > 0 else ""

    def chat_template(self) -> str:
        return self._template or super().chat_template()

    # ------------------------------------------------------------------ text

    def tokenize(self, text: str, add_special: bool = False) -> List[int]:
        data = text.encode("utf-8")
        cap = len(data) + 16
        buf = (self._lc.llama_token * cap)()
        n = self._lc.llama_tokenize(self._vocab, data, len(data), buf, cap, add_special, True)
        if n < 0:
            cap = -n
            buf = (self._lc.llama_token * cap)()
            n = self._lc.llama_tokenize(self._vocab, data, len(data), buf, cap, add_special, True)
        return list(buf[:n])

    def _piece(self, token: int) -> bytes:
        piece = self._pieces.get(token)
        if piece is None:
            buf = ctypes.create_string_buffer(64)
            n = self._lc.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, False)
            if n < 0:
                buf = ctypes.create_string_buffer(-n)
                n = self._lc.llama_token_to_piece(self._vocab, token, buf, len(buf), 0, False)
            piece = buf.raw[:n]
            self._pieces[token] = piece
        return piece

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self._piece(t) for t in ids).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages):  # type: ignore[override]
        if self._compiled is None:  # compiling the template costs more than rendering it
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
            self._compiled = env.from_string(self.chat_template())
        return self._compiled.render(messages=list(messages), add_generation_prompt=True, bos_token=self._bos, eos_token="")

    # ------------------------------------------------------------------ compute

    def _seq(self, seq_id: str, start_pos: int) -> int:
        idx = self._seq_ids.get(seq_id)
        if idx is None:
            if not self._free_ids:
                raise RuntimeError("more concurrent sequences than llama.cpp n_seq_max")
            idx = self._free_ids.pop()
            self._seq_ids[seq_id] = idx
            self._lc.llama_memory_seq_rm(self._mem, idx, -1, -1)
        elif start_pos == 0:
            self._lc.llama_memory_seq_rm(self._mem, idx, -1, -1)
        return idx

    def forward(self, items: Sequence[BatchItem]) -> Dict[str, np.ndarray]:
        total = sum(len(i.tokens) for i in items)
        out: Dict[str, np.ndarray] = {}
        # Steps never exceed n_batch (the scheduler's token budget is n_batch); split defensively.
        if total > self.n_batch:
            for item in items:
                for start in range(0, len(item.tokens), self.n_batch):
                    part = item.tokens[start : start + self.n_batch]
                    last = start + self.n_batch >= len(item.tokens)
                    out.update(self.forward([BatchItem(item.seq_id, part, item.start_pos + start, item.want_logits and last)]))
            return out
        b = self._batch
        n = 0
        logit_rows: List[tuple] = []
        for item in items:
            idx = self._seq(item.seq_id, item.start_pos)
            for j, tok in enumerate(item.tokens):
                b.token[n] = tok
                b.pos[n] = item.start_pos + j
                b.n_seq_id[n] = 1
                b.seq_id[n][0] = idx
                want = item.want_logits and j == len(item.tokens) - 1
                b.logits[n] = 1 if want else 0
                if want:
                    logit_rows.append((item.seq_id, n))
                n += 1
        b.n_tokens = n
        rc = self._lc.llama_decode(self._ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
        for seq_id, row in logit_rows:
            ptr = self._lc.llama_get_logits_ith(self._ctx, row)
            out[seq_id] = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()
        return out

    def release(self, seq_id: str) -> None:
        idx = self._seq_ids.pop(seq_id, None)
        if idx is not None:
            self._lc.llama_memory_seq_rm(self._mem, idx, -1, -1)
            self._free_ids.append(idx)

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lc.llama_batch_free(self._batch)
            self._lc.llama_free(self._ctx)
            self._lc.llama_model_free(self._model)
            self._ctx = None
