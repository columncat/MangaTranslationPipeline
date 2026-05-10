"""llama.cpp backend (direct GGUF load via ``llama-cpp-python``).

The user picks a local ``.gguf`` file and we load it on construction.
Loading is heavy (multi-second) and pinning the model to a single
:class:`LlamaCppTranslator` instance lets follow-up batches reuse it.

Inference uses the OpenAI-style chat completion shim so the same JSON
output schema we use for the cloud backends works without modification.

Optional GPU acceleration: install ``llama-cpp-python`` with the
appropriate CUDA / Metal wheel; we pass ``n_gpu_layers`` from the
TranslatorConfig (-1 means "offload all layers to GPU if possible").
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import (
    BackendUnavailable,
    TranslatorConfig,
    build_system_prompt,
    parse_batch_response,
)


class LlamaCppTranslator:
    def __init__(self, cfg: TranslatorConfig):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise BackendUnavailable(
                "The 'llama-cpp-python' package is not installed. Run "
                "`pip install llama-cpp-python` (or use the prebuilt CUDA "
                "wheel index for GPU acceleration)."
            ) from e

        if not cfg.model_path:
            raise BackendUnavailable(
                "Pick a GGUF model file in the side panel first."
            )
        path = Path(cfg.model_path)
        if not path.exists():
            raise BackendUnavailable(f"Model file not found: {path}")

        # Verbose=False keeps the GUI's stdout from being flooded with
        # llama.cpp's per-token chatter on every batch.
        self._llama = Llama(
            model_path=str(path),
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
            chat_format="chatml",  # works for Qwen / many recent models
        )
        self._max_tokens = cfg.max_tokens

    def translate_batch(
        self,
        lines: list[str],
        glossary: str = "",
        style: str = "",
    ) -> list[str]:
        if not lines:
            return []
        out = self._batch_call(lines, glossary, style)
        for i, ko in enumerate(out):
            if not ko:
                out[i] = self._single_call(lines[i], glossary, style)
        return out

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        # llama-cpp-python exposes an OpenAI-compatible chat method that
        # also accepts a ``response_format`` arg; for models that don't
        # honour it, the JSON-extraction fallback in base.py copes.
        resp = self._llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        choice = resp["choices"][0]
        return choice["message"].get("content", "") or ""

    def _batch_call(
        self, lines: list[str], glossary: str, style: str
    ) -> list[str]:
        system = build_system_prompt(glossary, style)
        payload = [{"id": i, "ja": line} for i, line in enumerate(lines)]
        user = json.dumps(payload, ensure_ascii=False)
        text = self._chat(system, user, self._max_tokens)
        return parse_batch_response(text, len(lines))

    def _single_call(self, line: str, glossary: str, style: str) -> str:
        if not line.strip():
            return ""
        system = build_system_prompt(glossary, style)
        user = json.dumps([{"id": 0, "ja": line}], ensure_ascii=False)
        text = self._chat(system, user, 512)
        out = parse_batch_response(text, 1)
        return out[0] if out and out[0] else line
