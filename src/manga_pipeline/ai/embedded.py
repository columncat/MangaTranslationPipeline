"""Embedded translator: Gemma 4 E4B-it Q4_K_M via llama-cpp-python.

The user picks "embedded" in the side panel and the rest just works:
the GGUF auto-downloads on first use into ``models/`` (about 5 GB) and
subsequent launches reuse the on-disk file.

Internally this is just :class:`LlamaCppTranslator` with the
model_path filled in for the user. The download is triggered before
the llama_cpp ``Llama`` constructor runs, so the dialog that wraps
this backend can show a separate progress popup if it wants to.

Gemma 4 supports image / audio modalities through its full multimodal
checkpoint, but we only ever feed text via chat completion, so the
text-only GGUF path is all we need.
"""
from __future__ import annotations

from .base import BackendUnavailable, TranslatorConfig
from .llamacpp import LlamaCppTranslator


class EmbeddedTranslator(LlamaCppTranslator):
    def __init__(self, cfg: TranslatorConfig):
        # Lazy imports so importing the AI package never pulls in the
        # ML weights module (which loads ``requests`` at module scope).
        from ..ml.weights import (
            ensure_embedded_llm_weights,
            embedded_llm_weights_present,
        )

        if not embedded_llm_weights_present():
            # Don't kick off a multi-GB blocking download from inside
            # the pipeline thread — surface a clear error so the GUI's
            # _on_run_phase / _on_run_all checks can route to the
            # download dialog before retrying.
            raise BackendUnavailable("EMBEDDED_WEIGHTS_MISSING")

        # Replace whatever the user typed with the auto-managed path.
        # llama.cpp likes a few llama-cpp tuning knobs to be sane for
        # this model: 8K context fits the 128K window down to something
        # reasonable for VRAM, and chatml is wrong for Gemma — its
        # template lives in the GGUF metadata so we leave chat_format
        # auto-detected by llama.cpp.
        weights_path = ensure_embedded_llm_weights()
        cfg = TranslatorConfig(
            provider=cfg.provider,
            model="gemma-4-E4B-it-Q4_K_M",
            max_tokens=cfg.max_tokens or 4096,
            api_key=None,
            base_url=None,
            model_path=str(weights_path),
            n_ctx=cfg.n_ctx if cfg.n_ctx else 8192,
            n_gpu_layers=cfg.n_gpu_layers if cfg.n_gpu_layers else -1,
        )
        super().__init__(cfg)
