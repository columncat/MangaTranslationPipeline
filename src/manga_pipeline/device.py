from __future__ import annotations


def auto_device(min_free_gb: float = 1.5) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        free, _ = torch.cuda.mem_get_info()
        if free < min_free_gb * 1024**3:
            return "cpu"
    except Exception:
        pass
    return "cuda"


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
