from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from ..types import Backend
from .base import BaseBackend
from .dashscope import DashScopeBackend
from .llamacpp import LlamaCppBackend
from .vllm import OpenAIBackend, SGLangBackend, VLLMBackend

logger = logging.getLogger("qwen-think.backends")

# Backend enum -> factory callable.
_BACKEND_REGISTRY: Dict[Backend, Callable[..., BaseBackend]] = {
    Backend.VLLM: VLLMBackend,
    Backend.SGLANG: SGLangBackend,
    Backend.DASHSCOPE: DashScopeBackend,
    Backend.LLAMACPP: LlamaCppBackend,
    Backend.OPENAI: OpenAIBackend,
}


def get_backend(
    backend: Backend,
    sampling_manager: Optional[object] = None,
    **kwargs,
) -> BaseBackend:
    factory = _BACKEND_REGISTRY.get(backend)
    if factory is None:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: {list(_BACKEND_REGISTRY.keys())}"
        )
    if sampling_manager is not None:
        kwargs["sampling_manager"] = sampling_manager
    return factory(**kwargs)


def detect_backend(
    base_url: str,
    sampling_manager: Optional[object] = None,
) -> BaseBackend:
    candidates: List[tuple[float, BaseBackend]] = []

    for _backend_type, factory in _BACKEND_REGISTRY.items():
        kwargs = {}
        if sampling_manager is not None:
            kwargs["sampling_manager"] = sampling_manager
        instance = factory(**kwargs)
        score = instance.detect(base_url)
        if score > 0:
            candidates.append((score, instance))

    if not candidates:
        raise ValueError(
            f"Could not auto-detect backend for URL: {base_url}. "
            f"Pass backend= explicitly."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "SGLangBackend",
    "OpenAIBackend",
    "DashScopeBackend",
    "LlamaCppBackend",
    "get_backend",
    "detect_backend",
]
