"""
qwen-think backends: Backend auto-detection and registry.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from ..types import Backend
from .base import BaseBackend
from .dashscope import DashScopeBackend
from .llamacpp import LlamaCppBackend
from .vllm import OpenAIBackend, SGLangBackend, VLLMBackend

logger = logging.getLogger("qwen-think.backends")

_BACKEND_REGISTRY: Dict[Backend, Type[BaseBackend]] = {
    Backend.VLLM: VLLMBackend,
    Backend.SGLANG: SGLangBackend,
    Backend.DASHSCOPE: DashScopeBackend,
    Backend.LLAMACPP: LlamaCppBackend,
    Backend.OPENAI: OpenAIBackend,
}


def get_backend(backend: Backend, **kwargs) -> BaseBackend:
    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: {list(_BACKEND_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def detect_backend(base_url: str) -> BaseBackend:
    """Auto-detect the backend from a base URL.

    Raises ValueError if no backend matches the URL -- callers should
    fall back to a default or ask the user to specify explicitly.
    """
    candidates: List[tuple[float, BaseBackend]] = []

    for backend_type, cls in _BACKEND_REGISTRY.items():
        instance = cls()
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
