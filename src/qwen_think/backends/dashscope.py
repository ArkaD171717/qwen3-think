from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..types import Backend, BackendPayload, ThinkingMode
from .base import BaseBackend

if TYPE_CHECKING:
    from ..sampling import SamplingManager


class DashScopeBackend(BaseBackend):
    backend = Backend.DASHSCOPE

    def __init__(self, sampling_manager: Optional["SamplingManager"] = None) -> None:
        super().__init__(sampling_manager)

    # URL patterns that suggest DashScope
    _DASHSCOPE_PATTERNS = [
        r"dashscope",
        r"aliyuncs\.com",
        r"modelstudio",
        r"aigc",
    ]

    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload:
        enable_thinking = mode == ThinkingMode.THINK

        extra_body: Dict[str, Any] = {
            "enable_thinking": enable_thinking,
        }

        if preserve_thinking:
            extra_body["preserve_thinking"] = True

        if "extra_body" in kwargs:
            for k, v in kwargs["extra_body"].items():
                extra_body[k] = v

        sampling_params = self._common_sampling(mode, sampling)

        warnings: list[str] = []

        if "extra_body" in kwargs:
            nested = kwargs["extra_body"].get("chat_template_kwargs")
            if nested and "enable_thinking" in nested:
                warnings.append(
                    "DashScope expects enable_thinking at the top level of "
                    "extra_body, NOT nested inside chat_template_kwargs. "
                    "The nested format is for vLLM/SGLang only. "
                    "This backend has applied the correction."
                )

        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if "/no_think" in content or "/think" in content:
                    warnings.append(
                        "Qwen3.6 does not support /think or /no_think prompt "
                        "prefixes. Use enable_thinking parameter instead."
                    )
                    break

        return BackendPayload(
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            extra_body=extra_body,
            sampling=sampling_params,
            warnings=warnings,
        )

    def detect(self, base_url: Optional[str] = None) -> float:
        if base_url is None:
            return 0.0

        url_lower = base_url.lower()
        for pattern in self._DASHSCOPE_PATTERNS:
            if re.search(pattern, url_lower):
                return 0.9

        return 0.0
