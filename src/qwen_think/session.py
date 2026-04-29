"""ThinkingSession: main entry point wrapping conversation lifecycle,
backend detection, thinking flag normalization, and budget tracking."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .backends import BaseBackend, VLLMBackend, detect_backend, get_backend
from .budget import BudgetManager
from .router import ComplexityRouter
from .sampling import SamplingManager
from .types import (
    Backend,
    BudgetAction,
    BudgetStatus,
    Complexity,
    Message,
    RouterDecision,
    ThinkingMode,
)

logger = logging.getLogger("qwen-think")

# Params the OpenAI client accepts as top-level kwargs to create().
# Everything else must go in extra_body for vLLM/SGLang.
_OPENAI_TOP_LEVEL = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "seed",
    "logprobs",
    "top_logprobs",
    "logit_bias",
}


class ThinkingSession:
    """Manages Qwen3.6 thinking state across a conversation session."""

    def __init__(
        self,
        client: Any,
        backend: Optional[Backend | str] = None,
        model: str = "Qwen/Qwen3.6-35B-A3B",
        budget: int = 200_000,
        min_context: int = 128_000,
        preserve_thinking: bool = True,
        auto_route: bool = True,
        force_thinking: bool = False,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.preserve_thinking = preserve_thinking
        self.auto_route = auto_route
        self.force_thinking = force_thinking

        if backend is not None:
            if isinstance(backend, str):
                backend = Backend(backend)
            self._backend_instance: BaseBackend = get_backend(backend)
        else:
            base_url = getattr(client, "base_url", None)
            if base_url:
                try:
                    self._backend_instance = detect_backend(str(base_url))
                    logger.info(
                        "Auto-detected backend: %s",
                        self._backend_instance.backend.value,
                    )
                except ValueError:
                    self._backend_instance = VLLMBackend()
                    logger.warning(
                        "Could not auto-detect backend for %s, defaulting to vLLM",
                        base_url,
                    )
            else:
                self._backend_instance = VLLMBackend()
                logger.warning("No base_url on client, defaulting to vLLM")

        self.budget_manager = BudgetManager(
            total_budget=budget,
            min_context=min_context,
            token_counter=token_counter,
        )
        self.sampling_manager = SamplingManager()
        self.router = ComplexityRouter(force_thinking=force_thinking)

        self._messages: List[Message] = []
        self._thinking_mode: ThinkingMode = ThinkingMode.THINK

    @property
    def backend(self) -> Backend:
        return self._backend_instance.backend

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    @property
    def thinking_mode(self) -> ThinkingMode:
        return self._thinking_mode

    @property
    def budget_status(self) -> BudgetStatus:
        return self.budget_manager.check_budget(self._messages)

    def chat(
        self,
        message: str,
        *,
        mode: Optional[ThinkingMode] = None,
        preserve: Optional[bool] = None,
        complexity: Optional[Complexity] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,  # Qwen3.6 default; matches vLLM/SGLang typical max_new_tokens
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        budget_status = self.budget_manager.check_budget(self._messages)
        if budget_status.action == BudgetAction.REFUSE:
            raise RuntimeError(f"Context budget exhausted: {budget_status.message}")
        if budget_status.action == BudgetAction.COMPRESS:
            logger.warning("Trimming conversation: %s", budget_status.message)
            self._messages = self.budget_manager.trim(self._messages)

        if mode is not None:
            self._thinking_mode = mode
            decision = RouterDecision(
                complexity=complexity or Complexity.MODERATE,
                mode=mode,
                preserve_thinking=preserve
                if preserve is not None
                else self.preserve_thinking,
                sampling=self.sampling_manager.get_config(mode),
                confidence=1.0,
                reasoning="Explicitly set by user",
            )
        elif self.auto_route:
            context_strs = [m.content for m in self._messages[-6:]]
            decision = self.router.route(message, context=context_strs)
            if complexity is not None:
                decision = RouterDecision(
                    complexity=complexity,
                    mode=decision.mode,
                    preserve_thinking=decision.preserve_thinking,
                    sampling=decision.sampling,
                    confidence=decision.confidence,
                    reasoning=f"Overridden complexity to {complexity.value}",
                )
            self._thinking_mode = decision.mode
        else:
            decision = RouterDecision(
                complexity=Complexity.MODERATE,
                mode=self._thinking_mode,
                preserve_thinking=preserve
                if preserve is not None
                else self.preserve_thinking,
                sampling=self.sampling_manager.get_config(self._thinking_mode),
                confidence=1.0,
                reasoning="Using current mode",
            )

        preserve_val = preserve if preserve is not None else decision.preserve_thinking

        payload = self._backend_instance.build_payload(
            mode=decision.mode,
            preserve_thinking=preserve_val,
            sampling=decision.sampling.to_dict(),
        )

        for warning in payload.warnings:
            logger.warning(warning)

        user_msg = Message(
            role="user",
            content=message,
            token_count=self.budget_manager.count_tokens(message),
        )
        self._messages.append(user_msg)

        api_params = self._build_api_params(
            payload=payload,
            system=system,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        response = self.client.chat.completions.create(**api_params)

        if stream:
            logger.debug(
                "Streaming response -- call add_message() after consuming "
                "the stream to keep history in sync."
            )
        else:
            self._store_response(response, decision)

        return response

    def set_thinking_mode(self, mode: ThinkingMode) -> None:
        self._thinking_mode = mode

    def set_backend(self, backend: Backend | str) -> None:
        if isinstance(backend, str):
            backend = Backend(backend)
        self._backend_instance = get_backend(backend)
        logger.info("Switched backend to: %s", backend.value)

    def add_message(
        self,
        role: str,
        content: str,
        thinking_content: Optional[str] = None,
    ) -> Message:
        msg = Message(
            role=role,
            content=content,
            thinking_content=thinking_content,
            token_count=self.budget_manager.count_tokens(content)
            + (
                self.budget_manager.count_tokens(thinking_content)
                if thinking_content
                else 0
            ),
        )
        self._messages.append(msg)
        return msg

    def clear_history(self, keep_system: bool = True) -> None:
        if keep_system:
            self._messages = [m for m in self._messages if m.role == "system"]
        else:
            self._messages = []

    def trim_history(self, keep_recent: int = 4) -> BudgetStatus:
        self._messages = self.budget_manager.trim(self._messages, keep_recent)
        return self.budget_manager.check_budget(self._messages)

    def get_openai_messages(
        self,
        include_thinking: bool = False,
    ) -> List[Dict[str, Any]]:
        return [
            m.to_openai_dict(include_thinking=include_thinking) for m in self._messages
        ]

    def _build_api_params(
        self,
        payload: Any,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_messages: List[Dict[str, Any]] = []

        if system:
            api_messages.append({"role": "system", "content": system})

        include_thinking = payload.preserve_thinking and payload.enable_thinking
        for msg in self._messages:
            api_messages.append(msg.to_openai_dict(include_thinking=include_thinking))

        top_level = {
            k: v for k, v in payload.sampling.items() if k in _OPENAI_TOP_LEVEL
        }
        extra_sampling = {
            k: v for k, v in payload.sampling.items() if k not in _OPENAI_TOP_LEVEL
        }

        extra_body = {**payload.extra_body, **extra_sampling}

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "extra_body": extra_body,
            **top_level,
        }

        params.update(kwargs)

        return params

    def _store_response(self, response: Any, decision: RouterDecision) -> None:
        if not response.choices:
            return

        choice = response.choices[0]
        message = choice.message

        content = getattr(message, "content", "") or ""
        thinking_content = getattr(message, "reasoning_content", None)

        assistant_msg = Message(
            role="assistant",
            content=content,
            thinking_content=thinking_content,
            token_count=self.budget_manager.count_tokens(content)
            + (
                self.budget_manager.count_tokens(thinking_content)
                if thinking_content
                else 0
            ),
        )
        self._messages.append(assistant_msg)

    def __repr__(self) -> str:
        return (
            f"ThinkingSession("
            f"backend={self.backend.value}, "
            f"mode={self._thinking_mode.value}, "
            f"messages={len(self._messages)}, "
            f"budget={self.budget_manager.total_budget:,}"
            f")"
        )

    def __len__(self) -> int:
        return len(self._messages)
