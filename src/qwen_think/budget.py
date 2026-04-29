"""Context token budget management with 128K minimum guard for Qwen3.6."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from .types import BudgetAction, BudgetStatus, Message

logger = logging.getLogger("qwen-think.budget")

DEFAULT_MIN_CONTEXT = 128_000
# Warn at 30% above min_context (166K), compress at 15% above (147K).
# These give ~2 turns of warning before forced trimming at 128K.
WARN_RATIO = 0.3
COMPRESS_RATIO = 0.15
# Midpoint between English (~0.25 tok/char) and CJK (~1.0 tok/char).
# Qwen's tokenizer skews toward CJK; this errs on the side of overestimating
# token count, which is the safe direction for a budget guard.
AVG_TOKENS_PER_CHAR = 0.5
# Truncated messages keep ~200 tokens (~400 chars) of context --
# enough for the model to recall the topic without wasting budget.
COMPRESSED_MESSAGE_MAX_TOKENS = 200


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) * AVG_TOKENS_PER_CHAR))


def truncate_text(text: str, max_tokens: int = COMPRESSED_MESSAGE_MAX_TOKENS) -> str:
    if not text:
        return ""
    max_chars = int(max_tokens / AVG_TOKENS_PER_CHAR)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def truncate_old_messages(
    messages: List[Message],
    keep_recent: int = 4,  # 2 user-assistant pairs to preserve conversation flow
    max_tokens_per_message: int = COMPRESSED_MESSAGE_MAX_TOKENS,
) -> List[Message]:
    if len(messages) <= keep_recent:
        return messages

    result: List[Message] = []

    for i, msg in enumerate(messages):
        if i >= len(messages) - keep_recent or msg.role == "system":
            result.append(msg)
        else:
            new_content = truncate_text(msg.content, max_tokens_per_message)
            new_thinking = (
                truncate_text(msg.thinking_content, max_tokens_per_message)
                if msg.thinking_content
                else None
            )
            result.append(
                Message(
                    role=msg.role,
                    content=new_content,
                    thinking_content=new_thinking,
                    token_count=estimate_tokens(new_content)
                    + (estimate_tokens(new_thinking) if new_thinking else 0),
                )
            )

    return result


class BudgetManager:
    """Tracks token usage and guards the 128K minimum context threshold."""

    def __init__(
        self,
        total_budget: int = 200_000,
        min_context: int = DEFAULT_MIN_CONTEXT,
        token_counter: Optional[Callable[[str], int]] = None,
        warn_ratio: float = WARN_RATIO,
        compress_ratio: float = COMPRESS_RATIO,
    ) -> None:
        if total_budget < min_context:
            raise ValueError(
                f"total_budget ({total_budget:,}) must be >= "
                f"min_context ({min_context:,})"
            )
        self.total_budget = total_budget
        self.min_context = min_context
        self.token_counter = token_counter or estimate_tokens
        self.warn_ratio = warn_ratio
        self.compress_ratio = compress_ratio

    def count_tokens(self, text: str) -> int:
        return self.token_counter(text)

    def count_message_tokens(self, message: Message) -> int:
        tokens = self.count_tokens(message.content)
        if message.thinking_content:
            tokens += self.count_tokens(message.thinking_content)
        return tokens

    def count_messages_tokens(self, messages: List[Message]) -> int:
        return sum(self.count_message_tokens(msg) for msg in messages)

    def check_budget(self, messages: List[Message]) -> BudgetStatus:
        used = self.count_messages_tokens(messages)
        available = self.total_budget - used

        warn_threshold = self.min_context * (1.0 + self.warn_ratio)
        compress_threshold = self.min_context * (1.0 + self.compress_ratio)

        if available < self.min_context:
            action = BudgetAction.REFUSE
            message = (
                f"CRITICAL: Available context ({available:,} tokens) is below "
                f"the minimum ({self.min_context:,} tokens) required to "
                f"preserve Qwen3.6 thinking capabilities. Reasoning quality "
                f"will be silently degraded. Reduce conversation history."
            )
        elif available < compress_threshold:
            action = BudgetAction.COMPRESS
            message = (
                f"Available context ({available:,} tokens) is approaching "
                f"the {self.min_context:,}-token minimum. "
                f"Trimming older messages recommended."
            )
        elif available < warn_threshold:
            action = BudgetAction.WARN
            message = (
                f"Available context ({available:,} tokens) is below the "
                f"{warn_threshold:,.0f}-token threshold. "
                f"Used {used:,} of {self.total_budget:,} total."
            )
        else:
            action = BudgetAction.OK
            message = f"Available: {available:,} of {self.total_budget:,} tokens."

        return BudgetStatus(
            total_tokens=self.total_budget,
            used_tokens=used,
            available_tokens=available,
            min_context=self.min_context,
            action=action,
            message=message,
        )

    def trim(
        self,
        messages: List[Message],
        keep_recent: int = 4,
    ) -> List[Message]:
        original_tokens = self.count_messages_tokens(messages)
        trimmed = truncate_old_messages(messages, keep_recent=keep_recent)
        new_tokens = self.count_messages_tokens(trimmed)
        freed = original_tokens - new_tokens

        logger.info(
            "Trimmed conversation: %d -> %d tokens (freed %d)",
            original_tokens,
            new_tokens,
            freed,
        )

        return trimmed

    def update_message_counts(self, messages: List[Message]) -> None:
        for msg in messages:
            msg.token_count = self.count_message_tokens(msg)
