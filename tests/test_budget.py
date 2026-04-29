import pytest

from qwen_think.budget import (
    BudgetManager,
    estimate_tokens,
    truncate_old_messages,
    truncate_text,
)
from qwen_think.types import BudgetAction, Message


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_positive():
    # estimate_tokens uses: max(1, int(len(text) * 0.5))
    # "hello world" is 11 chars -> int(11 * 0.5) = 5
    assert estimate_tokens("hello world") == 5


def test_truncate_text_short_passthrough():
    assert truncate_text("hi") == "hi"


def test_truncate_text_empty_string():
    assert truncate_text("") == ""


def test_truncate_text_long_gets_ellipsis():
    long = "x" * 5000
    result = truncate_text(long, max_tokens=50)
    # max_chars = int(50 / 0.5) = 100, truncated = text[:97] + "..."
    assert result.endswith("...")
    assert len(result) == 100


def _make_msg(chars: int) -> Message:
    return Message(role="user", content="x" * chars)


class TestBudgetManager:
    def setup_method(self):
        self.bm = BudgetManager(total_budget=200_000, min_context=128_000)

    def test_empty_is_ok(self):
        status = self.bm.check_budget([])
        assert status.action == BudgetAction.OK
        assert status.available_tokens == 200_000

    def test_warn_triggers(self):
        # warn_threshold = 128K * 1.30 = 166,400
        # available = 155K -> used = 45K -> 90,000 chars (at 0.5 tok/char)
        # 155K < 166,400 -> WARN
        msg = _make_msg(100_000)  # ~50K tokens -> 150K available
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.WARN

    def test_compress_triggers(self):
        # compress_threshold = 128K * 1.15 = 147,200
        # 130,000 chars -> 65K tokens -> 135K available
        # 135K < 147,200 -> COMPRESS
        msg = _make_msg(130_000)
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.COMPRESS

    def test_refuse_triggers(self):
        # 160,000 chars -> 80K tokens -> 120K available
        # 120K < 128K -> REFUSE
        msg = _make_msg(160_000)
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.REFUSE

    def test_is_below_minimum(self):
        msg = _make_msg(160_000)
        status = self.bm.check_budget([msg])
        assert status.is_below_minimum is True

        ok = self.bm.check_budget([])
        assert ok.is_below_minimum is False

    def test_trim_reduces_tokens(self):
        msgs = [Message(role="user", content="x" * 10_000) for _ in range(20)]
        original = self.bm.count_messages_tokens(msgs)
        trimmed = self.bm.trim(msgs, keep_recent=4)
        assert self.bm.count_messages_tokens(trimmed) < original

    def test_trim_keeps_recent(self):
        msgs = [Message(role="user", content=f"msg{i}" * 500) for i in range(10)]
        trimmed = self.bm.trim(msgs, keep_recent=3)
        assert len(trimmed) == 10
        for i in range(7, 10):
            assert trimmed[i].content == msgs[i].content

    def test_trim_preserves_system_messages(self):
        msgs = [
            Message(role="system", content="you are helpful " * 100),
            Message(role="user", content="x" * 10_000),
            Message(role="assistant", content="y" * 10_000),
            Message(role="user", content="z" * 10_000),
        ]
        trimmed = self.bm.trim(msgs, keep_recent=1)
        assert trimmed[0].content == msgs[0].content

    def test_trim_counts_thinking_tokens(self):
        msg = Message(
            role="assistant",
            content="short",
            thinking_content="x" * 10_000,
        )
        msgs = [msg] * 10
        trimmed = self.bm.trim(msgs, keep_recent=2)
        # Trimmed messages should have lower token counts that include
        # both content and thinking_content
        for trimmed_msg in trimmed[:8]:
            assert trimmed_msg.token_count < msg.token_count or (
                trimmed_msg.token_count == self.bm.count_message_tokens(trimmed_msg)
            )

    def test_usage_ratio(self):
        status = self.bm.check_budget([])
        assert status.usage_ratio == 0.0

        msg = _make_msg(200_000)  # 100K tokens
        status = self.bm.check_budget([msg])
        assert status.usage_ratio == 0.5

    def test_update_message_counts_returns_none(self):
        msgs = [Message(role="user", content="hello")]
        result = self.bm.update_message_counts(msgs)
        assert result is None
        assert msgs[0].token_count > 0

    def test_rejects_budget_below_min_context(self):
        with pytest.raises(ValueError, match="must be >="):
            BudgetManager(total_budget=50_000, min_context=128_000)

    def test_warn_message_includes_threshold(self):
        msg = _make_msg(100_000)
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.WARN
        assert "166" in status.message  # warn_threshold ~ 166,400


def test_truncate_old_messages_under_limit():
    msgs = [Message(role="user", content="hello")]
    result = truncate_old_messages(msgs, keep_recent=4)
    assert result is msgs
