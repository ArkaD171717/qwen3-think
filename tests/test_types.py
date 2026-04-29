import pytest

from qwen_think.types import (
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    Backend,
    Complexity,
    Message,
    SamplingConfig,
    ThinkingMode,
)


def test_backend_values():
    assert Backend.VLLM.value == "vllm"
    assert Backend.SGLANG.value == "sglang"
    assert Backend.DASHSCOPE.value == "dashscope"
    assert Backend.LLAMACPP.value == "llamacpp"
    assert Backend.OPENAI.value == "openai"


def test_thinking_mode_values():
    assert ThinkingMode.THINK.value == "think"
    assert ThinkingMode.NO_THINK.value == "no_think"


def test_complexity_values():
    assert Complexity.SIMPLE.value == "simple"
    assert Complexity.MODERATE.value == "moderate"
    assert Complexity.COMPLEX.value == "complex"


def test_sampling_config_frozen():
    with pytest.raises(AttributeError):
        THINKING_SAMPLING.temperature = 999.0


def test_sampling_config_to_dict():
    d = THINKING_SAMPLING.to_dict()
    assert d["temperature"] == 1.0
    assert d["top_p"] == 0.95
    assert d["top_k"] == 20


def test_sampling_constants_differ():
    assert THINKING_SAMPLING.temperature != NON_THINKING_SAMPLING.temperature
    assert THINKING_SAMPLING.top_p != NON_THINKING_SAMPLING.top_p


def test_sampling_config_hashable():
    s = {THINKING_SAMPLING, NON_THINKING_SAMPLING}
    assert len(s) == 2


def test_custom_sampling_config():
    cfg = SamplingConfig(temperature=0.5, top_p=0.9)
    assert cfg.temperature == 0.5
    assert cfg.top_k == 20  # default


def test_message_openai_dict_plain():
    msg = Message(role="assistant", content="hello", thinking_content="reasoning")
    d = msg.to_openai_dict()
    assert d == {"role": "assistant", "content": "hello"}
    assert "reasoning_content" not in d


def test_message_openai_dict_with_thinking():
    msg = Message(role="assistant", content="hello", thinking_content="reasoning")
    d = msg.to_openai_dict(include_thinking=True)
    assert d["reasoning_content"] == "reasoning"


def test_message_no_thinking():
    msg = Message(role="user", content="hi")
    d = msg.to_openai_dict(include_thinking=True)
    assert "reasoning_content" not in d


# ---------------------------------------------------------------------------
# BackendPayload.has_warnings
# ---------------------------------------------------------------------------

def test_backend_payload_has_warnings_false():
    from qwen_think.types import BackendPayload

    p = BackendPayload()
    assert p.has_warnings is False


def test_backend_payload_has_warnings_true():
    from qwen_think.types import BackendPayload

    p = BackendPayload(warnings=["something went wrong"])
    assert p.has_warnings is True


# ---------------------------------------------------------------------------
# BudgetStatus.usage_ratio edge case: total_tokens == 0
# ---------------------------------------------------------------------------

def test_budget_status_usage_ratio_zero_total():
    from qwen_think.types import BudgetStatus

    status = BudgetStatus(total_tokens=0, used_tokens=0)
    assert status.usage_ratio == 0.0
