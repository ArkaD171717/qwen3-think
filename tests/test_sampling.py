from qwen_think.sampling import SamplingManager
from qwen_think.types import ThinkingMode


def test_get_params_thinking():
    sm = SamplingManager()
    p = sm.get_params(ThinkingMode.THINK)
    assert p["temperature"] == 0.6
    assert p["top_p"] == 0.95


def test_get_params_non_thinking():
    sm = SamplingManager()
    p = sm.get_params(ThinkingMode.NO_THINK)
    assert p["temperature"] == 0.7
    assert p["top_p"] == 0.80


def test_swap_params_changes_values():
    sm = SamplingManager()
    result = sm.swap_params(
        ThinkingMode.THINK,
        ThinkingMode.NO_THINK,
        {"temperature": 0.6, "top_p": 0.95, "custom": 42},
    )
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.80
    assert result["custom"] == 42


def test_swap_params_noop_same_mode():
    sm = SamplingManager()
    original = {"temperature": 0.6}
    result = sm.swap_params(ThinkingMode.THINK, ThinkingMode.THINK, original)
    assert result is original


def test_validate_detects_mismatch():
    sm = SamplingManager()
    result = sm.validate_params(ThinkingMode.THINK, {"temperature": 0.5, "top_p": 0.95})
    assert result["valid"] is False
    assert "temperature" in result["mismatches"]


def test_validate_respects_user_values():
    sm = SamplingManager()
    result = sm.validate_params(ThinkingMode.THINK, {"temperature": 0.5})
    # User set 0.5 -- merged dict should keep that, not override to 0.6
    assert result["merged"]["temperature"] == 0.5


def test_validate_fills_missing_from_defaults():
    sm = SamplingManager()
    result = sm.validate_params(ThinkingMode.THINK, {"temperature": 0.6})
    assert result["merged"]["top_p"] == 0.95
    assert result["merged"]["top_k"] == 20


def test_validate_passes_correct():
    sm = SamplingManager()
    params = sm.get_params(ThinkingMode.THINK)
    result = sm.validate_params(ThinkingMode.THINK, params)
    assert result["valid"] is True
    assert len(result["mismatches"]) == 0
