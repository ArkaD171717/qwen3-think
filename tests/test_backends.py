import pytest
from unittest.mock import patch

from qwen_think.backends import (
    DashScopeBackend,
    LlamaCppBackend,
    OpenAIBackend,
    VLLMBackend,
    detect_backend,
    get_backend,
)
from qwen_think.backends.vllm import SGLangBackend
from qwen_think.types import Backend, ThinkingMode


class TestVLLMBackend:
    def setup_method(self):
        self.b = VLLMBackend()

    def test_think_mode_payload(self):
        p = self.b.build_payload(ThinkingMode.THINK)
        assert p.extra_body["chat_template_kwargs"]["enable_thinking"] is True

    def test_no_think_explicitly_sets_false(self):
        p = self.b.build_payload(ThinkingMode.NO_THINK, preserve_thinking=False)
        ctk = p.extra_body["chat_template_kwargs"]
        assert "enable_thinking" in ctk
        assert ctk["enable_thinking"] is False

    def test_preserve_thinking_in_payload(self):
        p = self.b.build_payload(ThinkingMode.THINK, preserve_thinking=True)
        assert p.extra_body["chat_template_kwargs"]["preserve_thinking"] is True

    def test_no_think_soft_switch_warning(self):
        p = self.b.build_payload(
            ThinkingMode.THINK,
            messages=[{"role": "user", "content": "/no_think answer"}],
        )
        assert any("/no_think" in w for w in p.warnings)

    def test_detect_port_8000(self):
        assert self.b.detect("http://localhost:8000/v1") > 0

    def test_detect_v1_low_confidence(self):
        score = self.b.detect("http://some-server:9000/v1")
        assert 0 < score < 0.5


class TestOpenAIBackend:
    def test_reports_openai_backend(self):
        b = OpenAIBackend()
        assert b.backend == Backend.OPENAI

    def test_same_payload_as_vllm(self):
        b = OpenAIBackend()
        p = b.build_payload(ThinkingMode.THINK)
        assert "chat_template_kwargs" in p.extra_body

    def test_never_autodetected(self):
        b = OpenAIBackend()
        assert b.detect("http://api.openai.com/v1") == 0.0


class TestDashScopeBackend:
    def setup_method(self):
        self.b = DashScopeBackend()

    def test_top_level_format(self):
        p = self.b.build_payload(ThinkingMode.NO_THINK)
        assert "enable_thinking" in p.extra_body
        assert "chat_template_kwargs" not in p.extra_body
        assert p.extra_body["enable_thinking"] is False

    def test_warns_on_nested_format(self):
        p = self.b.build_payload(
            ThinkingMode.THINK,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        assert len(p.warnings) > 0

    def test_detect_aliyuncs(self):
        assert self.b.detect("https://dashscope.aliyuncs.com/v1") > 0


class TestLlamaCppBackend:
    def test_mismatch_warning(self):
        b = LlamaCppBackend(server_enable_thinking=True)
        p = b.build_payload(ThinkingMode.NO_THINK)
        assert len(p.warnings) >= 1
        assert p.enable_thinking is True

    def test_startup_command_no_think(self):
        cmd = LlamaCppBackend.get_startup_command(enable_thinking=False)
        assert "--reasoning-budget 0" in cmd
        assert '"enable_thinking": false' in cmd

    def test_startup_command_think(self):
        cmd = LlamaCppBackend.get_startup_command(enable_thinking=True)
        assert "--reasoning-budget 0" not in cmd

    def test_does_not_match_generic_8080(self):
        b = LlamaCppBackend()
        assert b.detect("http://localhost:8080/v1") == 0.0

    def test_matches_llama_in_url(self):
        b = LlamaCppBackend()
        assert b.detect("http://llama-server:8080/v1") > 0


class TestSGLangBackend:
    def test_same_format_as_vllm(self):
        b = SGLangBackend()
        p = b.build_payload(ThinkingMode.NO_THINK)
        assert p.extra_body["chat_template_kwargs"]["enable_thinking"] is False

    def test_reports_sglang_backend(self):
        b = SGLangBackend()
        assert b.backend == Backend.SGLANG


class TestAutoDetection:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://localhost:8000/v1", "vllm"),
            ("http://localhost:30000/v1", "sglang"),
            ("https://dashscope.aliyuncs.com/v1", "dashscope"),
            ("http://llama-server:8080/v1", "llamacpp"),
        ],
    )
    def test_detect_backend(self, url, expected):
        b = detect_backend(url)
        assert b.backend.value == expected

    def test_unknown_url_raises(self):
        with pytest.raises(ValueError, match="Could not auto-detect"):
            detect_backend("http://totally-unknown:9999/api")


# ---------------------------------------------------------------------------
# get_backend: unknown backend path (registry miss)
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_unknown_backend_raises(self):
        """Clearing the registry exposes the defensive ValueError."""
        with patch.dict("qwen_think.backends._BACKEND_REGISTRY", {}, clear=True):
            with pytest.raises(ValueError, match="Unknown backend"):
                get_backend(Backend.VLLM)


# ---------------------------------------------------------------------------
# detect(None) returns 0.0 for all backends (None guard branch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_cls", [VLLMBackend, DashScopeBackend, LlamaCppBackend])
def test_detect_none_url_returns_zero(backend_cls):
    assert backend_cls().detect(None) == 0.0


# ---------------------------------------------------------------------------
# Additional VLLMBackend coverage
# ---------------------------------------------------------------------------

class TestVLLMBackendExtra:
    def setup_method(self):
        self.b = VLLMBackend()

    def test_extra_body_chat_template_kwargs_merged(self):
        """User-supplied chat_template_kwargs are merged into the payload."""
        p = self.b.build_payload(
            ThinkingMode.THINK,
            extra_body={
                "chat_template_kwargs": {"custom_key": "value"},
                "other_key": 42,
            },
        )
        ctk = p.extra_body["chat_template_kwargs"]
        assert ctk["custom_key"] == "value"
        assert p.extra_body["other_key"] == 42


# ---------------------------------------------------------------------------
# Additional DashScopeBackend coverage
# ---------------------------------------------------------------------------

class TestDashScopeBackendExtra:
    def setup_method(self):
        self.b = DashScopeBackend()

    def test_no_think_message_warning(self):
        """Message containing /no_think triggers a deprecation warning."""
        p = self.b.build_payload(
            ThinkingMode.THINK,
            messages=[{"role": "user", "content": "/no_think do this"}],
        )
        assert any("/no_think" in w for w in p.warnings)


# ---------------------------------------------------------------------------
# Additional LlamaCppBackend coverage
# ---------------------------------------------------------------------------

class TestLlamaCppBackendExtra:
    def test_server_thinking_none_defaults_true(self):
        """LlamaCppBackend() with no arg treats server as enable_thinking=True."""
        b = LlamaCppBackend()
        p = b.build_payload(ThinkingMode.THINK)
        assert p.enable_thinking is True

    def test_extra_body_merging(self):
        """User-supplied chat_template_kwargs and other keys are merged."""
        b = LlamaCppBackend()
        p = b.build_payload(
            ThinkingMode.THINK,
            extra_body={
                "chat_template_kwargs": {"custom": "val"},
                "other": 1,
            },
        )
        assert p.extra_body["chat_template_kwargs"]["custom"] == "val"
        assert p.extra_body["other"] == 1

    def test_no_think_message_warning(self):
        """Message containing /no_think triggers a deprecation warning."""
        b = LlamaCppBackend()
        p = b.build_payload(
            ThinkingMode.THINK,
            messages=[{"role": "user", "content": "/no_think something"}],
        )
        assert any("/no_think" in w for w in p.warnings)
