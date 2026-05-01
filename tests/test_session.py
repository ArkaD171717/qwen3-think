from unittest.mock import MagicMock

from qwen_think.session import ThinkingSession
from qwen_think.types import Backend, BudgetAction, Complexity, ThinkingMode


def _mock_client(content="response", thinking="reasoning"):
    client = MagicMock()
    client.base_url = "http://localhost:8000/v1"
    choice = MagicMock()
    choice.message.content = content
    choice.message.reasoning_content = thinking
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


class TestThinkingSession:
    def test_init_explicit_backend(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        assert s.backend == Backend.VLLM

    def test_init_openai_backend_reports_openai(self):
        s = ThinkingSession(_mock_client(), backend="openai")
        assert s.backend == Backend.OPENAI

    def test_init_auto_detect(self):
        s = ThinkingSession(_mock_client())
        assert s.backend == Backend.VLLM

    def test_init_unknown_url_falls_back(self):
        client = MagicMock()
        client.base_url = "http://my-custom-server:9999/api"
        choice = MagicMock()
        choice.message.content = "ok"
        choice.message.reasoning_content = None
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        s = ThinkingSession(client)
        assert s.backend == Backend.VLLM

    def test_chat_stores_messages(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert len(s.messages) == 2

    def test_chat_preserves_thinking(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert s.messages[1].thinking_content == "reasoning"

    def test_extra_body_has_nested_format(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat(
            "Refactor this module to use dependency injection", mode=ThinkingMode.THINK
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        eb = kwargs["extra_body"]
        assert "chat_template_kwargs" in eb
        assert eb["chat_template_kwargs"]["enable_thinking"] is True

    def test_non_standard_params_in_extra_body(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("implement something")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "top_k" not in kwargs
        assert "min_p" not in kwargs
        assert "repetition_penalty" not in kwargs
        assert kwargs["extra_body"]["top_k"] == 20

    def test_standard_params_top_level(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("implement something")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" in kwargs
        assert "top_p" in kwargs
        assert "presence_penalty" in kwargs

    def test_budget_refuse_raises(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm", budget=200_000, min_context=128_000)
        for _ in range(50):
            s.add_message("user", "x" * 5000)
        try:
            s.chat("one more")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "budget" in str(e).lower()

    def test_clear_history(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert len(s.messages) > 0
        s.clear_history(keep_system=False)
        assert len(s.messages) == 0

    def test_clear_history_keeps_system(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.add_message("system", "you are helpful")
        s.chat("hello")
        s.clear_history(keep_system=True)
        assert len(s.messages) == 1
        assert s.messages[0].role == "system"

    def test_set_thinking_mode(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.set_thinking_mode(ThinkingMode.NO_THINK)
        assert s.thinking_mode == ThinkingMode.NO_THINK

    def test_trim_history(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        for i in range(20):
            s.add_message("user", f"message {i}" * 500)
        original_count = len(s.messages)
        status = s.trim_history(keep_recent=4)
        assert len(s.messages) == original_count
        assert status.action in (BudgetAction.OK, BudgetAction.WARN)

    def test_stream_does_not_store_response(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("hello", stream=True)
        # Only the user message should be stored, not the assistant response
        assert len(s.messages) == 1
        assert s.messages[0].role == "user"

    def test_repr(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        r = repr(s)
        assert "vllm" in r
        assert "think" in r

    def test_add_message_counts_thinking_tokens(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        msg = s.add_message("assistant", "hello", thinking_content="long reasoning")
        assert msg.token_count > s.budget_manager.count_tokens("hello")

    def test_system_message_in_api_params(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("hi", system="be concise", mode=ThinkingMode.NO_THINK)
        kwargs = client.chat.completions.create.call_args.kwargs
        msgs = kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "be concise"}

    # -----------------------------------------------------------------------
    # Init: no base_url on client → defaults to vLLM (lines 82-83)
    # -----------------------------------------------------------------------

    def test_init_no_base_url_defaults_vllm(self):
        client = MagicMock()
        client.base_url = None  # falsy → triggers the else branch
        s = ThinkingSession(client)
        assert s.backend == Backend.VLLM

    # -----------------------------------------------------------------------
    # budget_status property (line 110)
    # -----------------------------------------------------------------------

    def test_budget_status_property(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        status = s.budget_status
        assert status.action == BudgetAction.OK
        assert status.total_tokens == 200_000

    # -----------------------------------------------------------------------
    # COMPRESS auto-trim during chat (lines 128-129)
    # -----------------------------------------------------------------------

    def test_chat_triggers_compress_and_trims(self):
        """Filling the session to the COMPRESS zone causes auto-trim before the call."""
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm", budget=200_000, min_context=128_000)
        # compress_threshold = 128K * 1.15 = 147,200
        # 110,000 chars * 0.5 tok/char = 55,000 tokens used
        # available = 145,000 < 147,200 → COMPRESS
        s.add_message("user", "x" * 110_000)
        s.chat("hello")  # Should trigger COMPRESS trim then succeed
        # After trim + new exchange: history contains at least the new user msg
        assert any(m.content == "hello" for m in s.messages)

    # -----------------------------------------------------------------------
    # Complexity override inside auto-route path (line 147)
    # -----------------------------------------------------------------------

    def test_chat_complexity_override_in_auto_route(self):
        """auto_route=True with explicit complexity= overrides the detected value."""
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm", auto_route=True)
        # "What is 2+2?" would normally be SIMPLE; override to COMPLEX
        s.chat("What is 2+2?", complexity=Complexity.COMPLEX)
        assert len(s.messages) == 2

    # -----------------------------------------------------------------------
    # auto_route=False path (line 157)
    # -----------------------------------------------------------------------

    def test_chat_auto_route_false(self):
        """When auto_route=False and no explicit mode, the session's current mode is used."""
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm", auto_route=False)
        s.chat("implement something")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "extra_body" in kwargs
        assert len(s.messages) == 2

    # -----------------------------------------------------------------------
    # Payload warnings are logged (line 177)
    # -----------------------------------------------------------------------

    def test_chat_payload_warning_does_not_raise(self):
        """LlamaCpp in NO_THINK mode always emits warnings; chat should still succeed."""
        client = _mock_client()
        # Use llamacpp backend — NO_THINK always emits a known-issue warning
        s = ThinkingSession(client, backend="llamacpp")
        s.chat("hello", mode=ThinkingMode.NO_THINK)
        assert any(m.role == "user" for m in s.messages)

    # -----------------------------------------------------------------------
    # set_backend (lines 210-213)
    # -----------------------------------------------------------------------

    def test_set_backend_string(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.set_backend("sglang")
        assert s.backend == Backend.SGLANG

    def test_set_backend_enum(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.set_backend(Backend.DASHSCOPE)
        assert s.backend == Backend.DASHSCOPE

    # -----------------------------------------------------------------------
    # get_openai_messages with include_thinking=True (line 249)
    # -----------------------------------------------------------------------

    def test_get_openai_messages_with_thinking(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        messages = s.get_openai_messages(include_thinking=True)
        # The assistant message should include reasoning_content
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") == "reasoning"

    # -----------------------------------------------------------------------
    # _store_response: empty choices list (line 294)
    # -----------------------------------------------------------------------

    def test_store_response_empty_choices(self):
        """When the API returns no choices, no assistant message is stored."""
        client = MagicMock()
        client.base_url = "http://localhost:8000/v1"
        resp = MagicMock()
        resp.choices = []
        client.chat.completions.create.return_value = resp

        s = ThinkingSession(client, backend="vllm")
        s.chat("hello")
        assert len(s.messages) == 1
        assert s.messages[0].role == "user"

    # -----------------------------------------------------------------------
    # __len__ (line 327)
    # -----------------------------------------------------------------------

    def test_len(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        assert len(s) == 0
        s.add_message("user", "hello")
        assert len(s) == 1
        s.add_message("assistant", "hi")
        assert len(s) == 2
