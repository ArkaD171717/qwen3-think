from unittest.mock import MagicMock

from qwen_think.router import ComplexityRouter, LLMClassifier, RuleBasedClassifier
from qwen_think.types import Complexity, ThinkingMode


class TestRuleBasedClassifier:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_simple_queries(self):
        for q in ["yes", "What is Python?", "translate hello"]:
            c = self.clf.classify(q)
            assert c == Complexity.SIMPLE, f"{q} -> {c}"

    def test_moderate_queries(self):
        for q in ["How do I set up a virtual environment for this project?"]:
            c = self.clf.classify(q)
            assert c == Complexity.MODERATE, f"{q} -> {c}"

    def test_complex_queries(self):
        for q in [
            "Refactor this module to use dependency injection",
            "Debug and rewrite the authentication middleware that is leaking sessions step by step",
        ]:
            c = self.clf.classify(q)
            assert c == Complexity.COMPLEX, f"{q} -> {c}"

    def test_code_boosts_score(self):
        q = "```python\ndef foo():\n    return bar\n```\nfix this function"
        c = self.clf.classify(q)
        assert c == Complexity.COMPLEX

    def test_word_count_boost(self):
        short = "hello"
        long_q = "word " * 55
        assert self.clf.classify(short) != Complexity.COMPLEX
        c = self.clf.classify(long_q + "implement something")
        assert c == Complexity.COMPLEX

    def test_context_adds_score(self):
        _ORDER = {Complexity.SIMPLE: 0, Complexity.MODERATE: 1, Complexity.COMPLEX: 2}
        q = "implement this"
        without = self.clf.classify(q, context=None)
        with_ctx = self.clf.classify(q, context=["a", "b", "c", "d", "e"])
        assert _ORDER[with_ctx] >= _ORDER[without]


class TestLLMClassifier:
    def _mock_client(self, response_text):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        return client

    def test_returns_llm_classification(self):
        clf = LLMClassifier(client=self._mock_client("COMPLEX"))
        assert clf.classify("anything") == Complexity.COMPLEX

    def test_garbage_response_falls_back(self):
        clf = LLMClassifier(client=self._mock_client("BANANA"))
        result = clf.classify("implement this")
        # "BANANA" is not a valid Complexity, so the fallback
        # RuleBasedClassifier runs. "implement" (+2) minus short
        # word count (-1) = score 1 -> MODERATE.
        assert result == Complexity.MODERATE

    def test_api_error_falls_back(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("down")
        clf = LLMClassifier(client=client)
        result = clf.classify("hello")
        # API error triggers fallback. "hello" is a short, simple query.
        assert result == Complexity.SIMPLE

    def test_no_client_uses_rules(self):
        clf = LLMClassifier(client=None)
        result = clf.classify("Refactor this module")
        assert result == Complexity.COMPLEX

    def test_custom_fallback_is_used_when_no_client(self):
        fallback = RuleBasedClassifier()
        clf = LLMClassifier(client=None, fallback=fallback)
        result = clf.classify("yes")
        assert result == Complexity.SIMPLE


class TestComplexityRouter:
    def setup_method(self):
        self.router = ComplexityRouter()

    def test_simple_routes_to_no_think(self):
        d = self.router.route("What is 2+2?")
        assert d.complexity == Complexity.SIMPLE
        assert d.mode == ThinkingMode.NO_THINK
        assert d.preserve_thinking is False

    def test_complex_routes_to_think_with_preserve(self):
        d = self.router.route("Refactor this entire module for async")
        assert d.complexity == Complexity.COMPLEX
        assert d.mode == ThinkingMode.THINK
        assert d.preserve_thinking is True
        assert d.sampling.temperature == 0.6

    def test_force_thinking_overrides(self):
        router = ComplexityRouter(force_thinking=True)
        d = router.route("yes")
        assert d.mode == ThinkingMode.THINK
        assert d.preserve_thinking is True

    def test_override_mode(self):
        d = self.router.route("Refactor this", override_mode=ThinkingMode.NO_THINK)
        assert d.mode == ThinkingMode.NO_THINK
        assert d.preserve_thinking is False

    def test_sampling_matches_mode(self):
        think = self.router.route("Implement a REST API with auth")
        assert think.sampling.temperature == 0.6
        no_think = self.router.route("yes", override_mode=ThinkingMode.NO_THINK)
        assert no_think.sampling.temperature == 0.7

    def test_confidence_is_always_1(self):
        d = self.router.route("Implement something complex step by step")
        assert d.confidence == 1.0

    def test_reasoning_includes_classification(self):
        d = self.router.route("What is 2+2?")
        assert "simple" in d.reasoning.lower()


class TestRuleBasedClassifierBranches:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_single_code_indicator_adds_one(self):
        result = self.clf.classify("does return help here")
        assert result == Complexity.MODERATE

    def test_word_count_21_to_50_adds_one(self):
        query = (
            "Explaining how this particular mechanism operates and what makes "
            "it quite different from others in the provided context today is helpful."
        )
        word_count = len(query.split())
        assert 20 < word_count <= 50
        result = self.clf.classify(query)
        assert result == Complexity.MODERATE

    def test_more_than_four_sentences_adds_two(self):
        query = (
            "This is sentence one. "
            "This is sentence two. "
            "This is sentence three. "
            "This is sentence four. "
            "This is sentence five."
        )
        result = self.clf.classify(query)
        assert result == Complexity.MODERATE

    def test_three_to_four_sentences_adds_one(self):
        import re

        query = "This is sentence one. This is sentence two. This is sentence three."
        sentence_count = len(re.split(r"[.!?]+", query))
        assert 2 < sentence_count <= 4, f"expected 3-4 parts, got {sentence_count}"
        result = self.clf.classify(query)
        assert result == Complexity.MODERATE
