"""Complexity classifier and thinking mode router."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Union

from .sampling import SamplingManager
from .types import (
    Complexity,
    RouterDecision,
    ThinkingMode,
)

logger = logging.getLogger("qwen-think.router")

_COMPLEX_PATTERNS = [
    r"\brefactor\b",
    r"\bimplement\b",
    r"\bdebug\b",
    r"\barchitect\b",
    r"\banalyze\b",
    r"\bdesign\b",
    r"\bscaffold\b",
    r"\brewrite\b",
    r"\boptimize\b",
    r"\bcode\s+review\b",
    r"\bmulti[- ]?step\b",
    r"\bchain\s+of\b",
    r"\bstep[- ]?by[- ]?step\b",
    r"\bthen\s+.*\bthen\b",
    r"\bafter\s+that\b",
    r"\bnext\s*,\s*\b",
    r"\bfirst\b.*\bsecond\b",
    r"\bplan\b.*\bexecute\b",
    r"\bbreak\s+down\b",
    r"\bcodebase\b",
    r"\brepository\b",
    r"\bmodule\b",
    r"\bservice\b.*\bintegrate?\b",
]

_SIMPLE_PATTERNS = [
    r"^(what|who|when|where|how\s+many|is\s+it)\b",
    r"^(yes|no|ok|sure|thanks?|please)\b",
    r"\bdefine\b",
    r"\btranslate\b",
    r"\bsummarize\b",
    r"\bconvert\b",
    r"\bformat\b",
]

_CODE_INDICATORS = [
    r"```",
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\bfunction\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\breturn\s+",
    r"[{}\[\]();]",
    r"\bif\s+.*:\s*$",
    r"\bfor\s+.*\bin\b",
]

_COMPLEX_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _COMPLEX_PATTERNS]
_SIMPLE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _SIMPLE_PATTERNS]
_CODE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _CODE_INDICATORS]


class RuleBasedClassifier:
    """Classify query complexity using heuristic pattern matching."""

    def classify(self, query: str, context: Optional[List[str]] = None) -> Complexity:
        complexity_score = 0

        for pattern in _COMPLEX_RE:
            if pattern.search(query):
                complexity_score += 2

        code_matches = sum(1 for pattern in _CODE_RE if pattern.search(query))
        if code_matches >= 2:
            complexity_score += 3
        elif code_matches >= 1:
            complexity_score += 1

        for pattern in _SIMPLE_RE:
            if pattern.search(query):
                complexity_score -= 2

        word_count = len(query.split())
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        elif word_count <= 5:
            complexity_score -= 1

        if context and len(context) > 3:
            complexity_score += 1

        sentence_count = len(re.split(r"[.!?]+", query))
        if sentence_count > 4:
            complexity_score += 2
        elif sentence_count > 2:
            complexity_score += 1

        if complexity_score >= 3:
            return Complexity.COMPLEX
        elif complexity_score >= 0:
            return Complexity.MODERATE
        else:
            return Complexity.SIMPLE


class LLMClassifier:
    """Classify query complexity using a small LLM, with rule-based fallback."""

    CLASSIFICATION_PROMPT = """Classify the following query into exactly one category:
- SIMPLE: Single-turn, factual, short queries (e.g., "what is X?", "translate Y")
- MODERATE: Multi-sentence queries requiring some reasoning
- COMPLEX: Multi-step reasoning, coding tasks, debugging, refactoring

Query: {query}

Previous context length: {context_length}

Respond with exactly one word: SIMPLE, MODERATE, or COMPLEX"""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "Qwen/Qwen3-0.6B",
        fallback: Optional[RuleBasedClassifier] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.fallback = fallback or RuleBasedClassifier()

    def classify(self, query: str, context: Optional[List[str]] = None) -> Complexity:
        if self.client is None:
            return self.fallback.classify(query, context)

        try:
            prompt = self.CLASSIFICATION_PROMPT.format(
                query=query,
                context_length=len(context) if context else 0,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                extra_body={"enable_thinking": False},
            )

            result = response.choices[0].message.content.strip().upper()

            for complexity in Complexity:
                if complexity.value.upper() in result:
                    return complexity

            return self.fallback.classify(query, context)

        except Exception as e:
            logger.warning("LLM classification failed: %s. Falling back to rules.", e)
            return self.fallback.classify(query, context)


class ComplexityRouter:
    """Routes queries to the appropriate thinking mode based on complexity."""

    COMPLEXITY_MAP = {
        Complexity.SIMPLE: (ThinkingMode.NO_THINK, False),
        Complexity.MODERATE: (ThinkingMode.THINK, False),
        Complexity.COMPLEX: (ThinkingMode.THINK, True),
    }

    def __init__(
        self,
        classifier: Optional[Union[RuleBasedClassifier, LLMClassifier]] = None,
        sampling_manager: Optional[SamplingManager] = None,
        force_thinking: bool = False,
    ) -> None:
        self.classifier = classifier or RuleBasedClassifier()
        self.sampling_manager = sampling_manager or SamplingManager()
        self.force_thinking = force_thinking

    def classify(self, query: str, context: Optional[List[str]] = None) -> Complexity:
        return self.classifier.classify(query, context)

    def route(
        self,
        query: str,
        context: Optional[List[str]] = None,
        override_mode: Optional[ThinkingMode] = None,
    ) -> RouterDecision:
        complexity = self.classify(query, context)
        mode, preserve = self.COMPLEXITY_MAP[complexity]

        if self.force_thinking:
            mode = ThinkingMode.THINK
            preserve = True

        if override_mode is not None:
            mode = override_mode
            if mode == ThinkingMode.NO_THINK:
                preserve = False

        sampling = self.sampling_manager.get_config(mode)

        reasoning_parts = [
            f"Classified as {complexity.value}",
        ]
        if self.force_thinking:
            reasoning_parts.append("force_thinking=True, overriding to THINK")
        if override_mode is not None:
            reasoning_parts.append(f"Explicitly overridden to {mode.value}")

        return RouterDecision(
            complexity=complexity,
            mode=mode,
            preserve_thinking=preserve,
            sampling=sampling,
            confidence=1.0,
            reasoning="; ".join(reasoning_parts),
        )
