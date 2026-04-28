# qwen-think

**The Thinking Session Manager for Qwen3.6** — SDK + Router + Bug Fixes

The canonical Python library for managing Qwen3.6's thinking state across sessions, backends, and frameworks.

**[Colab Notebook](../qwen_think_colab.ipynb)** — end-to-end test harness (unit tests + live GPU tests)

## What It Solves

1. **Backend Normalization** — Qwen3.6's `enable_thinking` flag has three different invocation patterns across backends. This library normalizes them into a single API.

2. **Atomic Sampling Swap** — Qwen3.6 requires *different* sampling parameters for thinking vs. non-thinking mode. A router that flips `enable_thinking` without also swapping params produces silently degraded output.

3. **Context Budget Guard** — Qwen3.6 advises maintaining at least 128K tokens of context to preserve thinking capabilities. This library tracks and guards against silent degradation.

## Installation

```bash
pip install qwen-think
# With OpenAI client support:
pip install qwen-think[openai]
```

## Quick Start

```python
from openai import OpenAI
from qwen_think import ThinkingSession

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
session = ThinkingSession(client, backend="vllm", budget=200_000)

# Auto-routes: detects complexity, sets mode, swaps sampling
response = session.chat("refactor this module", preserve=True)
```

## The Three Backend Patterns

| Backend | Flag Format | Notes |
|---------|------------|-------|
| **vLLM / SGLang** | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | Nested |
| **DashScope** | `extra_body={"enable_thinking": False}` | Top-level |
| **llama.cpp** | `--chat-template-kwargs '{"enable_thinking": false}'` | Server-side only |

## Bug Fixes Included

### vLLM Semantic Router (#858)

When `use_reasoning: false` is configured, the router *removes* the field instead of explicitly setting `enable_thinking: false`. Since Qwen3.6 thinks by default, removing the field has no effect.

**Fix**: This library always explicitly sets the boolean — never omits it.

### Ray Serve (#52979)

`enable_thinking: false` in the HTTP body doesn't propagate to the model, and thinking output continues appearing.

**Fix**: The normalized payload always includes the explicit flag.

## Sampling Parameters

Qwen3.6 requires different sampling params depending on thinking mode:

**Thinking mode:**
```python
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
presence_penalty=1.5, repetition_penalty=1.0
```

**Instruct / non-thinking mode:**
```python
temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
presence_penalty=1.5
```

A router that flips `enable_thinking` without atomically swapping these params produces **silently degraded output** — not incorrect, just suboptimal in ways that don't surface as errors.

## Context Budget

Qwen3.6 advises maintaining a context length of at least **128K tokens** to preserve thinking capabilities. The `BudgetManager` tracks usage and:

- **Warns** when approaching the threshold
- **Auto-compresses** older messages when running low
- **Refuses** to continue when below the minimum

```python
session = ThinkingSession(client, budget=200_000, min_context=128_000)
status = session.budget_status
# BudgetStatus(total_tokens=200000, used_tokens=45000, available_tokens=155000,
#              action=BudgetAction.OK, message="Context usage: 22.5%...")
```

## Thinking Preservation

Qwen3.6 introduces `preserve_thinking` — a feature that retains thinking context across conversation history, improving reasoning quality for iterative development.

```python
session = ThinkingSession(client, preserve_thinking=True)
# Thinking content is cached and included in subsequent turns
```

## Complexity Router

The router classifies query complexity and selects the appropriate mode:

| Complexity | Mode | Preserved | Use Case |
|-----------|------|-----------|----------|
| SIMPLE | NO_THINK | No | "What is X?" |
| MODERATE | THINK | No | Multi-sentence reasoning |
| COMPLEX | THINK + preserve | Yes | Coding, debugging, refactoring |
| AGENTIC | THINK + preserve + budget | Yes | Multi-step workflows |

```python
from qwen_think import ComplexityRouter

router = ComplexityRouter()
decision = router.route("implement a REST API with authentication")
# RouterDecision(complexity=COMPLEX, mode=THINK, preserve_thinking=True, ...)
```

## API Reference

### ThinkingSession

```python
session = ThinkingSession(
    client,                # OpenAI-compatible client
    backend="vllm",       # or "sglang", "dashscope", "llamacpp"
    model="Qwen/Qwen3.6-35B-A3B",
    budget=200_000,        # Total context budget
    min_context=128_000,   # 128K minimum for thinking
    preserve_thinking=True,
    auto_route=True,       # Auto-classify complexity
)
```

### Manual Mode Control

```python
# Force a specific mode
session.chat("quick answer", mode=ThinkingMode.NO_THINK)

# Let the router decide
session.chat("refactor this module")

# Check current state
session.thinking_mode  # → ThinkingMode.THINK
session.budget_status  # → BudgetStatus(...)
```

## License

Apache-2.0
