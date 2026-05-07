# qwen-think

Python library for managing Qwen3.6's `enable_thinking` flag across backends (vLLM, SGLang, DashScope, llama.cpp), with sampling parameter sync and context budget tracking.

Handles three things:
- **Backend normalization** -- each backend expects `enable_thinking` in a different payload shape; this library builds the correct one.
- **Sampling parameter swap** -- thinking and non-thinking modes need different sampling configs; the swap is atomic with the mode toggle.
- **Context budget guard** -- tracks token usage and warns/trims/refuses when available context drops below 128K (Qwen3.6's recommended minimum for thinking quality).

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

## Error Handling

The library raises specific exceptions you should handle:

```python
from qwen_think import ThinkingSession

session = ThinkingSession(client, backend="vllm", budget=200_000, min_context=128_000)

try:
    response = session.chat("refactor this module")
except RuntimeError as e:
    # Raised when the context budget is exhausted (used tokens leave
    # less than min_context available). Trim history or start a new session.
    print(f"Budget exhausted: {e}")
except ValueError as e:
    # Raised on unknown backend or unrecognized base_url during auto-detection.
    # Pass backend= explicitly to avoid auto-detection.
    print(f"Configuration error: {e}")
```

Check budget status before it becomes critical:

```python
status = session.budget_status
if status.action.value == "warn":
    session.trim_history(keep_recent=4)
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

**Fix**: This library always explicitly sets the boolean -- never omits it.

### Ray Serve (#52979)

`enable_thinking: false` in the HTTP body doesn't propagate to the model, and thinking output continues appearing.

**Fix**: The normalized payload always includes the explicit flag.

## Sampling Parameters

Qwen3.6 uses different sampling params per mode:

**Thinking mode:**
```python
temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
presence_penalty=1.5, repetition_penalty=1.0
```

**Non-thinking mode:**
```python
temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
presence_penalty=1.5
```

The library swaps these atomically when toggling `enable_thinking`.

## Context Budget

`BudgetManager` tracks token usage against a total budget and enforces a 128K minimum available context. Actions: warn, auto-compress older messages, or refuse.

```python
session = ThinkingSession(client, budget=200_000, min_context=128_000)
status = session.budget_status
# BudgetStatus(total_tokens=200000, used_tokens=10000, available_tokens=190000,
#              min_context=128000, action=BudgetAction.OK,
#              message="Available: 190,000 of 200,000 tokens.")
```

## Thinking Preservation

`preserve_thinking` retains thinking content across conversation turns.

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

```python
from qwen_think import ComplexityRouter

router = ComplexityRouter()
decision = router.route("implement a REST API with authentication")
# RouterDecision(complexity=MODERATE, mode=THINK, preserve_thinking=False)
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
session.thinking_mode  # -> ThinkingMode.THINK
session.budget_status  # -> BudgetStatus(...)
```

## License

Apache-2.0
