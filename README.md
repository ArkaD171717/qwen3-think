# qwen-think

Manages Qwen3.6's `enable_thinking` flag across inference backends (vLLM, SGLang, DashScope, llama.cpp). Handles the per-backend payload differences, swaps sampling parameters when toggling thinking mode, and tracks context token budget against the 128K minimum.

## Install

```bash
pip install qwen-think
pip install qwen-think[openai]  # with OpenAI client support
```

## Usage

```python
from openai import OpenAI
from qwen_think import ThinkingSession

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
session = ThinkingSession(client, backend="vllm", budget=200_000)

response = session.chat("refactor this module", preserve=True)
```

The session auto-detects query complexity and sets thinking mode accordingly. You can override:

```python
from qwen_think import ThinkingMode

session.chat("what is 2+2", mode=ThinkingMode.NO_THINK)
```

## Error handling

```python
session = ThinkingSession(client, backend="vllm", budget=200_000, min_context=128_000)

try:
    response = session.chat("refactor this module")
except RuntimeError as e:
    # Budget exhausted -- trim history or start a new session
    print(f"Budget exhausted: {e}")
except ValueError as e:
    # Unknown backend or failed auto-detection
    print(f"Configuration error: {e}")
```

Check budget before it becomes critical:

```python
status = session.budget_status
if status.action.value == "warn":
    session.trim_history(keep_recent=4)
```

## Backend payload formats

Each backend expects `enable_thinking` in a different shape:

| Backend | Format | Notes |
|---------|--------|-------|
| vLLM / SGLang | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | Nested |
| DashScope | `extra_body={"enable_thinking": False}` | Top-level |
| llama.cpp | `--chat-template-kwargs '{"enable_thinking": false}'` | Server-side only |

## Known bugs addressed

**vLLM semantic router (#858)** -- the router removes `enable_thinking` instead of explicitly setting it false. Since Qwen3.6 thinks by default, omitting the field has no effect. This library always sets the boolean explicitly.

**llama.cpp (#20182)** -- `enable_thinking: false` via `--chat-template-kwargs` doesn't reliably disable thinking. Workaround: combine with `--reasoning-budget 0`.

## Sampling parameters

Different sampling configs per mode (from Alibaba's recommendations):

| | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking | 0.6 | 0.95 | 20 | 1.5 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

The library swaps these atomically when toggling mode.

## Complexity router

Rule-based classifier maps queries to modes:

| Classification | Mode | Preserve thinking | Typical query |
|---|---|---|---|
| SIMPLE | NO_THINK | No | "what is X?" |
| MODERATE | THINK | No | multi-sentence reasoning |
| COMPLEX | THINK | Yes | coding, debugging |

```python
from qwen_think import ComplexityRouter

router = ComplexityRouter()
decision = router.route("implement a REST API with authentication")
# decision.mode == ThinkingMode.THINK
```

## License

Apache-2.0
