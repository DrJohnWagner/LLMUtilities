# LLMUtilities

Provider-normalised helpers for LLM chat, embeddings, image generation, token counting,
cost estimation, structured parsing, tracing and output comparison.

Supported providers:

- OpenAI
- Anthropic
- Google Gemini
- Moonshot AI
- DeepSeek

## Install

```bash
pip install pydantic

# Provider SDKs (install what you use)
pip install openai
pip install anthropic
pip install google-genai

# Optional
pip install tiktoken      # OpenAI token counting
pip install python-dotenv # .env loading in config.py
```

Missing SDKs are loaded lazily. You only get `MissingDependencyError` when you call a
feature that needs that SDK.

## Configuration

Set keys in environment variables (or `.env`):

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...  # GEMINI_API_KEY also works
MOONSHOT_API_KEY=...
DEEPSEEK_API_KEY=...
```

Common settings:

| Variable                                 | Default                    |
| ---------------------------------------- | -------------------------- |
| `LLMUTILITIES_DEFAULT_PROVIDER`        | `openai`                 |
| `LLMUTILITIES_REQUEST_TIMEOUT_SECONDS` | `120`                    |
| `LLMUTILITIES_MAX_RETRIES`             | `3`                      |
| `OPENAI_CHAT_MODEL`                    | `gpt-5-mini`             |
| `ANTHROPIC_CHAT_MODEL`                 | `claude-sonnet-4-6`      |
| `GOOGLE_CHAT_MODEL`                    | `gemini-2.5-flash`       |
| `OPENAI_EMBEDDING_MODEL`               | `text-embedding-3-small` |
| `GOOGLE_EMBEDDING_MODEL`               | `text-embedding-004`     |
| `OPENAI_IMAGE_MODEL`                   | `gpt-image-1.5`          |
| `MOONSHOT_CHAT_MODEL`                  | `kimi-k2.6`              |
| `DEEPSEEK_CHAT_MODEL`                  | `deepseek-v4-flash`      |

Image defaults can also be set via `OPENAI_IMAGE_SIZE`, `OPENAI_IMAGE_QUALITY`,
`OPENAI_IMAGE_BACKGROUND`, and `OPENAI_IMAGE_FORMAT`.

For OpenAI image generation:

- `gpt-image-1.5` remains the default model.
- `gpt-image-2` is supported via `model="gpt-image-2"` or `OPENAI_IMAGE_MODEL=gpt-image-2`.
- `gpt-image-2` accepts `size="auto"` or explicit `WIDTHxHEIGHT` values matching OpenAI constraints.

## Quick start

```python
from LLMUtilities.chat import chat

response = chat(
    provider_name="openai",
    system="You are concise.",
    user="Write a two-line poem about recursion.",
)

print(response.text)
if response.usage:
    print(response.usage.input_tokens, response.usage.output_tokens)
```

## Provider support matrix

| Feature          | OpenAI            | Anthropic | Google   | Moonshot AI | DeepSeek |
| ---------------- | ----------------- | --------- | -------- | ----------- | -------- |
| Chat             | yes               | yes       | yes      | yes         | yes      |
| Embeddings       | yes               | no        | yes      | no          | no       |
| Image generation | yes               | no        | no       | no          | no       |
| Token counting   | local`tiktoken` | API call  | API call | no          | no       |

Notes:

- Chat adapters normalise output to `ChatResponse`.
- OpenAI chat uses the Responses API (`client.responses.create`).
- Unsupported features raise `ConfigurationError` or `ProviderError`.

## Chat API

```python
from LLMUtilities.chat import chat, chat_text, chat_usage

text = chat_text(provider_name="anthropic", user="Say hello.")
usage = chat_usage(provider_name="google", user="Count to ten.")

full = chat(provider_name="openai", user="Give me one sentence.")
print(full.text)
```

You can also pass a provider instance with a `.chat(...)` method:

```python
from LLMUtilities.providers.openai import OpenAIChatModel
from LLMUtilities.chat import chat

provider = OpenAIChatModel(model="gpt-5-mini")
response = chat(provider=provider, user="Hello")
```

## Message schema and multimodal content

`Message.content` accepts either:

- a non-empty string
- a non-empty list of content parts (`TextContentPart` / `ImageContentPart`)

```python
from LLMUtilities import Message, TextContentPart, ImageContentPart

msg = Message(
    role="user",
    content=[
        TextContentPart(type="text", text="Describe this image"),
        ImageContentPart(type="image", source={"type": "url", "url": "https://example.com/a.png"}),
    ],
)
```

Current provider adapters extract text parts when list content is passed.

## Embeddings

```python
from LLMUtilities.embeddings import embed_text, embed_texts, cosine_similarity

v1 = embed_text("hello", provider="openai")
v2 = embed_text("world", provider="openai")
print(cosine_similarity(v1, v2))

google_vectors = embed_texts(
    ["query text"],
    provider="google",
    task_type="RETRIEVAL_QUERY",
)
```

Anthropic embeddings are not available in this package.

## Image generation

```python
from LLMUtilities.image import generate_image, generate_image_b64

img = generate_image(
    provider_name="openai",
    prompt="A moonlit harbor in ink",
    model="gpt-image-2",
    size="1024x1024",
    quality="high",
    format="png",
)

first = img.artifacts[0]
print(first.mime_type, bool(first.b64_data), bool(first.url))

b64 = generate_image_b64(provider_name="openai", prompt="A raven in woodcut style")
```

`generate_image_b64(...)` raises `ResponseError` if the first artifact has no base64 data.

## Token counting

```python
from LLMUtilities.tokens import count_text_tokens, count_message_tokens
from LLMUtilities.types import Message

n1 = count_text_tokens("Hello", provider="openai")
n2 = count_text_tokens("Hello", provider="anthropic")
n3 = count_text_tokens("Hello", provider="google")

messages = [Message(role="user", content="Count my tokens")]
n4 = count_message_tokens(messages, provider="openai")
```

For OpenAI, token counting requires `tiktoken`.
For multimodal `Message.content` lists, token counting uses text parts and ignores image parts.

Moonshot AI and DeepSeek use the OpenAI chat completions API with provider-specific base URLs.

## Cost estimation

```python
from LLMUtilities.costs import estimate_cost, estimate_image_cost, get_pricing, print_cost_breakdown
from LLMUtilities.costs import cost_for_image_response, print_image_cost_breakdown
from LLMUtilities.types import ChatUsage

usage = ChatUsage(input_tokens=1200, output_tokens=300)
estimate = estimate_cost(model="claude-sonnet-4.6", usage=usage)
print_cost_breakdown(estimate=estimate, model="claude-sonnet-4.6")

batch_estimate = estimate_cost(
    model="gpt-5.4",
    usage=ChatUsage(input_tokens=1000, output_tokens=500),
    pricing_mode="batch",
)

img_estimate = estimate_image_cost(
    model="gpt-image-1.5",
    size="1024x1024",
    quality="medium",
    image_count=3,
)
print(img_estimate.total_cost_usd)

# Add token-based image costs to reference output estimates.
img_with_usage = estimate_image_cost(
    model="gpt-image-2",
    size="1024x1024",
    quality="low",
    image_count=1,
    text_input_tokens=1200,
    cached_text_input_tokens=200,
    image_input_tokens=640,
    image_output_tokens=240,
)
print_image_cost_breakdown(estimate=img_with_usage)

# Exact post-response costing from returned usage.
exact_img_cost = cost_for_image_response(
    response=img,
    size="1024x1024",
    quality="high",
)
print(exact_img_cost.total_cost_usd)

pricing = get_pricing("gemini-2.5-pro")
print(pricing.source_url)
```

The bundled pricing table is versioned and stored in `src/LLMUtilities/PRICING.json`.
`Pricing` is the canonical catalogue record and includes the explicit cache, batch,
and long-context fields. Use `get_pricing(...)` to retrieve it. Use `register_pricing(...)` for in-memory overrides.
Image pricing is also catalogue-backed via `src/LLMUtilities/IMAGE_PRICING.json`.
`ImagePricing` and `ImagePricingCatalogue` expose standard and batch token rates plus
reference output costs by quality and size.

Use `cost_for_image_response(...)` or `cost_for_image_usage(...)` for exact token-based
image costing when usage data is available. These paths bill returned image-output tokens
directly and do not add reference output pricing.
Use `estimate_image_cost(...)` for offline reference estimates. That path requires an
explicit listed size and quality, and it is the only place where reference output pricing
is applied.

## Structured output and JSON parsing

```python
from pydantic import BaseModel
from LLMUtilities.parsing import structured_output
from LLMUtilities.parsing.json_parsing import parse_json, parse_json_as

class PoemSummary(BaseModel):
    title: str
    themes: list[str]

result = structured_output(
    user_prompt="Summarise this poem as JSON.",
    output_model=PoemSummary,
    provider_name="openai",
    model="gpt-5-mini",
)

obj = parse_json('{"x": 1}')
typed = parse_json_as('{"title": "T", "themes": ["memory"]}', PoemSummary)
```

`structured_output` is an alias of `generate_structured_output`.

## Output comparison

```python
from LLMUtilities.compare import compare_outputs, print_comparison

comparison = compare_outputs(
    output_a="short answer",
    output_b="long answer",
    use_embeddings=True,
    embedding_provider="openai",
    use_judge=True,
    judge_provider="openai",  # or a provider instance
    judge_model="gpt-5-mini",
)

print_comparison(comparison)
```

## Tracing

```python
from LLMUtilities.tracing.tracing import log_chat_request, log_chat_response, log_error

log_chat_request("logs/traces.jsonl", request, provider="openai", resolved_model="gpt-5-mini")
log_chat_response("logs/traces.jsonl", response)
log_error("logs/traces.jsonl", RuntimeError("example"), provider="openai")
```

Writes JSONL records with `event_type`, `timestamp`, `provider`, `model`, and `payload`.

## Exceptions

All package exceptions inherit from `LLMUtilitiesError`:

- `ConfigurationError`
- `AuthenticationError`
- `MissingDependencyError`
- `RateLimitError`
- `RequestError`
- `ResponseError`
- `ResponseFormatError`
- `ProviderError`
