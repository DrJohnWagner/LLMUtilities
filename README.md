# LLMUtilities

A provider-neutral Python interface over several inconsistent LLM provider SDKs.

The core idea: give it a system instruction and a user instruction, pick a provider,
get text back. The library hides SDK-specific request construction, response parsing,
authentication and common failure modes — without pretending every provider has
identical capabilities, usage semantics or pricing. Provider-specific behaviour stays
owned by the provider; the generic layer only resolves, checks and delegates.

Supported providers and what each one implements:

| Provider   | Chat | Image | Embedding | Token counting |
|------------|:----:|:-----:|:---------:|:---------------:|
| OpenAI     | ✓    | ✓     | ✓         | ✓ (local, tiktoken) |
| Anthropic  | ✓    |       |           | ✓ (live API)     |
| Google     | ✓    |       | ✓         | ✓ (live API)     |
| Moonshot   | ✓    |       |           |                  |
| DeepSeek   | ✓    |       |           |                  |

## Installation

The base package has no provider SDK dependencies. Install only the providers you need:

```bash
pip install "LLMUtilities[openai]"
pip install "LLMUtilities[anthropic]"
pip install "LLMUtilities[google]"
pip install "LLMUtilities[moonshot]"     # uses the openai package under the hood
pip install "LLMUtilities[deepseek]"    # uses the openai package under the hood
pip install "LLMUtilities[openai,anthropic,google]"
pip install "LLMUtilities[all]"         # every provider SDK
```

Asking for a provider whose SDK isn't installed raises `MissingDependencyError`
rather than failing at import time — `import LLMUtilities` always works, and
resolving one provider never imports another provider's SDK.

## Quick start

```python
from LLMUtilities import chat_text

text = chat_text(
    provider_name="anthropic",
    system="You are a poet who prefers restraint over ornament.",
    user="Write a haiku about recursion.",
)
print(text)
```

`chat_text` returns just the response string. `chat` returns the full `ChatResponse`
(text, provider, requested/resolved model, usage, stop reason, raw SDK response):

```python
from LLMUtilities import chat

response = chat(provider_name="openai", user="Summarise the plot of Hamlet in one sentence.")
print(response.text)
print(response.requested_model, "->", response.resolved_model)
print(response.usage.total_input_tokens, response.usage.total_output_tokens)
```

A provider may accept an alias but bill (and report back) a more specific, dated
model — that's why both `requested_model` and `resolved_model` are tracked.

## Providers and capabilities

A central registry answers "which providers does this version know about" without
importing any SDK, and lazily constructs the one you ask for:

```python
from LLMUtilities import list_providers, get_provider

list_providers()
# ['anthropic', 'deepseek', 'google', 'moonshot', 'openai']

openai = get_provider("openai")
openai.capabilities()      # ['chat', 'image_generation', 'embedding', 'token_counting']
openai.list_chat_models()  # static catalogue bundled with the library
```

Calling a capability a provider doesn't implement raises
`UnsupportedCapabilityError` immediately, rather than failing deep inside a request:

```python
from LLMUtilities import generate_image, get_provider

generate_image(provider=get_provider("anthropic"), prompt="a fox in the snow")
# UnsupportedCapabilityError: Provider 'anthropic' does not implement 'ImageGenerationCapability'.
```

## Image generation

```python
from LLMUtilities import generate_image

response = generate_image(provider_name="openai", prompt="a lighthouse at dusk, watercolour")
artifact = response.artifacts[0]
print(artifact.mime_type, len(artifact.b64_data or ""))
```

## Embeddings

```python
from LLMUtilities import embed_texts, cosine_similarity
from LLMUtilities.embeddings import embed_text

response = embed_texts(["cats", "dogs", "astrophysics"], provider_name="openai")
similarity = cosine_similarity(response.vectors[0], response.vectors[1])

vector = embed_text("a single string", provider_name="google")
```

## Token counting

```python
from LLMUtilities import count_text_tokens, count_chat_request_tokens

result = count_text_tokens("How many tokens is this?", provider_name="openai")
print(result.count, result.method)  # method: "exact" | "provider_reported" | "local_estimate"

count_chat_request_tokens(
    system="Be terse.", user="Explain monads.", provider_name="anthropic"
)
```

OpenAI's counts are a local `tiktoken` estimate; Anthropic and Google report exact
counts from a live API call — `result.method` tells you which you got.

## Costs

```python
from LLMUtilities import chat, get_cost_summary

response = chat(provider_name="openai", user="hello")
summary = get_cost_summary(response)
print(summary.input_cost, summary.output_cost, summary.total_cost, summary.currency)
```

`get_cost_summary` resolves the provider from `response.provider` and delegates —
it performs no billing logic itself. The returned `CostSummary` is a deliberately
small, common projection (`input_cost`, `output_cost`, `other_cost`, `total_cost`,
`currency`, plus requested/resolved model). Each provider decides how its detailed
charges roll into those categories — Anthropic, for instance, folds ordinary input,
cache reads *and* cache writes into `input_cost`.

## Provider-specific detail (the escape hatch)

The generic API is intentionally small. A provider-aware caller can go around it
for anything more specific — detailed usage, detailed cost breakdowns, and the raw
pricing record used:

```python
from LLMUtilities import get_provider

anthropic = get_provider("anthropic")
response = anthropic.chat(...)

anthropic.get_detailed_usage(response)   # AnthropicChatUsageDetails: cache reads/writes broken out
anthropic.get_detailed_cost(response)    # AnthropicChatCostDetails: cost per category
anthropic.get_pricing("claude-sonnet-5") # AnthropicChatPricing: the raw rate card
response.raw                             # the untouched SDK response object
```

Every provider's pricing catalogue is static data bundled with the library
(`providers/<name>/pricing.json`) — no network call, no credentials required to
inspect models or pricing.

## Structured output

Repairs and retries a model's JSON output against a Pydantic schema when the
provider has no native structured-output mode:

```python
from pydantic import BaseModel
from LLMUtilities.parsing import generate_structured_output

class PoemSummary(BaseModel):
    title: str
    themes: list[str]
    rating: int

result = generate_structured_output(
    user_prompt="Summarise this poem: ...",
    output_model=PoemSummary,
    provider_name="openai",
    system="You are a precise literary analyst.",
)
print(result.themes)
```

On a parse failure it retries once with a repair prompt that shows the model its
own broken output before giving up.

## Comparing outputs

```python
from LLMUtilities.compare import compare_outputs, print_comparison

comparison = compare_outputs(
    output_a=response_a.text,
    output_b=response_b.text,
    use_embeddings=True,
    embedding_provider="openai",
    use_judge=True,
    judge_provider="anthropic",
)
print_comparison(comparison)
```

`use_judge` sends both outputs to an LLM judge and records its verdict alongside
exact-match, normalised-match, length/word-count and embedding-similarity stats.

## Tracing

```python
from LLMUtilities.tracing import log_chat_request, log_chat_response

log_chat_request("logs/traces.jsonl", request, provider="openai")
log_chat_response("logs/traces.jsonl", response, cost_summary=summary)
```

Appends JSONL records with truncated content, usage and (optionally) cost summary
and raw response — cheap to grep, safe to ship large prompts through without
blowing up a log file.

## Configuration

Environment variables (a `.env` file is loaded automatically if `python-dotenv` is
installed):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `MOONSHOT_API_KEY`, `DEEPSEEK_API_KEY` | Provider credentials |
| `LLMUTILITIES_DEFAULT_PROVIDER` | Used when no `provider`/`provider_name` is given (default: `openai`) |
| `OPENAI_CHAT_MODEL`, `ANTHROPIC_CHAT_MODEL`, `GOOGLE_CHAT_MODEL`, `MOONSHOT_CHAT_MODEL`, `DEEPSEEK_CHAT_MODEL` | Default chat model per provider |
| `OPENAI_IMAGE_MODEL`, `OPENAI_IMAGE_SIZE`, `OPENAI_IMAGE_QUALITY`, `OPENAI_IMAGE_BACKGROUND`, `OPENAI_IMAGE_FORMAT` | OpenAI image defaults |
| `OPENAI_EMBEDDING_MODEL`, `GOOGLE_EMBEDDING_MODEL` | Default embedding model per provider |
| `LLMUTILITIES_REQUEST_TIMEOUT_SECONDS`, `LLMUTILITIES_MAX_RETRIES` | Transport defaults (120s / 3 retries) |

Constructing a provider never requires credentials — only calling a capability
method that needs them does. `get_settings()`/`reload_settings()` are exposed for
callers that need to pick up environment changes at runtime.

## Errors

All exceptions derive from `LLMUtilitiesError`: `ConfigurationError`,
`AuthenticationError`, `RateLimitError`, `RequestError`, `ResponseError`,
`ProviderError`, `MissingDependencyError`, `UnsupportedCapabilityError`,
`PricingUnavailableError`, `CostCalculationUnavailableError`. Providers translate
SDK-specific exceptions into these categories, so callers never need to catch
`openai.RateLimitError` and `anthropic.RateLimitError` separately.

## Development

```bash
pip install -e ".[all,test,dev]"
pytest
ruff check src tests
```

The test suite mocks every provider SDK at the transport boundary — it makes no
network calls and needs no real API keys.

## Design

The full architectural rationale — capability protocols, provider-owned pricing,
transport vs. provider responsibilities, the deferred-decisions list — is in
[`DESIGN-2.0.md`](DESIGN-2.0.md).
