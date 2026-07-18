

## Public API contract

The package exposes provider-normalised request and response types. Public models use Pydantic unless otherwise noted.

### Chat types

```python
MessageRole = Literal["system", "user", "assistant"]
ContentPart = TextContentPart | ImageContentPart
ContentType = str | list[ContentPart]
```

```python
class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str
```

`text` must be non-empty.

```python
class ImageContentPart(BaseModel):
    type: Literal["image"]
    source: dict
```

```python
class Message(BaseModel):
    role: MessageRole
    content: ContentType
```

String content and content-part lists must be non-empty. Unknown fields are rejected.

```python
class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
```

At least one message is required.

```python
class ChatUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
```

```python
class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: ChatUsage | None = None
    stop_reason: str | None = None
    raw: object | None = None
```

### Image types

The image API uses the following conceptual models:

```python
class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    format: str | None = None
    n: int = 1
    seed: int | None = None
    user: str | None = None
```

The prompt must be non-empty and `n` must be at least one.

```python
class ImageArtifact(BaseModel):
    b64_data: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    mime_type: str | None = None
```

```python
class ImageUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
```

```python
class ImageResponse(BaseModel):
    artifacts: list[ImageArtifact]
    provider: str
    model: str
    usage: ImageUsage | None = None
    raw: object | None = None
```

At least one image artifact is required.

### Provider injection

Functions that accept `provider=` may receive an instantiated provider object instead of a provider name.

A chat provider must implement:

```python
def chat(self, request: ChatRequest) -> ChatResponse:
    ...
```

An image provider must implement:

```python
def generate(self, request: ImageRequest) -> ImageResponse:
    ...
```

This interface is used by the bundled fake provider and may be used by application-specific providers.

---

## Chat construction and provider translation

Passing `messages=` takes precedence over `system=`, `user=` and `assistant=`.

When individual message arguments are used, messages are constructed in this order:

1. system
2. user
3. assistant

Empty optional values are omitted. A chat request must contain at least one resulting message.

Provider names are case-insensitive and surrounding whitespace is ignored.

### OpenAI

OpenAI chat uses the Responses API.

* System messages are removed from the conversational message list.
* Multiple system messages are joined with two newline characters.
* The joined system text is passed as `instructions`.
* User and assistant messages are passed through `input`.
* `temperature` and `max_output_tokens` are omitted when unset.
* Text is collected from every `output_text` part in every output item.
* Multiple text parts are concatenated without inserting additional separators.
* Non-text output parts are ignored.
* Missing or empty textual output raises `ResponseError`.

### Anthropic

Anthropic system prompts are separate from conversational messages.

* Multiple system messages are joined with two newline characters.
* The resulting system prompt is supplied as Anthropic text content.
* The final static system block may use Anthropic prompt caching.
* User and assistant messages remain in the message list.
* `max_output_tokens` maps to `max_tokens`.
* If no maximum is supplied through the request or configuration, the adapter uses 8192.
* Text is collected from all text blocks.
* Tool-use and other non-text blocks are ignored.
* Missing or empty textual output raises `ResponseError`.

### Google Gemini

Google chat uses `google.genai.Client.models.generate_content`.

* System messages are combined into a system instruction.
* User messages map to the Gemini user role.
* Assistant messages map to the Gemini model role.
* Temperature and maximum output tokens are supplied through the generation configuration when set.
* Text is collected from the returned candidate content parts.
* Missing candidates or candidates with no usable text raise `ResponseError`.

### Multimodal content

The public message schema accepts text and image content parts. Current chat adapters extract and forward text content. Image parts are not yet translated into provider-native multimodal requests.

Applications must not assume that accepting `ImageContentPart` implies provider-level image understanding.

---

## Errors and optional dependencies

All package exceptions inherit from `LLMUtilitiesError`. The public hierarchy is:

```text
LLMUtilitiesError
├── ConfigurationError
├── AuthenticationError
├── MissingDependencyError
├── RateLimitError
├── RequestError
├── ResponseError
│   └── ResponseFormatError
└── ProviderError
```

Their intended meanings are:

* `ConfigurationError`: missing or invalid configuration, unsupported provider names or unavailable capabilities
* `AuthenticationError`: credentials were rejected
* `MissingDependencyError`: an optional provider SDK is required but not installed
* `RateLimitError`: the provider rejected the request because of rate limiting
* `RequestError`: the request failed before a usable provider response was returned
* `ResponseError`: the provider response was missing, malformed or unusable
* `ResponseFormatError`: returned content could not be parsed into the required format
* `ProviderError`: another provider-specific failure occurred

Provider SDKs are optional and must be imported lazily. Importing `LLMUtilities` or any provider module must not require all provider SDKs to be installed.

A missing SDK raises `MissingDependencyError` only when the affected feature is invoked. The error message must name the required package and provider.

Invalid local arguments may raise `TypeError`, `ValueError` or Pydantic validation errors rather than being wrapped as provider errors.

---

## Token counting and pricing semantics

### Token counting

OpenAI token counting is local and requires `tiktoken`.

Anthropic and Google token counting use provider APIs and therefore require credentials, the corresponding SDK and network access.

```python
count_text_tokens(
    text,
    provider="openai",
    model=None,
)
```

```python
count_message_tokens(
    messages,
    provider="openai",
    model=None,
)
```

```python
count_chat_request_tokens(
    system=None,
    user=None,
    assistant=None,
    messages=None,
    provider="openai",
    model=None,
)
```

An empty message sequence returns zero.

Provider names are normalised by trimming whitespace and converting to lowercase.

Message counting follows the same message-construction and role-translation rules as chat. Token counts should be treated as estimates unless they come directly from a provider counting endpoint or response.

### Cost estimation

Text pricing is stored in US dollars per one million tokens.

```python
@dataclass(frozen=True)
class Pricing:
    input_per_million_tokens: float
    output_per_million_tokens: float
    cached_input_per_million_tokens: float | None = None
```

```python
@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    cache_creation_input_cost_usd: float = 0.0

    total_cost_usd: float = 0.0
```

The cached-input rate represents cache reads or hits. Cache creation is charged at the ordinary input rate unless explicitly registered otherwise.

Cost calculations use:

```text
token cost = token count / 1,000,000 × applicable rate
```

No currency conversion or automatic rounding is applied.

Pricing tables are snapshots and may become stale. Applications may override or extend them with `register_pricing(...)` and `register_image_pricing(...)`.

Image pricing is represented by model and requested size:

```python
@dataclass(frozen=True)
class ImagePricing:
    per_image_usd: dict[str, float]
    default_per_image_usd: float | None = None
```

Unknown models or unsupported sizes must fail clearly rather than silently returning zero.

---

## Structured output and JSON repair

`generate_structured_output(...)` and its alias `structured_output(...)` use a Pydantic model as the output contract.

The generated prompt instructs the model to:

* Return exactly one JSON object
* Return no markdown fences
* Return no commentary
* Include every required field
* Add no fields outside the schema

The model’s JSON schema is included by default.

Returned text is parsed and validated through `parse_json_as(...)`.

JSON extraction follows this order:

1. The first fenced JSON block
2. The first balanced JSON-looking object or array
3. The complete stripped response

When `strict=False`, parsing may apply light repairs before failing. Repairs may address common LLM output defects such as trailing commas or surrounding commentary. Repair must not silently invent missing semantic values.

When `strict=True`, invalid JSON is returned as the underlying parsing failure without repair.

Validation failures from the supplied Pydantic model must remain distinguishable from JSON syntax failures.

---

## Output comparison

```python
compare_outputs(
    output_a,
    output_b,
    *,
    use_embeddings=True,
    embedding_provider="openai",
    embedding_model=None,
    use_judge=False,
    judge_provider=None,
    judge_model=None,
    judge_system=None,
    judge_prompt=None,
)
```

Both outputs must be strings.

The comparison result records:

* Character length using `len`
* Word count using whitespace splitting
* Exact string equality
* Normalised string equality
* Optional embedding cosine similarity
* Optional LLM judge metadata and verdict

Normalised equality ignores differences in surrounding whitespace, repeated internal whitespace and letter case. It does not perform semantic rewriting.

Embedding comparison embeds each output and calculates cosine similarity. Vectors must have equal dimensions. Zero-length or zero-magnitude vectors must fail clearly.

The judge provider may be either a provider name or an instantiated chat provider. The judge verdict is stored as returned text rather than being forced into an undocumented scoring schema.

Disabling embeddings or judging leaves the corresponding result fields as `None`.

---

## Tracing and data safety

Tracing writes append-only UTF-8 JSON Lines records.

Each record contains:

```python
@dataclass(frozen=True)
class TraceRecord:
    event_type: str
    timestamp: str
    provider: str | None = None
    model: str | None = None
    payload: dict[str, object] = field(default_factory=dict)
```

Timestamps use timezone-aware UTC ISO 8601 strings.

Supported trace events include:

* `chat_request`
* `chat_response`
* `error`

Parent directories are created automatically.

Request records may include:

* Serialised messages
* Requested model
* Resolved model
* Temperature
* Maximum output tokens

Response records may include:

* Truncated response text
* Normalised usage
* Stop reason
* Provider and model

Raw provider responses are excluded unless explicitly requested.

Error records include the exception class and message. Traceback capture is not guaranteed.

Tracing is diagnostic logging, not a hardened audit or security system.

* Prompts and responses may be written in plaintext.
* Truncation is not redaction.
* API keys must never be added intentionally.
* Error messages and caller-supplied extra payloads may contain sensitive data.
* Raw provider responses may contain user data or provider metadata.
* No encryption, secret detection, PII removal, log rotation or retention policy is provided.
* No cross-process file lock or atomic multi-record transaction is provided.
* Callers are responsible for file permissions, storage location, retention and redaction.

---

## Implementation requirements.

An implementation is considered compatible when all of the following hold:

* Every documented import path works.
* README examples run without modification.
* The package imports when none of the optional provider SDKs are installed.
* Missing SDKs fail lazily with `MissingDependencyError`.
* Provider names are normalised consistently.
* Provider instances can be injected for chat and image generation.
* Requests and responses use the documented normalised models.
* Empty or malformed provider responses raise package exceptions.
* Multiple provider text segments are concatenated in their original order.
* Usage fields are normalised where provider data is available.
* Unsupported provider capabilities fail explicitly.
* Structured output is validated against the supplied Pydantic model.
* Pricing and token-count behaviour is deterministic for fixed configuration.
* Tracing produces one valid JSON object per line.
* Core behaviour is covered by offline tests using fake or mocked providers.
* Tests must not require live API credentials.
