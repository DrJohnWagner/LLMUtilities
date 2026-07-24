# LLMUtilities Design

## 1. Purpose

LLMUtilities is a provider-neutral Python interface over several inconsistent LLM provider SDKs.

Its primary purpose is to make the most common interaction simple:

> Provide a system instruction and a user instruction as text, send them to a selected provider and receive the model response as text.

The library should hide SDK-specific request construction, response parsing, authentication handling, model defaults and common failure modes.

It should not pretend that providers have identical APIs, usage semantics or pricing structures. The generic interface standardises caller intent and deliberately limited results. Provider-specific behaviour remains owned by the provider implementation.

Chat is the primary capability. Image generation, embeddings and token counting follow the same provider-centred delegation pattern.

Structured-output parsing and output comparison are secondary capabilities built entirely on top of the generic chat (and, for comparison, embedding) façade. They require no provider-specific implementation of their own: a provider that implements `ChatCapability` automatically gets structured-output support and can be compared against another provider's output. These are not part of the provider abstraction — they are ordinary consumers of it, in the same way a caller's own application code would be.

Audio is explicitly outside the current design.

## 2. Design goals

The design should:

1. Provide a simple, stable and provider-neutral chat interface.
2. Make provider selection explicit but easy.
3. Delegate provider-specific behaviour to provider implementations.
4. Support provider-specific pricing and cost calculation without forcing a universal pricing schema.
5. Provide deliberately limited common response and cost-summary types.
6. Allow callers to obtain a concrete provider and use provider-specific APIs when they need more detail.
7. Separate reusable transport mechanics from provider semantics.
8. Support new capabilities and new providers without expanding central switch statements or universal datatypes.
9. Keep unsupported capabilities explicit.
10. Preserve raw provider responses as an escape hatch and audit aid.
11. Avoid backward-compatibility constraints while the package has no external users.
12. Support lightweight discovery of providers known to LLMUtilities without importing every provider SDK.
13. Allow each concrete provider to expose the models and pricing records it currently knows how to support.

## 3. Non-goals

The design does not attempt to:

- Define one universal pricing datatype for every provider.
- Define one universal detailed usage datatype for every provider.
- Force all providers to support the same capabilities.
- Hide every provider-specific feature behind the generic interface.
- Normalise every provider response field.
- Create a generic pricing rule engine before provider requirements are understood.
- Add audio transcription or speech generation.
- Preserve the current public API or internal class hierarchy.
- Preserve the current tests, which are out of date and will be replaced later.

## 4. Core architectural decision

The package is a collection of provider wrappers behind common capability interfaces.

The generic flow is:

1. A public helper receives a provider-neutral request.
2. The helper resolves the requested provider.
3. The helper checks that the provider implements the required capability.
4. The helper delegates the operation to the provider.
5. The provider performs all provider-specific work.
6. The provider returns a deliberately limited common response.
7. A provider-aware caller may ask the concrete provider for richer provider-specific information.

The generic layer defines what callers want to do.

The provider layer defines how that operation works for a particular provider.

## 5. Provider as the primary public implementation unit

The primary implementation object should be the provider, not a separate public object for each model or capability.

Examples include:

- `OpenAIProvider`
- `AnthropicProvider`
- `GoogleProvider`
- `MoonshotProvider`
- `DeepSeekProvider`

A provider may internally delegate to smaller components, but callers should normally interact with one provider object.

For example, the OpenAI provider may internally contain separate chat, image, embedding, token and pricing components. That internal decomposition should not require callers to instantiate an `OpenAIChatModel` and a separate `OpenAIImageModel`.

Models are provider-owned identifiers or configurations passed to capability methods. They are not the primary service abstraction.

## 6. Capability-based interfaces

Providers should not inherit from one enormous abstract base class containing every possible operation.

Instead, capabilities should be represented by separate protocols or abstract interfaces.

The current capability set is:

- `ChatCapability`
- `ImageGenerationCapability`
- `EmbeddingCapability`
- `TokenCountingCapability`

A provider implements only the capabilities it supports.

### 6.1 ChatCapability

A chat-capable provider is responsible for:

- Resolving a default or requested model
- Translating the common chat request into the provider SDK request
- Applying provider-specific message and system-instruction rules
- Calling the provider SDK
- Translating SDK exceptions into package exceptions
- Extracting response text
- Extracting a deliberately limited common usage summary where practical
- Preserving the raw SDK response
- Returning a common `ChatResponse`
- Calculating provider-specific costs for that response
- Producing a common `CostSummary`
- Exposing provider-specific usage, pricing and detailed cost information through provider-specific methods

### 6.2 ImageGenerationCapability

An image-capable provider is responsible for:

- Resolving the requested image model
- Validating provider-specific image options
- Translating the common image request into the provider SDK request
- Calling the provider SDK
- Extracting generated artefacts
- Extracting provider-specific usage
- Preserving the raw response
- Returning a common `ImageResponse`
- Calculating provider-specific image costs
- Producing a common `CostSummary`
- Exposing provider-specific image pricing and detailed cost information

The current `OpenAIImageModel` should be folded into `OpenAIProvider`.

### 6.3 EmbeddingCapability

An embedding-capable provider is responsible for:

- Resolving the embedding model
- Translating generic embedding input into the provider request
- Applying provider-specific options such as dimensions or task type
- Calling the provider SDK
- Extracting vectors
- Returning a common `EmbeddingResponse`
- Preserving provider-specific metadata where useful
- Calculating costs if embedding pricing is added

### 6.4 TokenCountingCapability

A token-counting provider is responsible for:

- Counting text tokens
- Counting message tokens
- Using the correct provider SDK or local tokenizer
- Applying provider-specific message formatting rules
- Resolving the relevant model
- Reporting whether the count is exact, provider-reported or locally estimated

The generic `tokens.py` module should not contain OpenAI, Anthropic and Google implementations. It should contain public helper functions that resolve the provider and delegate.

## 7. Public generic API

The generic public API should remain small.

Its main purpose is to provide convenient access to common operations without exposing provider SDK details.

### 7.1 Provider discovery and resolution

A central provider registry should initially expose two essential operations:

- `list_providers()` returns the names of all providers that the installed version of LLMUtilities knows how to support.
- `get_provider(name)` lazily imports and returns the requested concrete provider.

For the current package, `list_providers()` would return names equivalent to:

- `openai`
- `anthropic`
- `google`
- `moonshot`
- `deepseek`

`list_providers()` does not import every provider implementation, inspect installed SDKs or determine whether credentials are configured. It answers only:

> Which providers does this version of LLMUtilities know how to support?

`get_provider(name)` performs lazy loading for one provider only. If that provider's required SDK package is unavailable, it raises `MissingDependencyError`. Asking for one provider must not cause every other provider SDK to be imported.

Provider construction should not require an API key merely to inspect provider capabilities, models or pricing. Credentials should be required when a remote operation that needs them is invoked.

Provider registration and third-party provider factories are deferred until a concrete extension requirement exists.

The registry should replace provider-name switch statements scattered through `chat.py`, `image.py`, `tokens.py` and `embeddings.py`.

### 7.2 Chat helpers

The primary helper should accept:

- Provider name or provider object
- System instruction as text
- User instruction as text
- Optional previous messages
- Optional model
- Common generation options such as temperature and maximum output tokens

The most important convenience helper returns only response text.

A richer helper returns `ChatResponse`.

The generic helper must not construct provider SDK requests or parse provider SDK responses.

### 7.3 Image helpers

The generic image helper should accept a deliberately limited common request and delegate it to an image-capable provider.

It should return `ImageResponse`.

Provider-specific options should remain available through the concrete provider API rather than continually expanding the generic helper.

### 7.4 Embedding helpers

The generic embedding helper should delegate to an embedding-capable provider and return `EmbeddingResponse`.

Common options may be exposed where they have consistent meaning. Provider-specific options should remain on the concrete provider.

### 7.5 Token helpers

The generic token helpers should delegate to a token-counting provider.

They should not import provider SDKs or implement provider-specific token rules directly.

### 7.6 Generic cost helpers

Generic cost helpers should return deliberately limited common summaries.

For example, `get_cost_summary()` may return:

- Input cost
- Output cost
- Other cost
- Total cost
- Currency
- Provider
- Requested model
- Resolved model

The generic helper must delegate the calculation of every summary field to the provider.

It must not inspect provider pricing fields or reproduce provider billing rules.

## 8. Common response types

Common response types should contain only information that is genuinely useful across providers.

They should not attempt to represent every provider-specific field.

### 8.1 ChatResponse

A common `ChatResponse` should contain:

- Response text
- Provider identifier
- Requested model identifier
- Resolved or provider-reported model identifier
- Optional common usage summary
- Optional stop reason
- Raw provider response

The distinction between requested and resolved model is important. A provider may accept an alias but bill a dated or otherwise more specific model.

### 8.2 ImageResponse

A common `ImageResponse` should contain:

- Provider identifier
- Requested model identifier
- Resolved model identifier
- Generated image artefacts
- Optional common usage summary
- Raw provider response

Each image artefact may contain:

- MIME type
- Raw bytes or base64 data
- URL
- Width
- Height
- Revised prompt where available
- Other deliberately common metadata

Provider-specific metadata remains accessible through the raw response or provider-specific result types.

### 8.3 EmbeddingResponse

A common `EmbeddingResponse` should contain:

- Provider identifier
- Requested model identifier
- Resolved model identifier
- Embedding vectors
- Optional common usage summary
- Raw provider response

### 8.4 CostSummary

`CostSummary` is a common projection, not a universal detailed cost model.

It should contain at least:

- `input_cost`
- `output_cost`
- `other_cost`
- `total_cost`
- `currency`
- `provider`
- `requested_model`
- `resolved_model`

The provider defines how its detailed charges roll into those common categories.

For example, an Anthropic provider may include ordinary input, cache reads and cache writes in `input_cost`. A future image provider may include prompt processing and source-image processing in `input_cost`.

The generic layer must treat the provider’s summary as authoritative.

## 9. Provider-specific data types

Each provider may define its own Pydantic models for:

- Pricing
- Usage details
- Cost details
- Capability-specific request options
- Capability-specific result metadata
- Model metadata
- Provider capabilities

Examples include:

- `AnthropicChatPricing`
- `AnthropicChatUsageDetails`
- `AnthropicChatCostDetails`
- `OpenAIChatPricing`
- `OpenAIImagePricing`
- `GoogleChatPricing`
- `MoonshotChatPricing`
- `DeepSeekChatPricing`

These types do not need to share identical fields.

A caller that explicitly wants Anthropic pricing should:

1. Ask the generic provider registry for the Anthropic provider.
2. Receive an `AnthropicProvider`.
3. Ask that provider for its pricing.
4. Receive an `AnthropicChatPricing` instance.

This is intentional. Provider-aware callers are allowed to use provider-specific APIs and datatypes.

## 10. Provider-owned pricing and costing

Pricing and cost calculation should be owned by the provider.

The generic package should not maintain a single detailed pricing datatype or a central calculator that attempts to understand every provider.

Each provider is responsible for:

- Loading its pricing catalogue
- Validating its pricing data
- Selecting the correct model pricing
- Selecting the correct effective date
- Interpreting provider usage
- Applying provider-specific caching rules
- Applying provider-specific batch or service-tier rules
- Applying provider-specific long-context rules
- Applying provider-specific modality rules
- Producing detailed provider-specific cost information
- Producing a common `CostSummary`

This removes the need for a universal pricing model whose fields grow as the cross-product of providers, modalities, cache modes, processing modes, context tiers and future billing distinctions.

### 10.1 What provider-specific pricing fixes

Provider-specific pricing types prevent fields relevant to one provider from leaking into every other provider.

Anthropic may model cache-write TTLs.

Google may model modality-specific rates and context tiers.

OpenAI may model OpenAI service tiers, caching and endpoint-specific distinctions.

Moonshot and DeepSeek may model their own cache-hit and cache-miss semantics.

No provider pricing type needs optional fields for every other provider.

### 10.2 What provider-specific pricing does not automatically fix

A provider can still design its own pricing model badly.

For example, a Google pricing model could still flatten every combination of modality, context tier and processing mode into dozens of fields.

Provider-specific types solve the cross-provider field explosion. They do not remove the need for sensible design inside each provider.

The exact provider pricing model shapes are deliberately deferred until each provider’s billing requirements are examined in detail.

## 11. Generic and detailed costing

The design supports two levels of cost access.

### 11.1 Generic costing

A normal caller uses a helper such as `get_cost_summary(response)`.

The helper:

1. Determines the provider from the response.
2. Resolves the provider object.
3. Delegates to the provider.
4. Returns the provider-calculated `CostSummary`.

The helper performs no detailed billing calculation.

### 11.2 Provider-specific costing

A provider-aware caller obtains the concrete provider and requests detailed cost information.

The resulting datatype may be completely provider-specific.

For example, Anthropic detailed costs may distinguish:

- Ordinary input
- Cache reads
- Five-minute cache writes
- One-hour cache writes
- Output
- Batch effects
- Regional effects

Google detailed costs may distinguish different categories.

Only the common summary needs a shared shape.

## 12. Usage interpretation

Providers should own usage extraction and interpretation because SDK field names and semantics differ.

Usage interpretation should occur close to the provider response adapter.

The provider may return:

- A deliberately limited common usage summary in the common response
- A detailed provider-specific usage object through a provider method
- The raw SDK response for unsupported or newly introduced fields

The generic cost helper should never guess how provider usage fields map to billable quantities.

### 12.1 Common usage summary

A common usage summary may contain only stable concepts such as:

- Total input tokens
- Total output tokens
- Total tokens

Even these fields need clear documented semantics.

Detailed cache categories, reasoning tokens, modality breakdowns and tool usage should remain provider-specific unless a genuinely stable common abstraction emerges.

## 13. Transport reuse versus provider semantics

Transport compatibility and provider compatibility are different.

Moonshot and DeepSeek use the OpenAI Python SDK through Chat Completions-compatible endpoints. They may share reusable transport mechanics such as:

- Client construction
- Base URL handling
- Message serialisation
- Chat Completions request submission
- Basic choice and text extraction
- Common SDK exception translation

They should not automatically share:

- Usage interpretation
- Pricing types
- Pricing lookup
- Cost calculation
- Provider-specific response details
- Provider-specific model validation

A shared OpenAI-compatible transport component should therefore be separate from provider classes.

The provider composes or delegates to the transport while retaining ownership of semantics.

## 14. Error handling

The generic package should expose common exception categories such as:

- Configuration error
- Missing dependency
- Authentication error
- Rate-limit error
- Request error
- Response error
- Unsupported capability
- Pricing unavailable
- Cost calculation unavailable

Providers translate SDK-specific errors into these categories.

Provider-specific error details should remain attached through exception chaining or structured metadata where useful.

An unsupported capability should fail clearly and immediately.

For example:

> Provider `anthropic` does not implement `ImageGenerationCapability`.

## 15. Configuration

Configuration should remain centrally accessible but provider-owned where appropriate.

Common configuration includes:

- Default provider
- Request timeout
- Retry policy
- Tracing options

Provider configuration includes:

- API key
- Default chat model
- Default image model
- Default embedding model
- Provider base URL where configurable
- Provider-specific defaults

The current provider credential environment variables are:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Moonshot: `MOONSHOT_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`

The Google provider must support both `GOOGLE_API_KEY` and `GEMINI_API_KEY`. If both are set, `GOOGLE_API_KEY` should take precedence and `GEMINI_API_KEY` should be treated as the fallback alias.

A provider should be obtainable without credentials so that callers can inspect its capabilities, model catalogue and pricing catalogue. A missing API key should fail only when a credentialled remote operation is attempted.

The provider should resolve its own model defaults rather than relying on unrelated feature modules.

## 16. Tracing and auditability

Tracing should record common request and response information without requiring provider-specific knowledge.

Common trace fields may include:

- Timestamp
- Provider
- Requested model
- Resolved model
- Capability
- Common request parameters
- Common response summary
- Common usage summary
- Generic cost summary

Provider-specific details may be added through a provider-supplied trace payload.

The raw SDK response should be optional because it may be large, contain sensitive data or fail straightforward serialisation.

Cost traces should record enough information to identify:

- Which provider performed the calculation
- Which provider pricing record was used
- Which pricing version or effective date applied
- The resulting generic summary

Detailed provider-specific audit data remains provider-owned.

## 17. Proposed package organisation

A possible package layout is:

- `LLMUtilities/`
  - `providers/`
    - `registry.py`
    - `openai/`
      - provider implementation
      - chat implementation
      - image implementation
      - embedding implementation
      - token-counting implementation
      - chat pricing and costing
      - image pricing and costing
      - provider-specific types
    - `anthropic/`
      - provider implementation
      - chat implementation
      - token-counting implementation
      - pricing and costing
      - provider-specific types
    - `google/`
      - provider implementation
      - chat implementation
      - embedding implementation
      - token-counting implementation
      - pricing and costing
      - provider-specific types
    - `moonshot/`
      - provider implementation
      - chat implementation
      - pricing and costing
      - provider-specific types
    - `deepseek/`
      - provider implementation
      - chat implementation
      - pricing and costing
      - provider-specific types
  - `transports/`
    - OpenAI Responses transport
    - OpenAI Chat Completions transport
    - Anthropic Messages transport
    - Google Generate Content transport
  - `capabilities/`
    - chat protocol
    - image-generation protocol
    - embedding protocol
    - token-counting protocol
  - `types/`
    - common requests
    - common responses
    - common cost summary
    - common exceptions
  - `chat.py`
  - `image.py`
  - `embeddings.py`
  - `tokens.py`
  - `costs.py`
  - `config.py`
  - `tracing/`
  - `parsing/`
    - JSON extraction and repair
    - structured-output prompt construction, generation and retry
  - `compare.py`

The exact files may change. The important boundary is that provider-specific behaviour lives under the provider.

The top-level feature modules remain thin public façades.

## 18. Role of the top-level modules

### 18.1 `chat.py`

Should:

- Build common chat requests
- Resolve providers
- Check `ChatCapability`
- Delegate
- Return common responses or text

Should not:

- Import provider SDKs
- Construct provider requests
- Parse provider responses
- Interpret detailed usage
- Calculate costs

### 18.2 `image.py`

Should:

- Build common image requests
- Resolve providers
- Check `ImageGenerationCapability`
- Delegate
- Return `ImageResponse`

Should not contain OpenAI-specific validation or pricing.

### 18.3 `embeddings.py`

Should:

- Resolve providers
- Check `EmbeddingCapability`
- Delegate
- Provide generic vector convenience functions such as cosine similarity where genuinely provider-independent

Should not instantiate provider SDK clients.

### 18.4 `tokens.py`

Should:

- Resolve providers
- Check `TokenCountingCapability`
- Delegate text or message counting
- Return common count results

Should not contain provider implementations.

### 18.5 `costs.py`

Should:

- Expose common helpers such as `get_cost_summary()`
- Resolve the provider from a response or explicit argument
- Delegate cost calculation
- Define or re-export `CostSummary`
- Contain only genuinely provider-independent formatting or aggregation helpers

Should not contain provider pricing catalogues, provider pricing datatypes or provider billing algorithms.

### 18.6 `parsing/`

`parsing/` gets a model to produce output that validates against a caller-supplied Pydantic schema, for providers with no native structured-output mode.

Should:

- Extract a JSON value from free-form model output (fenced code blocks, a bare object/array span, or the whole response as a last resort)
- Apply conservative, deterministic repair heuristics to common malformed-JSON mistakes (trailing commas, stray quoting, line comments, leading/trailing prose)
- Validate parsed JSON against a caller-supplied Pydantic model or type annotation
- Build a prompt that states the target JSON schema and formatting rules
- Call the generic chat façade (`chat_text()`) to obtain a response
- On a validation failure, build one repair prompt showing the model its own bad output and retry once
- Raise a clear error, including both the original and repair-attempt responses, if the retry also fails

Should not:

- Import provider SDKs or depend on any specific provider
- Construct provider requests or parse provider responses directly
- Know which provider or model produced the text it is parsing
- Attempt more than one repair retry; unbounded retries hide a schema or prompt problem rather than fixing it

### 18.7 `compare.py`

`compare.py` compares two already-generated outputs — it does not generate them. The caller is responsible for obtaining `output_a` and `output_b` (typically via the generic chat façade, possibly from two different providers).

Should:

- Compute provider-independent text statistics (length, word count, exact match, whitespace/case-normalised match)
- Optionally compute embedding-based similarity by delegating to the generic embedding façade (`embed_text()` and `cosine_similarity()`) — never to a provider SDK directly
- Optionally delegate an LLM-as-judge comparison to the generic chat façade (`chat_text()`), accepting either a provider name or an already-resolved provider object
- Return a single deliberately limited `OutputComparison` result covering whichever of the above were requested

Should not:

- Generate the outputs being compared
- Import provider SDKs or contain provider-specific judging logic
- Assume embeddings or a judge model are available — both are opt-in

## 19. Chat execution flow

A typical generic chat call should follow this sequence:

1. The caller invokes the generic chat helper.
2. The helper creates a common chat request.
3. The provider registry resolves the selected provider.
4. The helper confirms that the provider supports chat.
5. The provider resolves the requested or default model.
6. The provider translates the request into its SDK format.
7. The provider executes the SDK call.
8. The provider translates errors.
9. The provider extracts response text.
10. The provider creates the common response.
11. The response retains the raw SDK object.
12. The caller receives text or the common response.
13. A later generic cost request delegates back to the same provider.
14. The provider interprets detailed usage and pricing.
15. The provider returns a common cost summary.

## 20. Image execution flow

A typical generic image call should follow the analogous sequence:

1. The caller invokes the generic image helper.
2. The helper creates a common image request.
3. The provider registry resolves the provider.
4. The helper confirms image-generation support.
5. The provider resolves the model.
6. The provider validates common and provider-specific options.
7. The provider translates the request into the SDK format.
8. The provider executes the SDK call.
9. The provider extracts image artefacts and usage.
10. The provider returns `ImageResponse`.
11. Generic cost helpers delegate image costing back to the provider.
12. Provider-aware callers may request detailed provider-specific image costs.

A future image provider follows the same pattern without requiring central OpenAI-specific branching.

## 21. Provider-aware escape hatches

The design intentionally supports provider-specific access.

Examples of provider-aware operations include:

- Retrieving provider-specific pricing
- Retrieving detailed provider-specific usage
- Retrieving detailed provider-specific costs
- Using provider-only request options
- Accessing provider capability metadata
- Using raw SDK features not yet represented by the generic façade

This is not a failure of the abstraction. It is an explicit two-level API:

1. A simple common interface for common work
2. A concrete provider interface for detailed or specialised work

## 22. Capability discovery

Providers should expose capability information in a machine-readable form.

A caller should be able to determine whether a provider supports:

- Chat
- Image generation
- Embeddings
- Token counting
- Cost calculation for a capability
- Exact usage reporting
- Provider-specific advanced features

Capability discovery avoids trial-and-error calls and avoids generic helpers containing hard-coded provider lists.

## 23. Provider model and pricing discovery

Each provider should expose discovery methods for the models and pricing records that LLMUtilities currently knows how to use for that provider.

### 23.1 Model discovery

A provider does not return model objects. Models are identifiers passed to provider capability methods.

The provider should expose:

* `list_models() -> list[str]`

This returns the model identifiers that the installed version of LLMUtilities currently recognises as available and supported for that provider.

For example:

```python
provider = get_provider("anthropic")

provider.list_models()
# [
#     "claude-fable-5",
#     "claude-opus-4-8",
#     "claude-sonnet-5",
#     "claude-haiku-4-5",
# ]
```

A returned model name means:

> LLMUtilities currently knows how to submit supported requests to this model through this provider.

It does not necessarily mean:

* The model is available to the caller’s account
* The caller has sufficient quota
* The provider API will accept the model in every region
* Every capability supports that model
* The model was discovered dynamically from the provider API

Initially, model discovery should be based on provider-owned static catalogue data bundled with LLMUtilities.

Providers may later add live model discovery, but that should be treated as a separate operation because live results may depend on credentials, account permissions, region and provider API behaviour.

### 23.2 Capability-specific model discovery

A provider may support models for several capabilities.

Model discovery should therefore allow callers to request model identifiers for a particular capability where necessary.

Conceptually:

```python
provider.list_models()
provider.list_chat_models()
provider.list_image_models()
provider.list_embedding_models()
```

Alternatively, the provider may support a capability argument:

```python
provider.list_models(capability="chat")
provider.list_models(capability="image_generation")
provider.list_models(capability="embedding")
```

The exact method shape is deferred.

The important requirement is that callers can determine which model identifiers the provider supports for the operation they intend to perform.

### 23.3 Pricing discovery

Each provider owns its pricing datatypes and pricing catalogue.

The provider should expose:

* A list of all current pricing records known for that provider
* The pricing record applicable to a specific model identifier

Conceptually:

```python
provider = get_provider("anthropic")

pricings = provider.list_pricings()
pricing = provider.get_pricing("claude-sonnet-5")
```

`list_pricings()` returns provider-specific pricing objects.

For Anthropic, it returns a collection of `AnthropicPricing` instances. For OpenAI, it returns OpenAI-specific pricing objects. The generic API does not inspect or interpret those records.

`get_pricing(model)` accepts a model identifier and returns the provider-specific pricing record applicable to that model.

The returned datatype is determined by the concrete provider:

```python
anthropic = get_provider("anthropic")
anthropic_pricing = anthropic.get_pricing("claude-sonnet-5")
# AnthropicPricing

google = get_provider("google")
google_pricing = google.get_pricing("gemini-2.5-pro")
# GooglePricing
```

Pricing discovery does not require the generic layer to define a universal detailed pricing datatype.

### 23.4 Model names and pricing records are related but distinct

A provider’s model list and pricing catalogue should not be treated as the same thing.

A model may be supported even when:

* Pricing is temporarily unavailable
* Pricing depends on account-specific terms
* The provider does not publish public pricing
* A capability is free
* Pricing is inherited from a broader model family
* The provider charges through a mechanism not represented by a simple model-level record

Likewise, a pricing catalogue may retain historical or future pricing records that should not appear in the list of currently available models.

Therefore:

* `list_models()` reports currently supported model identifiers
* `list_pricings()` reports pricing records known to the provider
* `get_pricing(model)` resolves pricing for a model when available
* Missing pricing should raise a pricing-specific error rather than imply that the model itself is unsupported

### 23.5 Effective dates

A provider may retain multiple pricing records for the same model when pricing changes over time.

The provider should own the logic for selecting the applicable record using an effective timestamp.

Conceptually:

```python
provider.get_pricing(
    "claude-sonnet-5",
    effective_at=request_timestamp,
)
```

When no timestamp is supplied, the provider should return the currently effective pricing record.

The generic cost helper should pass the relevant request or response timestamp to the provider where available.

### 23.6 Generic discovery helpers

The package may expose generic convenience helpers that delegate to the provider:

```python
list_models(provider="anthropic")
list_pricings(provider="anthropic")
```

These helpers should perform no model or pricing interpretation.

They should be equivalent to:

```python
provider = get_provider("anthropic")
provider.list_models()
provider.list_pricings()
```

A generic `get_pricing(provider, model)` helper may also exist, but its return type will necessarily be broad because each provider returns its own pricing datatype.

Provider-aware callers should normally obtain the concrete provider first.

### 23.7 Static discovery versus live discovery

The initial design should support static discovery only.

Static discovery means:

* Model identifiers are supplied by provider-owned package data
* Pricing records are supplied by provider-owned package data
* Discovery does not require an API key
* Discovery does not call the provider API
* Results are deterministic for a given installed version of LLMUtilities

Live discovery may be added later as separate provider-specific functionality.

It should not silently replace static discovery because live discovery may be incomplete, credential-dependent, slow or inconsistent across providers.

## 24. Migration from the current implementation

Backward compatibility is not required.

A rational migration sequence is:

1. Introduce the provider registry with `list_providers()` and lazy `get_provider(name)` resolution.
2. Introduce capability protocols.
3. Replace `OpenAIChatModel`, `AnthropicChatModel`, `GoogleChatModel`, `MoonshotChatModel` and `DeepSeekChatModel` with provider-centred implementations.
4. Move provider SDK logic from `tokens.py` into providers.
5. Move provider SDK logic from `embeddings.py` into providers.
6. Fold `OpenAIImageModel` into `OpenAIProvider`.
7. Move OpenAI image validation, usage interpretation, pricing and costing out of the central cost module.
8. Define common response types with requested and resolved model identifiers.
9. Define common `CostSummary`.
10. Move each provider's model catalogues, pricing catalogues and costing into that provider.
11. Add provider-owned model and pricing discovery methods.
12. Reduce top-level modules to delegation façades.
13. Replace central provider switch statements with registry resolution.
14. Rewrite tracing around the new common responses and provider extensions.
15. Rewrite the README.
16. Replace the tests after the new design stabilises.

## 25. Deferred decisions

The following decisions should not be made prematurely:

- Exact fields in each provider pricing datatype
- Whether provider pricing catalogues share a common envelope
- Exact detailed usage types
- Exact detailed cost types
- Whether providers use composition, mixins or internal service objects
- Exact internal provider registry implementation beyond `list_providers()` and lazy `get_provider(name)`
- Exact method names and filtering options for provider model discovery
- How much model metadata belongs in configuration versus provider catalogues
- Whether common usage summaries should include caching fields
- Whether cost summaries should be attached to responses or calculated lazily
- Whether generic request objects include a provider-options mapping
- Whether image editing should share the text-to-image request type
- Whether price catalogues are bundled files, remote resources or both

These should be resolved from concrete provider requirements rather than abstract symmetry.

## 26. Summary

LLMUtilities should mostly focus on a simple abstract chat interface.

The package is a common façade over provider wrappers, not a universal implementation of LLM behaviour.

Providers own:

- SDK interaction
- Request translation
- Response parsing
- Usage interpretation
- Token counting
- Provider-specific validation
- Provider-specific model datatypes and model catalogues
- Provider-specific pricing datatypes and pricing catalogues
- Provider-specific cost calculations
- Detailed provider-specific APIs

Generic helpers own:

- Provider resolution
- Capability checks
- Common request construction
- Delegation
- Deliberately limited common responses
- Deliberately limited common cost summaries
- Common exception categories
- Common tracing hooks

Image generation follows the same pattern as chat. Embeddings and token counting follow the same pattern as well.

Structured-output parsing and output comparison sit one layer above all of this: they consume the generic chat and embedding façades exactly as any other caller would, and therefore need no provider-specific code, no entry in the capability matrix and no change when a new provider is added.

This architecture accepts the real diversity of provider APIs while still giving ordinary callers a simple and consistent interface.
