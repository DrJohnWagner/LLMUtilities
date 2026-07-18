
# TODO

## P0 — Correctness

### Normalise token accounting

* [ ] Redesign `ChatUsage` so token fields have explicit, provider-independent semantics:

  * [ ] `total_input_tokens`
  * [ ] `uncached_input_tokens`
  * [ ] `cached_read_input_tokens`
  * [ ] `cache_creation_input_tokens`
  * [ ] `output_tokens`
  * [ ] `total_tokens`
* [ ] Update the OpenAI Responses adapter to extract cached-input token details.
* [ ] Update the OpenAI-compatible Chat Completions adapter so cached tokens are not billed twice.
* [ ] Verify Anthropic cache-read and cache-creation accounting against the normalised schema.
* [ ] Verify Google cached-token semantics before exposing cached usage.
* [ ] Rewrite cost calculations to consume the normalised fields.
* [ ] Add provider-specific tests covering cached, uncached and cache-creation tokens.
* [ ] Add regression tests proving cached tokens cannot be charged at both full and discounted rates.

### Fix configuration reloads

* [ ] Stop capturing `settings` values in provider class attributes at import time.
* [ ] Resolve default models, API keys, timeouts and retry counts when provider instances are created.
* [ ] Make `reload_settings()` update all subsequent provider instances.
* [ ] Add tests that:

  * [ ] change environment variables
  * [ ] call `reload_settings()`
  * [ ] instantiate a new provider
  * [ ] confirm the new values are used
* [ ] Document whether existing provider instances retain their original configuration.

### Reject unsupported content

* [ ] Do not silently discard `ImageContentPart` objects.
* [ ] Raise a clear unsupported-content error until provider-specific multimodal translation is implemented.
* [ ] Add tests for text-only, image-only and mixed-content messages.

## P1 — Package contract

### Correct package metadata

* [ ] Replace `version = "0.0.0"` with an intentional package version.
* [ ] Replace `requires-python = "==3.10.13"` with a supported Python range.
* [ ] Decide which Python versions are officially supported.
* [ ] Move provider SDKs into optional dependency groups:

  * [ ] `LLMUtilities[openai]`
  * [ ] `LLMUtilities[anthropic]`
  * [ ] `LLMUtilities[google]`
  * [ ] `LLMUtilities[all]`
* [ ] Keep only genuinely universal runtime dependencies in the base package.
* [ ] Move exact environment pins into a lock file or development requirements file.
* [ ] Make the README installation instructions match `pyproject.toml`.
* [ ] Verify that the base package imports without any provider SDK installed.

### Apply or remove configuration options

* [ ] Apply provider temperature defaults consistently.
* [ ] Apply provider maximum-output-token defaults consistently.
* [ ] Decide whether image defaults belong in configuration or request construction.
* [ ] Remove configuration fields that are not supported.
* [ ] Document precedence:

  1. request value
  2. provider instance value
  3. environment default
  4. library fallback

### Remove ignored parameters

* [ ] Either support `ImageRequest.seed` or remove it.
* [ ] Audit all public request fields for silently ignored values.
* [ ] Raise explicit errors for parameters unsupported by the selected provider.

## P1 — Type safety and interfaces

* [ ] Define a `ChatProvider` protocol.
* [ ] Define an `ImageProvider` protocol.
* [ ] Add return types to provider factories.
* [ ] Type all public `provider` parameters.
* [ ] Resolve the mismatch between `chat_usage() -> ChatUsage` and optional `ChatResponse.usage`.
* [ ] Decide whether all adapters must always return an empty `ChatUsage` object rather than `None`.
* [ ] Run strict mypy over the complete package.
* [ ] Add strict mypy to CI.
* [ ] Remove unused imports and stale commented examples from implementation modules.

## P1 — Exception consistency

* [ ] Apply package exception normalisation to embeddings.
* [ ] Apply package exception normalisation to token-counting API calls.
* [ ] Apply package exception normalisation to structured-output retries.
* [ ] Distinguish:

  * [ ] authentication failures
  * [ ] rate limits
  * [ ] network failures
  * [ ] provider API failures
  * [ ] malformed responses
* [ ] Ensure raw SDK exceptions do not escape public provider-normalised APIs.
* [ ] Add contract tests for exception mapping across providers.

## P2 — API behaviour

### Make network activity explicit

* [ ] Change `compare_outputs(..., use_embeddings=True)` so embeddings are opt-in.
* [ ] Document which functions:

  * [ ] make network requests
  * [ ] may incur provider charges
  * [ ] require API credentials
* [ ] Consider separate names for local comparison and provider-assisted comparison.

### Clarify token counting

* [ ] Document OpenAI local token counting as an estimate.
* [ ] Document that image parts are currently excluded from token estimates.
* [ ] Review OpenAI message-framing overhead.
* [ ] Review whether Google system and conversation tokens can be counted in one provider request.
* [ ] Add tests distinguishing exact provider counts from local estimates.
* [ ] Consider returning a result object containing:

  * [ ] token count
  * [ ] provider
  * [ ] model
  * [ ] method
  * [ ] whether the value is exact or estimated

### Improve structured output

* [ ] Prefer provider-native structured output where supported.
* [ ] Retain prompt-and-repair fallback for unsupported providers.
* [ ] Catch narrower exception types during parsing and repair.
* [ ] Preserve the first parsing exception as useful diagnostic context.
* [ ] Add configurable repair-attempt limits.
* [ ] Add tests for nested schemas, arrays, unions and validation failures.
* [ ] Remove large commented examples from the module and place them in documentation.

## P2 — Pricing

* [ ] Define and document the pricing catalogue schema.
* [ ] Validate catalogue files at load time.
* [ ] Detect duplicate canonical model IDs.
* [ ] Validate aliases and reject alias cycles.
* [ ] Validate non-negative rates.
* [ ] Validate effective-date formats.
* [ ] Decide how expired pricing entries should behave.
* [ ] Add a command or script for refreshing pricing data.
* [ ] Add tests that verify catalogue provenance fields.
* [ ] Add tests for long-context pricing boundaries.
* [ ] Add tests for batch pricing fallbacks.
* [ ] Add tests for unknown and expired models.
* [ ] Document that pricing data can become stale despite verification timestamps.

## P2 — Multimodal support

* [ ] Define a provider-neutral image-source schema instead of using an unrestricted `dict`.
* [ ] Support URL images.
* [ ] Support base64 images.
* [ ] Support provider-specific MIME requirements.
* [ ] Translate multimodal messages for OpenAI.
* [ ] Translate multimodal messages for Anthropic.
* [ ] Translate multimodal messages for Google.
* [ ] Add explicit provider capability checks.
* [ ] Add multimodal contract tests.

## P2 — Tracing and privacy

* [ ] Document that traces may contain prompts, responses and sensitive data.
* [ ] Add a redaction callback.
* [ ] Add configurable field exclusion.
* [ ] Add an option to record hashes or metadata without recording content.
* [ ] Ensure API keys and authentication headers can never enter traces.
* [ ] Add tests for truncation, redaction and non-serialisable raw responses.
* [ ] Consider file locking or another strategy for concurrent JSONL writes.

## P2 — Testing

* [ ] Split `tests/test_package.py` into focused modules:

  * [ ] `test_chat.py`
  * [ ] `test_types.py`
  * [ ] `test_openai.py`
  * [ ] `test_anthropic.py`
  * [ ] `test_google.py`
  * [ ] `test_openai_compatible.py`
  * [ ] `test_embeddings.py`
  * [ ] `test_tokens.py`
  * [ ] `test_costs.py`
  * [ ] `test_image.py`
  * [ ] `test_parsing.py`
  * [ ] `test_tracing.py`
* [ ] Add shared provider contract tests.
* [ ] Add tests for empty system-only requests.
* [ ] Add tests for conflicting `provider` and `provider_name` arguments.
* [ ] Add tests for whitespace-only parameters.
* [ ] Add tests for malformed SDK response objects.
* [ ] Add tests for environment reload behaviour.
* [ ] Add package installation tests with no optional SDKs.
* [ ] Add package installation tests for each provider extra.
* [ ] Measure and publish test coverage.

## P2 — Continuous integration

* [ ] Add GitHub Actions.
* [ ] Run pytest on every supported Python version.
* [ ] Run strict mypy.
* [ ] Run Ruff checks.
* [ ] Run package builds.
* [ ] Verify source distributions and wheels with Twine.
* [ ] Test installation from the built wheel.
* [ ] Add dependency caching.
* [ ] Add a scheduled pricing-catalogue validation job.

## P3 — Public API and documentation

* [ ] Define the supported public API explicitly.
* [ ] Decide which functions should be exported from `LLMUtilities.__init__`.
* [ ] Export related helpers consistently or keep them deliberately internal.
* [ ] Add API documentation for every public function.
* [ ] Add a provider capability matrix generated from code rather than maintained manually.
* [ ] Add examples for:

  * [ ] custom provider instances
  * [ ] cached-token costing
  * [ ] batch pricing
  * [ ] structured output
  * [ ] tracing with redaction
  * [ ] missing optional dependencies
* [ ] Add a changelog.
* [ ] Add contribution guidelines.
* [ ] Add release instructions.
* [ ] Add a package maturity warning until the API stabilises.

## P3 — Future features

* [ ] Async chat interfaces.
* [ ] Streaming responses.
* [ ] Tool-call normalisation.
* [ ] Stop-sequence support.
* [ ] Provider-native reasoning controls.
* [ ] Native structured-output support.
* [ ] Usage and cost aggregation across multiple calls.
* [ ] Retry policies independent of provider SDK behaviour.
* [ ] Provider capability discovery.
* [ ] Request and response middleware.
* [ ] Pluggable pricing catalogues.
* [ ] Automatic tracing integration.

## Release readiness

The package should not be treated as stable until all of the following are complete:

* [ ] Cached-token accounting has one documented cross-provider meaning.
* [ ] Configuration reload behaviour is correct and tested.
* [ ] Optional dependencies match the installation documentation.
* [ ] Unsupported multimodal data is never silently discarded.
* [ ] Public APIs pass strict mypy.
* [ ] CI tests all supported Python versions.
* [ ] Package versioning and release documentation are in place.
