from __future__ import annotations

import importlib
from typing import Any, Final

from ..exceptions import ConfigurationError

_PROVIDERS: Final[dict[str, tuple[str, str]]] = {
    "openai": ("LLMUtilities.providers.openai", "OpenAIProvider"),
    "anthropic": ("LLMUtilities.providers.anthropic", "AnthropicProvider"),
    "google": ("LLMUtilities.providers.google", "GoogleProvider"),
    "moonshot": ("LLMUtilities.providers.moonshot", "MoonshotProvider"),
    "deepseek": ("LLMUtilities.providers.deepseek", "DeepSeekProvider"),
}


def list_providers() -> list[str]:
    """
    Names of all providers that this installed version of LLMUtilities knows
    how to support.

    This does not import any provider module, inspect installed SDKs or check
    credentials — it only answers "which providers does the library know about".
    """
    return sorted(_PROVIDERS)


def get_provider(name: str) -> Any:
    """
    Lazily import and construct the requested provider.

    Only the requested provider's module is imported — asking for 'openai'
    never imports the anthropic/google/moonshot/deepseek modules. Raises
    ``MissingDependencyError`` if that provider's SDK is not installed, and
    ``ConfigurationError`` if the name is not a known provider. Construction
    never requires an API key; credentials are checked lazily when a
    capability method that needs them is actually called.
    """
    provider_name = name.strip().lower()

    if provider_name not in _PROVIDERS:
        supported = ", ".join(list_providers())
        raise ConfigurationError(
            f"Unsupported provider: {provider_name!r}. Expected one of: {supported}."
        )

    module_path, class_name = _PROVIDERS[provider_name]
    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name)
    return provider_cls()
