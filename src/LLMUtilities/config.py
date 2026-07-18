import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return default

    return value


def _get_env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = _get_env_str(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {value!r}") from exc


def _get_env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = _get_env_str(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got: {value!r}") from exc


@dataclass(frozen=True)
class ProviderDefaults:
    chat_model: Optional[str] = None
    embedding_model: Optional[str] = None
    image_model: Optional[str] = None
    image_size: Optional[str] = None
    image_quality: Optional[str] = None
    image_background: Optional[str] = None
    image_format: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    google_api_key: Optional[str]
    moonshot_api_key: Optional[str]
    deepseek_api_key: Optional[str]

    openai: ProviderDefaults
    anthropic: ProviderDefaults
    google: ProviderDefaults
    moonshot: ProviderDefaults
    deepseek: ProviderDefaults

    default_provider: str
    request_timeout_seconds: float
    max_retries: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=_get_env_str("OPENAI_API_KEY"),
        anthropic_api_key=_get_env_str("ANTHROPIC_API_KEY"),
        google_api_key=_get_env_str("GOOGLE_API_KEY") or _get_env_str("GEMINI_API_KEY"),
        moonshot_api_key=_get_env_str("MOONSHOT_API_KEY"),
        deepseek_api_key=_get_env_str("DEEPSEEK_API_KEY"),
        openai=ProviderDefaults(
            chat_model=_get_env_str("OPENAI_CHAT_MODEL", "gpt-5-mini"),
            embedding_model=_get_env_str(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            image_model=_get_env_str("OPENAI_IMAGE_MODEL", "gpt-image-1.5"),
            image_size=_get_env_str("OPENAI_IMAGE_SIZE"),
            image_quality=_get_env_str("OPENAI_IMAGE_QUALITY"),
            image_background=_get_env_str("OPENAI_IMAGE_BACKGROUND"),
            image_format=_get_env_str("OPENAI_IMAGE_FORMAT"),
            temperature=_get_env_float("OPENAI_TEMPERATURE"),
            max_output_tokens=_get_env_int("OPENAI_MAX_OUTPUT_TOKENS"),
        ),
        anthropic=ProviderDefaults(
            chat_model=_get_env_str("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6"),
            embedding_model=_get_env_str("ANTHROPIC_EMBEDDING_MODEL"),
            image_model=_get_env_str("ANTHROPIC_IMAGE_MODEL"),
            image_size=_get_env_str("ANTHROPIC_IMAGE_SIZE"),
            image_quality=_get_env_str("ANTHROPIC_IMAGE_QUALITY"),
            image_background=_get_env_str("ANTHROPIC_IMAGE_BACKGROUND"),
            image_format=_get_env_str("ANTHROPIC_IMAGE_FORMAT"),
            temperature=_get_env_float("ANTHROPIC_TEMPERATURE"),
            max_output_tokens=_get_env_int("ANTHROPIC_MAX_OUTPUT_TOKENS"),
        ),
        google=ProviderDefaults(
            chat_model=_get_env_str("GOOGLE_CHAT_MODEL", "gemini-2.5-flash"),
            embedding_model=_get_env_str(
                "GOOGLE_EMBEDDING_MODEL", "text-embedding-004"
            ),
            image_model=_get_env_str("GOOGLE_IMAGE_MODEL"),
            image_size=_get_env_str("GOOGLE_IMAGE_SIZE"),
            image_quality=_get_env_str("GOOGLE_IMAGE_QUALITY"),
            image_background=_get_env_str("GOOGLE_IMAGE_BACKGROUND"),
            image_format=_get_env_str("GOOGLE_IMAGE_FORMAT"),
            temperature=_get_env_float("GOOGLE_TEMPERATURE"),
            max_output_tokens=_get_env_int("GOOGLE_MAX_OUTPUT_TOKENS"),
        ),
        moonshot=ProviderDefaults(
            chat_model=_get_env_str("MOONSHOT_CHAT_MODEL", "kimi-k2.6"),
            temperature=_get_env_float("MOONSHOT_TEMPERATURE"),
            max_output_tokens=_get_env_int("MOONSHOT_MAX_OUTPUT_TOKENS"),
        ),
        deepseek=ProviderDefaults(
            chat_model=_get_env_str("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash"),
            temperature=_get_env_float("DEEPSEEK_TEMPERATURE"),
            max_output_tokens=_get_env_int("DEEPSEEK_MAX_OUTPUT_TOKENS"),
        ),
        default_provider=_get_env_str("LLMUTILITIES_DEFAULT_PROVIDER", "openai")
        or "openai",
        request_timeout_seconds=_get_env_float(
            "LLMUTILITIES_REQUEST_TIMEOUT_SECONDS", 120.0
        )
        or 120.0,
        max_retries=_get_env_int("LLMUTILITIES_MAX_RETRIES", 3) or 3,
    )


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


settings = get_settings()
