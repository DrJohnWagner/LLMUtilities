from __future__ import annotations

from ..config import settings
from .openai_chat_completions import BaseOpenAIChatCompletionsModel


class DeepSeekChatModel(BaseOpenAIChatCompletionsModel):
    provider_name = "deepseek"
    default_model = settings.deepseek.chat_model
    api_key = settings.deepseek_api_key
    openai_base_url = "https://api.deepseek.com"