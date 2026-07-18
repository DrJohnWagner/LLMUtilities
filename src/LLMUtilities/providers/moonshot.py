from __future__ import annotations

from ..config import settings
from .openai_chat_completions import BaseOpenAIChatCompletionsModel


class MoonshotChatModel(BaseOpenAIChatCompletionsModel):
    provider_name = "moonshot"
    default_model = settings.moonshot.chat_model
    api_key = settings.moonshot_api_key
    openai_base_url = "https://api.moonshot.ai/v1"