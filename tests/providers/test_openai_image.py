from __future__ import annotations

import types

import LLMUtilities.providers.openai.image as openai_image
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ImageRequest


class _FakeImageItem:
    def __init__(self, b64_json):
        self.b64_json = b64_json
        self.url = None
        self.revised_prompt = "a revised prompt"


class _FakeImageUsage:
    def __init__(self):
        self.input_tokens_details = types.SimpleNamespace(text_tokens=20, image_tokens=0)
        self.output_tokens = 1000


class _FakeImageResponse:
    def __init__(self):
        self.data = [_FakeImageItem(b64_json="ZmFrZS1pbWFnZS1ieXRlcw==")]
        self.usage = _FakeImageUsage()


def _install_fake_client(monkeypatch, generate_fn):
    class _FakeClient:
        def __init__(self, **kwargs):
            self.images = types.SimpleNamespace(generate=generate_fn)

    monkeypatch.setattr(openai_image, "_OpenAIClient", _FakeClient)


def test_generate_image_extracts_artifacts_and_usage(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeImageResponse())

    provider = get_provider("openai")
    request = ImageRequest(prompt="a cat wearing a hat", model="gpt-image-1.5")
    response = provider.generate_image(request)

    assert response.provider == "openai"
    assert response.resolved_model == "gpt-image-1.5"
    assert len(response.artifacts) == 1
    assert response.artifacts[0].b64_data == "ZmFrZS1pbWFnZS1ieXRlcw=="
    assert response.artifacts[0].revised_prompt == "a revised prompt"


def test_image_cost_summary_uses_reference_and_token_rates(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeImageResponse())

    provider = get_provider("openai")
    request = ImageRequest(prompt="a cat wearing a hat", model="gpt-image-1.5")
    response = provider.generate_image(request)

    summary = provider.get_image_cost_summary(response)
    assert summary.provider == "openai"
    assert summary.total_cost > 0
