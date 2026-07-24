from __future__ import annotations

import httpx
import openai


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.example.com/v1/x")
    return httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})


def make_auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_response(401), body=None)


def make_rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("too many requests", response=_response(429), body=None)


def make_status_error() -> openai.APIStatusError:
    return openai.APIStatusError("server error", response=_response(500), body=None)


def make_connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://api.example.com/v1/x")
    return openai.APIConnectionError(message="connection failed", request=request)
