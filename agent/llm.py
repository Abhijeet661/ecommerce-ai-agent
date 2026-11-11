"""Small client for any OpenAI-compatible Chat Completions API.

Works with Google Gemini (free API key from Google AI Studio), Groq, OpenAI
and other providers that expose the same /chat/completions endpoint.
Settings come from environment variables (see .env.example).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, exactly as the model returned it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """Sends chat messages (and optional tool definitions) to the model."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("LLM_API_KEY is missing. Copy .env.example to .env and add your key.")

    def chat(self, messages, tools=None, temperature=0.2, retries=3) -> LLMResponse:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, retries + 1):
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            # Retry on rate limits and temporary server errors (free tiers hit 429 often)
            if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
            break

        message = resp.json()["choices"][0]["message"]
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)


def parse_json_reply(text: str) -> dict:
    """Pull a JSON object out of a model reply, even if it is wrapped in ```json fences."""
    if not text:
        raise ValueError("Empty reply from model")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model reply")
    return json.loads(text[start : end + 1])
