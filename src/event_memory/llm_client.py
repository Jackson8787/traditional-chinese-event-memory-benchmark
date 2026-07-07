from __future__ import annotations

from dataclasses import dataclass

from .config import EndpointConfig
from .http_client import JsonObject, PostJson, openai_compatible_url, post_json, post_json_with_retries


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    raw: JsonObject


class LLMClient:
    def __init__(self, config: EndpointConfig, transport: PostJson = post_json) -> None:
        self.config = config
        self.transport = transport
        self.url = openai_compatible_url(config.base_url, "chat/completions")

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        max_completion_tokens: int = 512,
    ) -> ChatResult:
        payload: JsonObject = {
            "model": model or self.config.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "max_completion_tokens": max_completion_tokens,
        }
        raw = post_json_with_retries(
            self.url,
            self.config.api_key,
            payload,
            self.config.timeout_seconds,
            self.config.max_retries,
            self.transport,
        )
        return ChatResult(
            content=_extract_content(raw),
            model=str(raw.get("model") or payload["model"]),
            raw=raw,
        )


def _extract_content(raw: JsonObject) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""
