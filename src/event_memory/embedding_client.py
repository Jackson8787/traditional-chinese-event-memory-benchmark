from __future__ import annotations

from dataclasses import dataclass

from .config import EndpointConfig
from .http_client import JsonObject, PostJson, openai_compatible_url, post_json, post_json_with_retries


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float]
    model: str
    raw: JsonObject


class EmbeddingClient:
    def __init__(self, config: EndpointConfig, transport: PostJson = post_json) -> None:
        self.config = config
        self.transport = transport
        self.url = openai_compatible_url(config.base_url, "embeddings")

    def embed(self, text: str, *, model: str | None = None) -> EmbeddingResult:
        payload: JsonObject = {
            "model": model or self.config.model,
            "input": text,
        }
        raw = post_json_with_retries(
            self.url,
            self.config.api_key,
            payload,
            self.config.timeout_seconds,
            self.config.max_retries,
            self.transport,
        )
        embedding = _extract_embedding(raw)
        return EmbeddingResult(
            embedding=embedding,
            model=str(raw.get("model") or payload["model"]),
            raw=raw,
        )


def _extract_embedding(raw: JsonObject) -> list[float]:
    data = raw.get("data")
    if not isinstance(data, list) or not data:
        return []
    first = data[0]
    if not isinstance(first, dict):
        return []
    embedding = first.get("embedding")
    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]
