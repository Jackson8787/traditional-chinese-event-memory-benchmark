from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


JsonObject = dict[str, Any]
PostJson = Callable[[str, str, JsonObject, int], JsonObject]


@dataclass(frozen=True)
class ApiError(Exception):
    status: int | str
    message: str
    url: str

    def __str__(self) -> str:
        return f"API request failed status={self.status} url={self.url}: {self.message}"


def post_json(url: str, api_key: str, payload: JsonObject, timeout_seconds: int) -> JsonObject:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, _preview(body), url) from exc
    except urllib.error.URLError as exc:
        raise ApiError("network", str(exc.reason), url) from exc


def post_json_with_retries(
    url: str,
    api_key: str,
    payload: JsonObject,
    timeout_seconds: int,
    max_retries: int,
    transport: PostJson = post_json,
) -> JsonObject:
    last_error: ApiError | None = None
    for attempt in range(max_retries + 1):
        try:
            return transport(url, api_key, payload, timeout_seconds)
        except ApiError as exc:
            last_error = exc
            if not _retryable(exc.status) or attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def openai_compatible_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    suffix = suffix.lstrip("/")
    if base.endswith("/openai/v1"):
        return f"{base}/{suffix}"
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/openai/v1/{suffix}"


def _retryable(status: int | str) -> bool:
    return status in {"network", 408, 409, 429, 500, 502, 503, 504}


def _preview(text: str, limit: int = 500) -> str:
    return text[:limit].replace("\n", " ")
