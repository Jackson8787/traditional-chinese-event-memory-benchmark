from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("api_config.env")


@dataclass(frozen=True)
class EndpointConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    max_retries: int = 3

    def redacted(self) -> dict[str, str | int]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": f"<redacted length={len(self.api_key)}>",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


@dataclass(frozen=True)
class AppConfig:
    llm: EndpointConfig
    embedding: EndpointConfig
    audit_model: str


def load_env_file(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Config file not found: {env_path}")

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _clean_value(value)
    return values


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    file_values = load_env_file(path)
    values = {**file_values, **_env_overrides(file_values)}
    timeout = int(values.get("REQUEST_TIMEOUT_SECONDS", "60") or "60")
    retries = int(values.get("MAX_RETRIES", "3") or "3")

    llm = EndpointConfig(
        provider=_required(values, "LLM_PROVIDER"),
        base_url=_required(values, "LLM_BASE_URL").rstrip("/"),
        api_key=_required_secret(values, "LLM_API_KEY"),
        model=_required(values, "LLM_MODEL"),
        timeout_seconds=timeout,
        max_retries=retries,
    )
    embedding = EndpointConfig(
        provider=values.get("EMBEDDING_PROVIDER") or llm.provider,
        base_url=(values.get("EMBEDDING_BASE_URL") or llm.base_url).rstrip("/"),
        api_key=_required_secret(values, "EMBEDDING_API_KEY", fallback=llm.api_key),
        model=_required(values, "EMBEDDING_MODEL"),
        timeout_seconds=timeout,
        max_retries=retries,
    )
    return AppConfig(
        llm=llm,
        embedding=embedding,
        audit_model=values.get("LLM_AUDIT_MODEL", llm.model),
    )


def _env_overrides(keys: dict[str, str]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if key in os.environ}


def _clean_value(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _required(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing required config value: {key}")
    return value


def _required_secret(values: dict[str, str], key: str, fallback: str | None = None) -> str:
    value = values.get(key, fallback or "").strip()
    if not value or value == "replace-with-your-api-key":
        raise ValueError(f"Missing required API key: {key}")
    return value
