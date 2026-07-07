import tempfile
import unittest
from pathlib import Path

from event_memory.config import load_config
from event_memory.embedding_client import EmbeddingClient
from event_memory.llm_client import ChatMessage, LLMClient


class ApiClientTest(unittest.TestCase):
    def test_load_config_and_redact_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "api_config.env"
            path.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai",
                        "LLM_BASE_URL=https://example.test/project",
                        "LLM_API_KEY=secret-llm",
                        "LLM_MODEL=gpt-test",
                        "EMBEDDING_PROVIDER=openai",
                        "EMBEDDING_BASE_URL=https://example.test",
                        "EMBEDDING_API_KEY=secret-embedding",
                        "EMBEDDING_MODEL=embed-test",
                        "REQUEST_TIMEOUT_SECONDS=12",
                        "MAX_RETRIES=0",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.llm.model, "gpt-test")
        self.assertEqual(config.embedding.model, "embed-test")
        self.assertEqual(config.llm.timeout_seconds, 12)
        self.assertNotIn("secret-llm", str(config.llm.redacted()))

    def test_llm_client_uses_openai_compatible_chat_endpoint(self) -> None:
        calls = []

        def fake_transport(url, api_key, payload, timeout_seconds):
            calls.append((url, api_key, payload, timeout_seconds))
            return {"model": payload["model"], "choices": [{"message": {"content": "OK"}}]}

        config = _endpoint_config("https://example.test/api/projects/proj-default", "gpt-test")
        result = LLMClient(config, transport=fake_transport).chat(
            [ChatMessage(role="user", content="hello")],
            max_completion_tokens=8,
        )

        self.assertEqual(result.content, "OK")
        self.assertEqual(calls[0][0], "https://example.test/api/projects/proj-default/openai/v1/chat/completions")
        self.assertEqual(calls[0][2]["messages"][0]["content"], "hello")
        self.assertEqual(calls[0][2]["max_completion_tokens"], 8)

    def test_embedding_client_uses_openai_compatible_embedding_endpoint(self) -> None:
        calls = []

        def fake_transport(url, api_key, payload, timeout_seconds):
            calls.append((url, api_key, payload, timeout_seconds))
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}], "model": payload["model"]}

        config = _endpoint_config("https://example.test/openai/v1", "embed-test")
        result = EmbeddingClient(config, transport=fake_transport).embed("繁中測試")

        self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(calls[0][0], "https://example.test/openai/v1/embeddings")
        self.assertEqual(calls[0][2]["input"], "繁中測試")


def _endpoint_config(base_url: str, model: str):
    from event_memory.config import EndpointConfig

    return EndpointConfig(
        provider="openai",
        base_url=base_url,
        api_key="secret",
        model=model,
        timeout_seconds=10,
        max_retries=0,
    )


if __name__ == "__main__":
    unittest.main()
