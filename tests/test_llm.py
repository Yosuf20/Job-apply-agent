"""Tests for LLM client, provider adapters, error handling, and structured output."""

import ssl
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from jobagent.config import Settings
from jobagent.llm.client import (
    AnthropicProvider,
    GeminiProvider,
    LLMAuthError,
    LLMClient,
    LLMNonTransientError,
    LLMRequestError,
    LLMTransientError,
    OpenAIProvider,
    StructuredOutputError,
    get_client,
    strip_code_fences,
)
from tests.conftest import FakeProvider


class SampleModel(BaseModel):
    """Test response model."""

    role: str
    confidence: float = Field(ge=0.0, le=1.0)


# -----------------------------------------------------------------------------
# Structured output & fence stripping tests
# -----------------------------------------------------------------------------


def test_strip_code_fences() -> None:
    """Fences with json, empty fences, and plain json strings are handled correctly."""
    assert strip_code_fences('```json\n{"role": "lead"}\n```') == '{"role": "lead"}'
    assert strip_code_fences('```\n{"role": "lead"}\n```') == '{"role": "lead"}'
    assert strip_code_fences('{"role": "lead"}') == '{"role": "lead"}'


def test_complete_structured_parses_valid_json() -> None:
    """complete_structured parses raw valid JSON into the target Pydantic model."""
    provider = FakeProvider(responses=['{"role": "ai_engineer", "confidence": 0.9}'])
    client = LLMClient(provider=provider, model_small="test-small")

    result = client.complete_structured(
        system="You are an analyst.",
        prompt="Analyze role.",
        response_model=SampleModel,
        tier="small",
    )

    assert result.role == "ai_engineer"
    assert result.confidence == 0.9
    assert len(provider.calls) == 1


def test_complete_structured_parses_fenced_json() -> None:
    """complete_structured strips ```json code fences."""
    fenced = '```json\n{"role": "fullstack", "confidence": 0.85}\n```'
    provider = FakeProvider(responses=[fenced])
    client = LLMClient(provider=provider, model_small="test-small")

    result = client.complete_structured(
        system="",
        prompt="Extract.",
        response_model=SampleModel,
        tier="small",
    )

    assert result.role == "fullstack"
    assert result.confidence == 0.85


def test_complete_structured_retries_once_and_succeeds() -> None:
    """When the first response is invalid JSON, client retries once with feedback and succeeds."""
    provider = FakeProvider(
        responses=[
            '{"role": "missing_confidence"}',  # Fails validation
            '{"role": "fixed", "confidence": 0.99}',  # Succeeds
        ]
    )
    client = LLMClient(provider=provider, model_small="test-small")

    result = client.complete_structured(
        system="Extract info.",
        prompt="Job details",
        response_model=SampleModel,
        tier="small",
    )

    assert result.role == "fixed"
    assert result.confidence == 0.99
    assert len(provider.calls) == 2
    # Verify feedback was included in the retry prompt
    assert "Your previous response failed validation" in provider.calls[1]["prompt"]


def test_complete_structured_raises_after_second_failure() -> None:
    """When both initial and retry responses fail validation, StructuredOutputError is raised."""
    provider = FakeProvider(
        responses=[
            "not a json at all",
            '{"role": "still_no_confidence"}',
        ]
    )
    client = LLMClient(provider=provider, model_small="test-small")

    with pytest.raises(
        StructuredOutputError, match="Failed to produce valid SampleModel"
    ):
        client.complete_structured(
            system="",
            prompt="Extract.",
            response_model=SampleModel,
            tier="small",
        )

    assert len(provider.calls) == 2


# -----------------------------------------------------------------------------
# Retry & Backoff tests
# -----------------------------------------------------------------------------


def test_transient_error_retries_and_succeeds() -> None:
    """Transient errors are retried up to 3 times with exponential backoff and sleep injection."""
    slept_durations: list[float] = []

    provider = FakeProvider(
        responses=[
            LLMTransientError("Rate limit 429"),
            LLMTransientError("Gateway timeout 504"),
            '{"role": "recovered", "confidence": 0.7}',
        ]
    )
    client = LLMClient(
        provider=provider,
        model_small="test-small",
        sleep_fn=slept_durations.append,
    )

    resp = client.complete(system="", prompt="Hello", tier="small")
    assert "recovered" in resp.text
    assert len(provider.calls) == 3
    assert slept_durations == [0.5, 1.0]


def test_transient_error_exceeds_max_attempts_raises() -> None:
    """Persistent transient errors raise after 3 attempts."""
    slept_durations: list[float] = []

    provider = FakeProvider(
        responses=[
            LLMTransientError("Error 1"),
            LLMTransientError("Error 2"),
            LLMTransientError("Error 3"),
        ]
    )
    client = LLMClient(
        provider=provider,
        model_small="test-small",
        sleep_fn=slept_durations.append,
    )

    with pytest.raises(LLMTransientError, match="Error 3"):
        client.complete(system="", prompt="Hello", tier="small")

    assert len(provider.calls) == 3
    assert slept_durations == [0.5, 1.0]


def test_llm_client_retries_transient_network_errors_3_times() -> None:
    """LLMClient retries transient network errors (ConnectTimeout / ReadTimeout) 3 times."""
    slept_durations: list[float] = []

    provider = FakeProvider(
        responses=[
            LLMTransientError("ConnectTimeout"),
            LLMTransientError("ReadTimeout"),
            '{"role": "ok", "confidence": 0.8}',
        ]
    )
    client = LLMClient(
        provider=provider,
        model_small="test-small",
        sleep_fn=slept_durations.append,
    )

    resp = client.complete(system="", prompt="Hi", tier="small")
    assert "ok" in resp.text
    assert len(provider.calls) == 3
    assert slept_durations == [0.5, 1.0]


def test_non_transient_error_does_not_retry() -> None:
    """Auth errors and bad request errors must NOT be retried."""
    slept_durations: list[float] = []

    provider = FakeProvider(responses=[LLMAuthError("Invalid API key 401")])
    client = LLMClient(
        provider=provider,
        model_small="test-small",
        sleep_fn=slept_durations.append,
    )

    with pytest.raises(LLMAuthError, match="Invalid API key"):
        client.complete(system="", prompt="Hello", tier="small")

    assert len(provider.calls) == 1
    assert slept_durations == []


def test_temperature_optional_and_forwarded() -> None:
    """Temperature is only forwarded when explicitly provided."""
    provider = FakeProvider()
    client = LLMClient(provider=provider, model_small="test-small")

    client.complete(system="", prompt="Hi", tier="small")
    assert provider.calls[0]["temperature"] is None

    client.complete(system="", prompt="Hi again", tier="small", temperature=0.7)
    assert provider.calls[1]["temperature"] == 0.7


# -----------------------------------------------------------------------------
# Configuration and Client Factory tests
# -----------------------------------------------------------------------------


def test_get_client_raises_when_api_key_empty() -> None:
    """get_client raises a clear error if LLM_API_KEY is not set."""
    settings = Settings(llm_provider="anthropic", llm_api_key="")
    with pytest.raises(ValueError, match="LLM_API_KEY is not set"):
        get_client(settings)


def test_config_rejects_unknown_llm_provider() -> None:
    """Config validation rejects unsupported LLM providers."""
    with pytest.raises(ValidationError):
        Settings(llm_provider="unsupported_vendor")  # type: ignore[arg-type]


def test_get_client_instantiates_supported_providers() -> None:
    """get_client builds the proper provider instance matching config."""
    for provider_name, expected_type in [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("gemini", GeminiProvider),
    ]:
        settings = Settings(
            llm_provider=provider_name,  # type: ignore[arg-type]
            llm_api_key="fake-test-key",
            model_small="small-model",
            model_strong="strong-model",
        )
        client = get_client(settings)
        assert isinstance(client.provider, expected_type)


# -----------------------------------------------------------------------------
# Provider SDK exception translation & timeout unit tests (fake exceptions, no network)
# -----------------------------------------------------------------------------


def test_anthropic_and_openai_timeout_in_seconds() -> None:
    """Anthropic and OpenAI adapters pass timeout in seconds."""
    with patch("anthropic.Anthropic") as mock_ant:
        mock_ant.return_value.messages.create.return_value = MagicMock(
            content=[MagicMock(text="hi")],
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        AnthropicProvider(api_key="k", timeout_seconds=45).generate("s", "p", "m")
        _, ant_kwargs = mock_ant.call_args
        assert ant_kwargs.get("timeout") == 45.0
        assert ant_kwargs.get("max_retries") == 0

    with patch("openai.OpenAI") as mock_oa:
        mock_oa.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi"))],
            usage=None,
        )
        OpenAIProvider(api_key="k", timeout_seconds=45).generate("s", "p", "m")
        _, oa_kwargs = mock_oa.call_args
        assert oa_kwargs.get("timeout") == 45.0
        assert oa_kwargs.get("max_retries") == 0


def test_gemini_timeout_converted_to_milliseconds() -> None:
    """Gemini adapter converts timeout from seconds to milliseconds (60s -> 60000ms)."""
    from google.genai import types

    provider = GeminiProvider(api_key="dummy", timeout_seconds=60)
    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text="hi", usage_metadata=None
        )

        provider.generate("sys", "prompt", "model")

        assert mock_cls.call_count == 1
        _, kwargs = mock_cls.call_args
        http_opts = kwargs.get("http_options")
        assert isinstance(http_opts, types.HttpOptions)
        assert http_opts.timeout == 60000
        assert http_opts.retry_options is not None
        assert http_opts.retry_options.attempts == 1


def test_anthropic_adapter_exception_translation() -> None:
    """Anthropic adapter maps transient vs non-transient exceptions accurately."""
    import anthropic

    dummy_request = httpx.Request("POST", "https://api.anthropic.com")
    dummy_resp_429 = httpx.Response(429, request=dummy_request)
    dummy_resp_401 = httpx.Response(401, request=dummy_request)
    dummy_resp_400 = httpx.Response(400, request=dummy_request)

    provider = AnthropicProvider(api_key="dummy")

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # 1. RateLimitError -> LLMTransientError
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            "rate limited", response=dummy_resp_429, body=None
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 2. AuthenticationError -> LLMAuthError
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            "invalid key", response=dummy_resp_401, body=None
        )
        with pytest.raises(LLMAuthError):
            provider.generate("sys", "prompt", "model")

        # 3. BadRequestError -> LLMRequestError
        mock_client.messages.create.side_effect = anthropic.BadRequestError(
            "bad parameter", response=dummy_resp_400, body=None
        )
        with pytest.raises(LLMRequestError):
            provider.generate("sys", "prompt", "model")

        # 4. httpx.ConnectTimeout -> LLMTransientError
        mock_client.messages.create.side_effect = httpx.ConnectTimeout(
            "connect timed out"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 5. SSL handshake timeout -> LLMTransientError
        mock_client.messages.create.side_effect = ssl.SSLError(
            "The handshake operation timed out"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")


def test_openai_adapter_exception_translation() -> None:
    """OpenAI adapter maps transient vs non-transient exceptions accurately."""
    import openai

    dummy_request = httpx.Request("POST", "https://api.openai.com")
    dummy_resp_429 = httpx.Response(429, request=dummy_request)
    dummy_resp_401 = httpx.Response(401, request=dummy_request)
    dummy_resp_400 = httpx.Response(400, request=dummy_request)

    provider = OpenAIProvider(api_key="dummy")

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # 1. RateLimitError -> LLMTransientError
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            "rate limit", response=dummy_resp_429, body=None
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 2. AuthenticationError -> LLMAuthError
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            "unauthorized", response=dummy_resp_401, body=None
        )
        with pytest.raises(LLMAuthError):
            provider.generate("sys", "prompt", "model")

        # 3. BadRequestError -> LLMRequestError
        mock_client.chat.completions.create.side_effect = openai.BadRequestError(
            "bad prompt", response=dummy_resp_400, body=None
        )
        with pytest.raises(LLMRequestError):
            provider.generate("sys", "prompt", "model")

        # 4. httpx.ReadTimeout -> LLMTransientError
        mock_client.chat.completions.create.side_effect = httpx.ReadTimeout(
            "read timed out"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 5. SSL handshake timeout -> LLMTransientError
        mock_client.chat.completions.create.side_effect = ssl.SSLError(
            "The handshake operation timed out"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")


def test_gemini_adapter_exception_translation() -> None:
    """Gemini adapter maps APIError status codes and network errors accurately."""
    from google.genai import errors

    provider = GeminiProvider(api_key="dummy")

    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # 1. 429 -> LLMTransientError
        mock_client.models.generate_content.side_effect = errors.APIError(
            429, "rate limit"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 2. 503 -> LLMTransientError
        mock_client.models.generate_content.side_effect = errors.APIError(
            503, "unavailable"
        )
        with pytest.raises(LLMTransientError):
            provider.generate("sys", "prompt", "model")

        # 3. 401 -> LLMAuthError
        mock_client.models.generate_content.side_effect = errors.APIError(
            401, "unauthenticated"
        )
        with pytest.raises(LLMAuthError):
            provider.generate("sys", "prompt", "model")

        # 4. 400 -> LLMRequestError
        mock_client.models.generate_content.side_effect = errors.APIError(
            400, "invalid argument"
        )
        with pytest.raises(LLMRequestError):
            provider.generate("sys", "prompt", "model")

        # 5. httpx.ConnectTimeout -> LLMTransientError
        mock_client.models.generate_content.side_effect = httpx.ConnectTimeout(
            "connect timed out"
        )
        with pytest.raises(LLMTransientError, match="ConnectTimeout"):
            provider.generate("sys", "prompt", "model")

        # 6. httpx.ReadTimeout -> LLMTransientError
        mock_client.models.generate_content.side_effect = httpx.ReadTimeout(
            "read timed out"
        )
        with pytest.raises(LLMTransientError, match="ReadTimeout"):
            provider.generate("sys", "prompt", "model")

        # 7. SSL handshake timeout -> LLMTransientError
        mock_client.models.generate_content.side_effect = ssl.SSLError(
            "The handshake operation timed out"
        )
        with pytest.raises(LLMTransientError, match="SSLError"):
            provider.generate("sys", "prompt", "model")

        # 8. Non-timeout SSL error -> LLMNonTransientError
        mock_client.models.generate_content.side_effect = ssl.SSLError(
            "certificate verify failed"
        )
        with pytest.raises(LLMNonTransientError, match="SSLError"):
            provider.generate("sys", "prompt", "model")

        # 9. Truly unknown exception includes type name in message
        mock_client.models.generate_content.side_effect = RuntimeError(
            "internal unexpected failure"
        )
        with pytest.raises(LLMNonTransientError, match=r"RuntimeError"):
            provider.generate("sys", "prompt", "model")
