"""Provider-agnostic LLM client wrapper and adapters."""

import logging
import re
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from jobagent.config import Settings, get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(Exception):
    """Base exception for all LLM client errors."""


LLMClientError = LLMError


class LLMTransientError(LLMError):
    """Transient errors that can be retried (rate limits, timeouts, connection errors, 5xx)."""


class LLMNonTransientError(LLMError):
    """Non-transient errors that must not be retried (auth, bad requests)."""


class LLMAuthError(LLMNonTransientError):
    """Authentication or permissions errors (401, 403)."""


class LLMRequestError(LLMNonTransientError):
    """Client request errors (400, 404, 422)."""


class StructuredOutputError(LLMError):
    """Raised when structured JSON parsing/validation fails after retry."""


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str
    latency_seconds: float = Field(ge=0.0)


class Provider(Protocol):
    """Protocol for LLM provider adapters."""

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given system and user prompts."""
        ...


class AnthropicProvider:
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=float(self.timeout_seconds),
            max_retries=0,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            resp = client.messages.create(**kwargs)
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as e:
            raise LLMTransientError(f"Anthropic transient error: {e}") from e
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
        ) as e:
            raise LLMAuthError(f"Anthropic authentication error: {e}") from e
        except (anthropic.BadRequestError, anthropic.NotFoundError) as e:
            raise LLMRequestError(f"Anthropic request error: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise LLMTransientError(
                    f"Anthropic transient status {e.status_code}: {e}"
                ) from e
            if e.status_code in (401, 403):
                raise LLMAuthError(f"Anthropic auth status {e.status_code}: {e}") from e
            raise LLMRequestError(
                f"Anthropic client status {e.status_code}: {e}"
            ) from e
        except anthropic.AnthropicError as e:
            raise LLMNonTransientError(f"Anthropic error: {e}") from e

        latency = time.perf_counter() - start
        text_content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text_content += block.text

        return LLMResponse(
            text=text_content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=model,
            latency_seconds=latency,
        )


class OpenAIProvider:
    """Adapter for the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            timeout=float(self.timeout_seconds),
            max_retries=0,
        )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as e:
            raise LLMTransientError(f"OpenAI transient error: {e}") from e
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
        ) as e:
            raise LLMAuthError(f"OpenAI authentication error: {e}") from e
        except (
            openai.BadRequestError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as e:
            raise LLMRequestError(f"OpenAI request error: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise LLMTransientError(
                    f"OpenAI transient status {e.status_code}: {e}"
                ) from e
            if e.status_code in (401, 403):
                raise LLMAuthError(f"OpenAI auth status {e.status_code}: {e}") from e
            raise LLMRequestError(f"OpenAI client status {e.status_code}: {e}") from e
        except openai.OpenAIError as e:
            raise LLMNonTransientError(f"OpenAI error: {e}") from e

        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        in_tokens = resp.usage.prompt_tokens if resp.usage else 0
        out_tokens = resp.usage.completion_tokens if resp.usage else 0

        return LLMResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=model,
            latency_seconds=latency,
        )


class GeminiProvider:
    """Adapter for the Google Gemini API (google-genai)."""

    def __init__(self, api_key: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        from google import genai
        from google.genai import errors, types

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=float(self.timeout_seconds)),
        )

        config_args: dict[str, Any] = {}
        if system:
            config_args["system_instruction"] = system
        if max_tokens:
            config_args["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_args["temperature"] = temperature

        config = types.GenerateContentConfig(**config_args) if config_args else None

        start = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except (errors.APIError, errors.ClientError, errors.ServerError) as e:
            code = getattr(e, "code", None)
            if code == 429 or (isinstance(code, int) and code >= 500):
                raise LLMTransientError(f"Gemini transient error ({code}): {e}") from e
            if code in (401, 403):
                raise LLMAuthError(f"Gemini authentication error ({code}): {e}") from e
            if isinstance(code, int) and 400 <= code < 500:
                raise LLMRequestError(f"Gemini request error ({code}): {e}") from e
            raise LLMNonTransientError(f"Gemini error ({code}): {e}") from e
        except (TimeoutError, ConnectionError) as e:
            raise LLMTransientError(f"Gemini connection/timeout error: {e}") from e
        except Exception as e:
            raise LLMNonTransientError(f"Gemini unexpected error: {e}") from e

        latency = time.perf_counter() - start
        text = resp.text or ""
        in_tokens = 0
        out_tokens = 0
        if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
            in_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            out_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=model,
            latency_seconds=latency,
        )


def strip_code_fences(content: str) -> str:
    """Remove markdown code fences (e.g. ```json ... ```) from response text."""
    text = content.strip()
    match = re.match(r"^```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class LLMClient:
    """Unified client for executing completions and structured outputs with retries."""

    def __init__(
        self,
        provider: Provider,
        model_small: str = "",
        model_strong: str = "",
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.model_small = model_small
        self.model_strong = model_strong
        self.sleep_fn = sleep_fn

    def _resolve_model(self, tier: Literal["small", "strong"]) -> str:
        model = self.model_small if tier == "small" else self.model_strong
        if not model.strip():
            raise ValueError(
                f"Model name for tier '{tier}' is not configured. "
                f"Please set LLM_MODEL_{tier.upper()} in your environment or .env file."
            )
        return model.strip()

    def complete(
        self,
        system: str,
        prompt: str,
        tier: Literal["small", "strong"] = "small",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call provider with exponential backoff on transient errors."""
        model = self._resolve_model(tier)
        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.provider.generate(
                    system=system,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                logger.info(
                    "LLM call succeeded: provider=%s model=%s tier=%s tokens_in=%d tokens_out=%d latency=%.2fs",
                    self.provider.__class__.__name__,
                    model,
                    tier,
                    response.input_tokens,
                    response.output_tokens,
                    response.latency_seconds,
                )
                return response
            except LLMTransientError as e:
                last_error = e
                if attempt < max_attempts:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM transient error on attempt %d/%d: %s. Backing off for %.2fs.",
                        attempt,
                        max_attempts,
                        e,
                        backoff,
                    )
                    self.sleep_fn(backoff)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s", max_attempts, e
                    )
                    raise
            except Exception as e:
                # Non-transient errors must NOT be retried.
                logger.error("LLM non-transient error: %s", e)
                raise

        if last_error:
            raise last_error
        raise LLMClientError("LLM call failed.")

    def complete_structured(
        self,
        system: str,
        prompt: str,
        response_model: type[ModelT],
        tier: Literal["small", "strong"] = "small",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> ModelT:
        """Execute a completion and validate output against a Pydantic model with 1 retry."""
        structured_prompt = (
            f"{prompt}\n\n"
            "Respond strictly with valid JSON conforming to the requested schema. "
            "Do not include explanation or markdown formatting outside the JSON."
        )

        resp = self.complete(
            system=system,
            prompt=structured_prompt,
            tier=tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cleaned_text = strip_code_fences(resp.text)
        try:
            return response_model.model_validate_json(cleaned_text)
        except (ValueError, ValidationError) as first_err:
            logger.warning(
                "Structured output validation failed on first attempt: %s. Retrying once with error feedback.",
                first_err,
            )
            retry_prompt = (
                f"{structured_prompt}\n\n"
                f"Your previous response failed validation with error:\n{first_err}\n"
                "Please output strictly valid JSON conforming to the schema."
            )
            retry_resp = self.complete(
                system=system,
                prompt=retry_prompt,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cleaned_retry_text = strip_code_fences(retry_resp.text)
            try:
                return response_model.model_validate_json(cleaned_retry_text)
            except (ValueError, ValidationError) as second_err:
                raise StructuredOutputError(
                    f"Failed to produce valid {response_model.__name__} after retry: {second_err}"
                ) from second_err


def get_client(settings: Settings | None = None) -> LLMClient:
    """Factory creating an LLMClient for the configured provider."""
    cfg = settings or get_settings()

    if not cfg.llm_api_key.strip():
        raise ValueError(
            "LLM_API_KEY is not set. Please set LLM_API_KEY in your .env file or environment variables."
        )

    provider: Provider
    if cfg.llm_provider == "anthropic":
        provider = AnthropicProvider(
            api_key=cfg.llm_api_key, timeout_seconds=cfg.llm_timeout_seconds
        )
    elif cfg.llm_provider == "openai":
        provider = OpenAIProvider(
            api_key=cfg.llm_api_key, timeout_seconds=cfg.llm_timeout_seconds
        )
    elif cfg.llm_provider == "gemini":
        provider = GeminiProvider(
            api_key=cfg.llm_api_key, timeout_seconds=cfg.llm_timeout_seconds
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")

    return LLMClient(
        provider=provider,
        model_small=cfg.model_small,
        model_strong=cfg.model_strong,
    )
