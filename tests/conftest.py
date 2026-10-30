"""Shared pytest configuration and test doubles."""

from typing import Any

from jobagent.llm.client import LLMResponse


class FakeProvider:
    """Mock provider returning canned responses or raising configured exceptions."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = responses or []
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.responses:
            return LLMResponse(
                text='{"role": "backend", "confidence": 0.95}',
                model=model,
                latency_seconds=0.05,
            )

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, str):
            return LLMResponse(text=resp, model=model, latency_seconds=0.05)
        if isinstance(resp, LLMResponse):
            return resp
        raise TypeError(f"Unsupported fake response type: {type(resp)}")
