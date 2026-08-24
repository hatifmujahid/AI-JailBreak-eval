from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from jailbreak_eval.config import EvalConfig


class LLMClient:
    """OpenAI-compatible chat client (OpenAI, Groq, Together, Ollama, ...)."""

    def __init__(self, cfg: EvalConfig, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = cfg.request.retries
        self.timeout_s = cfg.request.timeout_s
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or None
        if base_url == "":
            base_url = None
        if not api_key and not (base_url and "localhost" in base_url):
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Copy .env.example to .env, or pass --dry-run."
            )
        kwargs: dict[str, Any] = {"api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout_s,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = self._client.chat.completions.create(**kwargs)
                except Exception:
                    if not json_mode:
                        raise
                    kwargs.pop("response_format", None)
                    response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:  # noqa: BLE001 - surface provider errors as retries
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} tries: {last_error}")
