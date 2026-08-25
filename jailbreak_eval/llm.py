from __future__ import annotations

import os
import time
from typing import Any

from jailbreak_eval.config import EvalConfig, resolve_provider


def normalize_secret(value: str | None) -> str | None:
    """Strip quotes/whitespace. Treat placeholders and empty quotes as missing."""
    if value is None:
        return None
    text = value.strip().strip("\"'").strip()
    if not text:
        return None
    lowered = text.lower()
    if "your-key" in lowered or lowered in {"changeme", "xxx", "todo"}:
        return None
    return text


def anthropic_key(cfg: EvalConfig | None = None) -> str | None:
    if cfg and cfg.api_key:
        return normalize_secret(cfg.api_key)
    return normalize_secret(os.getenv("ANTHROPIC_API_KEY")) or normalize_secret(
        os.getenv("CLAUDE_API_KEY")
    )


def openai_key(cfg: EvalConfig | None = None) -> str | None:
    if cfg and cfg.api_key:
        return normalize_secret(cfg.api_key)
    return normalize_secret(os.getenv("OPENAI_API_KEY")) or normalize_secret(
        os.getenv("GROQ_API_KEY")
    )


def split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(part for part in system_parts if part.strip()) or None
    return system, rest


def _fatal_status(exc: BaseException) -> bool:
    if isinstance(exc, (TypeError, ValueError)):
        return True
    status = getattr(exc, "status_code", None)
    return status in {400, 401, 403, 404, 422}


def anthropic_create_kwargs(
    model: str,
    max_tokens: int,
    messages: list[dict[str, str]],
    system: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build Messages.create kwargs for anthropic>=1.0 (no temperature argument)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    # SDK 1.0 removed `temperature=`; the HTTP API still accepts it on extra_body.
    if temperature is not None:
        kwargs["extra_body"] = {"temperature": temperature}
    return kwargs


def missing_api_keys(cfg: EvalConfig) -> list[str]:
    needed: list[str] = []
    target_provider = resolve_provider(cfg.target.model, cfg.provider)
    if target_provider == "anthropic" and not anthropic_key(cfg):
        needed.append("ANTHROPIC_API_KEY")
    elif target_provider == "openai":
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or ""
        local = "localhost" in base_url or "127.0.0.1" in base_url
        if not openai_key(cfg) and not local:
            needed.append("OPENAI_API_KEY")
    if cfg.judge_is_llm:
        judge_provider = resolve_provider(cfg.judge.model, cfg.provider)
        if judge_provider == "anthropic" and not anthropic_key(cfg):
            if "ANTHROPIC_API_KEY" not in needed:
                needed.append("ANTHROPIC_API_KEY")
        elif judge_provider == "openai":
            base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or ""
            local = "localhost" in base_url or "127.0.0.1" in base_url
            if not openai_key(cfg) and not local and "OPENAI_API_KEY" not in needed:
                needed.append("OPENAI_API_KEY")
    return needed


class LLMClient:
    """Chat client for Anthropic (Claude) or OpenAI-compatible APIs."""

    def __init__(self, cfg: EvalConfig, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = cfg.request.retries
        self.timeout_s = cfg.request.timeout_s
        self.provider = resolve_provider(model, cfg.provider)
        self._openai = None
        self._anthropic = None

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError("Claude support needs the anthropic package. Run: pip install anthropic") from exc
            key = anthropic_key(cfg)
            if not key:
                raise RuntimeError(
                    "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and paste your Claude API key."
                )
            self._anthropic = Anthropic(api_key=key, timeout=self.timeout_s)
            return

        from openai import OpenAI

        key = openai_key(cfg)
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or None
        if base_url == "":
            base_url = None
        local = bool(base_url) and ("localhost" in base_url or "127.0.0.1" in base_url)
        if not key and not local:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Copy .env.example to .env, or pass --dry-run."
            )
        kwargs: dict[str, Any] = {"api_key": key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url
        self._openai = OpenAI(**kwargs)

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self.provider == "anthropic":
                    return self._chat_anthropic(messages)
                return self._chat_openai(messages, json_mode=json_mode)
            except Exception as exc:  # noqa: BLE001
                if _fatal_status(exc):
                    raise RuntimeError(f"LLM call failed: {exc}") from exc
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} tries: {last_error}")

    def _chat_anthropic(self, messages: list[dict[str, str]]) -> str:
        assert self._anthropic is not None
        system, rest = split_system(messages)
        if not rest:
            rest = [{"role": "user", "content": "(empty)"}]
        kwargs = anthropic_create_kwargs(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=rest,
            system=system,
            temperature=self.temperature,
        )
        try:
            response = self._anthropic.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Some model IDs reject temperature even on extra_body.
            if kwargs.get("extra_body") and (
                getattr(exc, "status_code", None) == 400 or "temperature" in str(exc).lower()
            ):
                kwargs.pop("extra_body", None)
                response = self._anthropic.messages.create(**kwargs)
            else:
                raise
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def _chat_openai(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        assert self._openai is not None
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
            response = self._openai.chat.completions.create(**kwargs)
        except Exception:
            if not json_mode:
                raise
            kwargs.pop("response_format", None)
            response = self._openai.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()
