from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

DEFAULT_CLAUDE_TARGET = "claude-sonnet-4-6"
DEFAULT_CLAUDE_JUDGE = "claude-haiku-4-5"


@dataclass
class TargetConfig:
    model: str = DEFAULT_CLAUDE_TARGET
    temperature: float = 0.0
    max_tokens: int = 400
    system_prompt: str = "You are a helpful assistant."


@dataclass
class JudgeConfig:
    type: str = "llm"
    model: str = DEFAULT_CLAUDE_JUDGE
    temperature: float = 0.0
    max_tokens: int = 200


@dataclass
class RequestConfig:
    timeout_s: float = 60.0
    retries: int = 3
    sleep_s: float = 0.2


@dataclass
class EvalConfig:
    dataset: str = "starter"
    include_benign: bool = True
    limit: int | None = None
    shuffle: bool = False
    seed: int = 0
    exclude_categories: list[str] = field(default_factory=list)
    wrapper: str = "direct"
    provider: str = "auto"
    target: TargetConfig = field(default_factory=TargetConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    dry_run: bool = False
    base_url: str | None = None
    api_key: str | None = None

    @property
    def judge_is_llm(self) -> bool:
        return self.judge.type.lower() == "llm" and not self.dry_run


def resolve_provider(model: str, provider: str = "auto") -> str:
    name = (provider or "auto").strip().lower()
    if name in {"openai", "anthropic"}:
        return name
    if name not in {"", "auto"}:
        raise ValueError(f"Unknown provider {provider!r}. Use auto, anthropic, or openai.")
    model_l = (model or "").strip().lower()
    if model_l.startswith("claude") or "claude" in model_l:
        return "anthropic"
    return "openai"


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_config(path: Path | None = None) -> EvalConfig:
    path = path or DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must be a mapping: {path}")
        raw = loaded

    target_raw = raw.get("target") or {}
    judge_raw = raw.get("judge") or {}
    request_raw = raw.get("request") or {}

    return EvalConfig(
        dataset=str(raw.get("dataset", "starter")),
        include_benign=bool(raw.get("include_benign", True)),
        limit=_as_int(raw.get("limit")),
        shuffle=bool(raw.get("shuffle", False)),
        seed=int(raw.get("seed", 0)),
        exclude_categories=list(raw.get("exclude_categories") or []),
        wrapper=str(raw.get("wrapper", "direct")),
        provider=str(raw.get("provider", "auto")),
        target=TargetConfig(
            model=str(target_raw.get("model", DEFAULT_CLAUDE_TARGET)),
            temperature=float(target_raw.get("temperature", 0)),
            max_tokens=int(target_raw.get("max_tokens", 400)),
            system_prompt=str(
                target_raw.get("system_prompt", "You are a helpful assistant.")
            ),
        ),
        judge=JudgeConfig(
            type=str(judge_raw.get("type", "llm")),
            model=str(judge_raw.get("model", DEFAULT_CLAUDE_JUDGE)),
            temperature=float(judge_raw.get("temperature", 0)),
            max_tokens=int(judge_raw.get("max_tokens", 200)),
        ),
        request=RequestConfig(
            timeout_s=float(request_raw.get("timeout_s", 60)),
            retries=int(request_raw.get("retries", 3)),
            sleep_s=float(request_raw.get("sleep_s", 0.2)),
        ),
    )
