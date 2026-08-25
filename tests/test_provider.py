from jailbreak_eval.config import EvalConfig, resolve_provider
from jailbreak_eval.llm import (
    anthropic_create_kwargs,
    missing_api_keys,
    normalize_secret,
    split_system,
)


def test_normalize_secret_strips_quotes_and_placeholders():
    assert normalize_secret("''") is None
    assert normalize_secret('""') is None
    assert normalize_secret("  sk-ant-abc  ") == "sk-ant-abc"
    assert normalize_secret("'sk-ant-abc'") == "sk-ant-abc"
    assert normalize_secret("sk-ant-your-key-here") is None


def test_claude_models_resolve_to_anthropic():
    assert resolve_provider("claude-sonnet-4-6", "auto") == "anthropic"
    assert resolve_provider("claude-haiku-4-5", "auto") == "anthropic"
    assert resolve_provider("gpt-4o-mini", "auto") == "openai"
    assert resolve_provider("claude-sonnet-4-6", "openai") == "openai"


def test_split_system_extracts_system_prompt():
    system, rest = split_system(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
    )
    assert system == "You are a helpful assistant."
    assert rest == [{"role": "user", "content": "hello"}]


def test_anthropic_kwargs_do_not_use_temperature_argument():
    kwargs = anthropic_create_kwargs(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        temperature=0,
    )
    assert "temperature" not in kwargs
    assert kwargs["extra_body"]["temperature"] == 0
    assert kwargs["system"] == "sys"


def test_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    cfg = EvalConfig()
    cfg.target.model = "claude-sonnet-4-6"
    cfg.judge.model = "claude-haiku-4-5"
    assert missing_api_keys(cfg) == ["ANTHROPIC_API_KEY"]
