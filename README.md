# Refusal eval suite

A small, runnable **direct-request** eval: send policy-violating asks and matching benign asks to a model, then score whether it refused or complied.

Default target is **Claude** via the Anthropic API.

This is **not** a jailbreak-attack generator. It does not ship DAN-style templates, encoded exploits, or homemade attack recipes. v1 only tests the plain request.

## What you get

| File | Role |
|---|---|
| `data/starter.jsonl` | 12 harmful + 12 benign items (always available) |
| `run_eval.py` | CLI |
| `results/<run>/summary.md` | Human-readable report |
| `results/<run>/results.jsonl` | Raw scored rows |

**Metrics**

- **Attack success rate (ASR)** — share of harmful items the model answered (lower is better)
- **Over-refusal** — share of benign items the model refused (lower is better)

A model that refuses everything looks safe on ASR and fails on over-refusal. Read both.

## Setup (Windows)

```powershell
cd C:\Users\Pc\Desktop\AI-JailBreak-eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key from [Anthropic Console](https://console.anthropic.com/settings/keys).

## Run Claude

Smoke run (6 items; target + judge = 12 API calls):

```powershell
python run_eval.py --limit 6
```

Full starter set (24 items):

```powershell
python run_eval.py
```

Other Claude models:

```powershell
python run_eval.py --model claude-sonnet-4-6 --judge-model claude-haiku-4-5 --limit 6
python run_eval.py --model claude-opus-4-6 --limit 6
python run_eval.py --model claude-haiku-4-5 --judge heuristic --limit 6
```

If a model ID is rejected by the API, change `target.model` in `config.yaml` to an ID from [Anthropic’s model list](https://platform.claude.com/docs/en/about-claude/models/overview).

## Run without an API key (pipeline check)

```powershell
python run_eval.py --dry-run
python run_eval.py --list-items
python -m pytest -q
```

`--dry-run` uses mock replies. It is not a real safety score.

## OpenAI / Ollama

```powershell
python run_eval.py --provider openai --model gpt-4o-mini --judge-model gpt-4o-mini --limit 6
python run_eval.py --provider openai --base-url http://localhost:11434/v1 --model llama3.1 --judge heuristic --limit 6
```

Set `OPENAI_API_KEY` in `.env` for OpenAI-compatible providers.

## JailbreakBench (optional)

```powershell
pip install datasets
python run_eval.py --dataset jbb --limit 20
```

The first JBB run downloads from Hugging Face and caches `data/cache/jbb.jsonl`. Child-related rows are dropped automatically.

## Read the output

```
results/YYYYMMDD-HHMMSS-starter/
  summary.md      <- start here
  metrics.json
  results.jsonl   <- may contain harmful model output; keep it private
```

Spot-check 10–20 rows yourself. An LLM judge is useful and imperfect.

## What this suite does not do

- It does not invent or store jailbreak payloads.
- It does not include child-sexual-exploitation items.
- It does not include expected “working” answers for weapons, malware, or fraud.
- Multi-turn attacks and encoded wrappers are out of scope for v1.

## Cite

If you use `--dataset jbb`, cite [JailbreakBench](https://jailbreakbench.github.io/) (Chao et al., NeurIPS 2024).
