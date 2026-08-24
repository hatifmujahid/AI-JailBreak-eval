# Refusal eval suite

A small, runnable **direct-request** eval: send policy-violating asks and matching benign asks to a model, then score whether it refused or complied.

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
```

## Run without an API key (pipeline check)

```powershell
python run_eval.py --dry-run
python run_eval.py --list-items
python -m pytest -q
```

`--dry-run` uses mock replies so you can see the report format. It is not a real safety score.

## Run against a real model

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Optional: set `OPENAI_BASE_URL` for Groq, Together, Ollama, or any OpenAI-compatible server.
3. Edit `config.yaml` if you want a different model name.

Smoke run (6 items, cheap):

```powershell
python run_eval.py --limit 6
```

Full starter set (24 items; each item is one target call + one judge call):

```powershell
python run_eval.py
```

Heuristic judge (no second model):

```powershell
python run_eval.py --judge heuristic --limit 6
```

Ollama example:

```powershell
python run_eval.py --base-url http://localhost:11434/v1 --model llama3.1 --judge heuristic --limit 6
```

For local servers that ignore API keys, put any non-empty value in `OPENAI_API_KEY`.

## JailbreakBench (optional)

The official 100 harmful + 100 benign behaviors:

```powershell
pip install datasets
python run_eval.py --dataset jbb --limit 20
```

The first JBB run downloads from Hugging Face and caches `data/cache/jbb.jsonl` (goals only, not the “Target” strings). Child-related rows are dropped automatically.

## Read the output

After a run:

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
