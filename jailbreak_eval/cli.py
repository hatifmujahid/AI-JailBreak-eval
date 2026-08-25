from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from jailbreak_eval.config import DEFAULT_CONFIG, ROOT, EvalConfig, load_config, resolve_provider
from jailbreak_eval.dataset import load_items
from jailbreak_eval.llm import anthropic_key, missing_api_keys
from jailbreak_eval.report import compute_metrics, write_report
from jailbreak_eval.runner import run_eval


def _print_metrics(metrics: dict) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    print()
    print("=== Headline metrics ===")
    print(f"  Attack success rate : {pct(metrics['asr'])}   (harmful items answered)")
    print(f"  Harmful refusal     : {pct(metrics['harmful_refusal'])}")
    print(f"  Over-refusal        : {pct(metrics['over_refusal'])}   (benign items refused)")
    print(f"  Benign answers      : {pct(metrics['benign_answer'])}")
    print(f"  Errors              : {metrics['n_error']}/{metrics['n_total']}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a direct-request refusal eval (starter set or JailbreakBench)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=("starter", "jbb"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", help="Target model name")
    parser.add_argument("--judge-model", help="Judge model name")
    parser.add_argument("--judge", choices=("llm", "heuristic"))
    parser.add_argument(
        "--provider",
        choices=("auto", "anthropic", "openai"),
        help="API provider. auto infers Claude models as anthropic",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--dry-run", action="store_true", help="No API calls; mock replies")
    parser.add_argument(
        "--list-items",
        action="store_true",
        help="Print the items that would be sent, then exit",
    )
    parser.add_argument("--out", type=Path, help="Directory for this run's report")
    return parser


def apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    if args.dataset:
        cfg.dataset = args.dataset
    if args.limit is not None:
        cfg.limit = args.limit
    if args.model:
        cfg.target.model = args.model
    if args.judge_model:
        cfg.judge.model = args.judge_model
    if args.judge:
        cfg.judge.type = args.judge
    if args.provider:
        cfg.provider = args.provider
    if args.base_url:
        cfg.base_url = args.base_url
    cfg.dry_run = bool(args.dry_run)
    return cfg


def main(argv: list[str] | None = None) -> int:
    # Project .env must win over an empty/quoted shell variable like ANTHROPIC_API_KEY=''
    load_dotenv(ROOT / ".env", override=True)
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    items = load_items(cfg)

    if args.list_items:
        for item in items:
            kind = "HARM" if item.should_refuse else "OK  "
            print(f"{item.id:16} {kind}  {item.category:22} {item.goal}")
        print(f"\n{len(items)} items")
        return 0

    target_provider = resolve_provider(cfg.target.model, cfg.provider)
    print(
        f"Running {len(items)} items | dataset={cfg.dataset} | "
        f"target={'dry-run' if cfg.dry_run else cfg.target.model} | "
        f"provider={'dry-run' if cfg.dry_run else target_provider} | "
        f"judge={'dry-run' if cfg.dry_run else cfg.judge.type}"
    )
    if not cfg.dry_run:
        missing = missing_api_keys(cfg)
        if missing:
            print("Missing API key(s): " + ", ".join(missing))
            print("Copy .env.example to .env and set the key, or pass --dry-run.")
            return 2
        key = anthropic_key(cfg) if target_provider == "anthropic" else None
        if key:
            print(f"Using Anthropic key from .env ({len(key)} chars, starts with {key[:7]})")

    rows = run_eval(items, cfg, progress=print)
    out_dir = write_report(cfg, rows, args.out)
    metrics = compute_metrics(rows)
    _print_metrics(metrics)
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0
