from __future__ import annotations

import time
from typing import Callable

from jailbreak_eval.config import EvalConfig
from jailbreak_eval.dataset import Item
from jailbreak_eval.judge import heuristic_judge, judge_messages, parse_judge_json
from jailbreak_eval.llm import LLMClient
from jailbreak_eval.wrappers import wrap

MOCK_REFUSAL = (
    "I can't help with that request. If you have a legal, high-level question "
    "about staying safe, I can discuss that instead."
)
MOCK_ANSWER = (
    "Here is a concise, on-topic answer with practical next steps and no "
    "disallowed content."
)
MOCK_COMPLIANCE = (
    "[MOCK COMPLIANCE — no real harmful content] Sure, I would provide the "
    "requested material here. This string only exists so the dry-run judge "
    "can label a compliance case."
)


class DryRunClient:
    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        joined = " ".join(m["content"] for m in messages).lower()
        if "json schema" in joined or "pick exactly one label" in joined:
            if "mock compliance" in joined:
                return '{"label": "COMPLIANCE", "rationale": "dry-run mock compliance"}'
            if "i can't help" in joined or "can't help with that request" in joined:
                return '{"label": "REFUSAL", "rationale": "dry-run mock refusal"}'
            return '{"label": "COMPLIANCE", "rationale": "dry-run mock answer"}'
        return MOCK_ANSWER


def _target_reply(item: Item, index: int, dry_run: bool) -> str:
    if not dry_run:
        raise RuntimeError("target mock used outside dry-run")
    if not item.should_refuse:
        return MOCK_ANSWER
    # Alternate so a dry-run report is not 0% ASR / 0% over-refusal.
    return MOCK_REFUSAL if index % 2 == 0 else MOCK_COMPLIANCE


def run_eval(
    items: list[Item],
    cfg: EvalConfig,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    log = progress or (lambda _msg: None)
    target: LLMClient | DryRunClient
    judge_client: LLMClient | DryRunClient | None

    if cfg.dry_run:
        target = DryRunClient()
        judge_client = DryRunClient()
    else:
        target = LLMClient(
            cfg,
            model=cfg.target.model,
            temperature=cfg.target.temperature,
            max_tokens=cfg.target.max_tokens,
        )
        judge_client = None
        if cfg.judge_is_llm:
            judge_client = LLMClient(
                cfg,
                model=cfg.judge.model,
                temperature=cfg.judge.temperature,
                max_tokens=cfg.judge.max_tokens,
            )

    rows: list[dict] = []
    for index, item in enumerate(items):
        prompt = wrap(item, cfg.wrapper)
        messages = []
        if cfg.target.system_prompt.strip():
            messages.append({"role": "system", "content": cfg.target.system_prompt})
        messages.append({"role": "user", "content": prompt})

        error = ""
        try:
            if cfg.dry_run:
                reply = _target_reply(item, index, True)
            else:
                reply = target.chat(messages)
        except Exception as exc:  # noqa: BLE001
            reply = ""
            error = str(exc)

        if error:
            verdict = {"label": "ERROR", "rationale": error}
        elif cfg.dry_run or cfg.judge_is_llm:
            assert judge_client is not None
            try:
                raw = judge_client.chat(judge_messages(item, reply), json_mode=True)
                verdict = parse_judge_json(raw)
            except Exception as exc:  # noqa: BLE001
                verdict = {"label": "UNCLEAR", "rationale": f"judge failed: {exc}"}
        else:
            verdict = heuristic_judge(reply)

        row = {
            **item.to_dict(),
            "wrapper": cfg.wrapper,
            "prompt": prompt,
            "response": reply,
            "label": verdict["label"],
            "rationale": verdict["rationale"],
            "target_model": "dry-run" if cfg.dry_run else cfg.target.model,
            "judge": "dry-run" if cfg.dry_run else cfg.judge.type,
        }
        rows.append(row)
        log(f"[{index + 1}/{len(items)}] {item.id} -> {row['label']}")
        if not cfg.dry_run and cfg.request.sleep_s:
            time.sleep(cfg.request.sleep_s)
    return rows
