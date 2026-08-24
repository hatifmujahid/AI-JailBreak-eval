from pathlib import Path

from jailbreak_eval.config import EvalConfig
from jailbreak_eval.report import compute_metrics, write_report


def test_metrics_asr_and_over_refusal():
    rows = [
        {"should_refuse": True, "label": "COMPLIANCE", "category": "Fraud/Deception"},
        {"should_refuse": True, "label": "REFUSAL", "category": "Fraud/Deception"},
        {"should_refuse": False, "label": "REFUSAL", "category": "Benign"},
        {"should_refuse": False, "label": "COMPLIANCE", "category": "Benign"},
        {"should_refuse": True, "label": "ERROR", "category": "Privacy"},
    ]
    metrics = compute_metrics(rows)
    assert metrics["n_harmful"] == 2
    assert metrics["n_benign"] == 2
    assert metrics["n_error"] == 1
    assert metrics["asr"] == 0.5
    assert metrics["over_refusal"] == 0.5
    assert metrics["by_category"]["Fraud/Deception"]["n"] == 2


def test_write_report(tmp_path: Path):
    cfg = EvalConfig(dry_run=True)
    rows = [
        {
            "id": "a",
            "category": "Benign",
            "behavior": "x",
            "goal": "hello",
            "should_refuse": False,
            "source": "starter",
            "wrapper": "direct",
            "prompt": "hello",
            "response": "hi",
            "label": "COMPLIANCE",
            "rationale": "answered",
            "target_model": "dry-run",
            "judge": "dry-run",
        }
    ]
    out = write_report(cfg, rows, tmp_path)
    assert (out / "results.jsonl").exists()
    assert (out / "metrics.json").exists()
    assert "Headline metrics" in (out / "summary.md").read_text(encoding="utf-8")
