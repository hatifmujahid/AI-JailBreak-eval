from jailbreak_eval.config import EvalConfig, load_config
from jailbreak_eval.dataset import load_items
from jailbreak_eval.report import compute_metrics
from jailbreak_eval.runner import run_eval


def test_dry_run_end_to_end():
    cfg = load_config()
    cfg.dry_run = True
    cfg.dataset = "starter"
    cfg.limit = 6
    items = load_items(cfg)
    assert len(items) == 6
    rows = run_eval(items, cfg)
    assert len(rows) == 6
    assert {row["label"] for row in rows} <= {"REFUSAL", "COMPLIANCE", "UNCLEAR"}
    metrics = compute_metrics(rows)
    assert metrics["n_total"] == 6
    assert metrics["n_error"] == 0
