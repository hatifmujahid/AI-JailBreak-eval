from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from jailbreak_eval.config import ROOT, EvalConfig

STARTER_PATH = ROOT / "data" / "starter.jsonl"
CACHE_DIR = ROOT / "data" / "cache"

# Always drop child-exploitation items, including accidental JBB rows.
CHILD_RE = re.compile(
    r"\b(child(?:ren)?|underage|csam|preteen|pedo(?:phile)?|loli)\b",
    re.IGNORECASE,
)


@dataclass
class Item:
    id: str
    category: str
    behavior: str
    goal: str
    should_refuse: bool
    source: str = "starter"

    def to_dict(self) -> dict:
        return asdict(self)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_starter() -> list[Item]:
    if not STARTER_PATH.exists():
        raise FileNotFoundError(f"Missing starter dataset: {STARTER_PATH}")
    items = []
    for row in _read_jsonl(STARTER_PATH):
        items.append(
            Item(
                id=str(row["id"]),
                category=str(row["category"]),
                behavior=str(row["behavior"]),
                goal=str(row["goal"]),
                should_refuse=bool(row["should_refuse"]),
                source=str(row.get("source", "starter")),
            )
        )
    return items


def _row_text(item: Item) -> str:
    return f"{item.behavior} {item.goal} {item.category}"


def is_blocked(item: Item, exclude_categories: list[str]) -> bool:
    if CHILD_RE.search(_row_text(item)):
        return True
    excluded = {c.strip().lower() for c in exclude_categories if c.strip()}
    return item.category.strip().lower() in excluded


def load_jbb() -> list[Item]:
    """Load JailbreakBench behaviors. Prefers HuggingFace `datasets`, caches JSONL."""
    cache_path = CACHE_DIR / "jbb.jsonl"
    if cache_path.exists():
        return [
            Item(
                id=str(row["id"]),
                category=str(row["category"]),
                behavior=str(row["behavior"]),
                goal=str(row["goal"]),
                should_refuse=bool(row["should_refuse"]),
                source=str(row.get("source", "jbb")),
            )
            for row in _read_jsonl(cache_path)
        ]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "JailbreakBench requires the `datasets` package. "
            "Run: pip install datasets"
        ) from exc

    raw = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    items: list[Item] = []
    for split, should_refuse in (("harmful", True), ("benign", False)):
        if split not in raw:
            continue
        for index, row in enumerate(raw[split], start=1):
            goal = str(row.get("Goal") or row.get("goal") or "").strip()
            if not goal:
                continue
            behavior = str(row.get("Behavior") or row.get("behavior") or goal[:80])
            category = str(row.get("Category") or row.get("category") or "unknown")
            source = str(row.get("Source") or row.get("source") or "jbb")
            items.append(
                Item(
                    id=f"jbb-{'h' if should_refuse else 'b'}-{index:03d}",
                    category=category,
                    behavior=behavior,
                    goal=goal,
                    should_refuse=should_refuse,
                    source=f"jbb/{source}",
                )
            )

    if not items:
        raise RuntimeError("JailbreakBench download returned no rows.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for item in items:
            # Do not persist JBB "Target" strings (affirmative harmful prefixes).
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return items


def load_items(cfg: EvalConfig) -> list[Item]:
    name = cfg.dataset.strip().lower()
    if name == "starter":
        items = load_starter()
    elif name == "jbb":
        items = load_jbb()
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}. Use 'starter' or 'jbb'.")

    items = [item for item in items if not is_blocked(item, cfg.exclude_categories)]
    if not cfg.include_benign:
        items = [item for item in items if item.should_refuse]
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(items)
    if cfg.limit is not None:
        items = items[: cfg.limit]
    if not items:
        raise RuntimeError("No eval items left after filtering.")
    return items
