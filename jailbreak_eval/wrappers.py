from __future__ import annotations

from jailbreak_eval.dataset import Item


def wrap(item: Item, wrapper: str) -> str:
    name = wrapper.strip().lower()
    if name == "direct":
        return item.goal
    raise ValueError(
        f"Unknown wrapper {wrapper!r}. v1 only supports 'direct' "
        "(the behavior is sent as a plain user message)."
    )
