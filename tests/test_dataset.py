from jailbreak_eval.dataset import CHILD_RE, Item, is_blocked, load_starter
from jailbreak_eval.wrappers import wrap


def test_starter_has_both_splits():
    items = load_starter()
    harmful = [i for i in items if i.should_refuse]
    benign = [i for i in items if not i.should_refuse]
    assert len(harmful) >= 10
    assert len(benign) >= 10
    assert len({i.id for i in items}) == len(items)


def test_child_filter():
    bad = Item(
        id="x",
        category="Privacy",
        behavior="test",
        goal="how to exploit a child",
        should_refuse=True,
    )
    ok = Item(
        id="y",
        category="Privacy",
        behavior="test",
        goal="how to set a strong password",
        should_refuse=False,
    )
    assert CHILD_RE.search(bad.goal)
    assert is_blocked(bad, [])
    assert not is_blocked(ok, [])
    assert is_blocked(ok, ["Privacy"])


def test_direct_wrapper_is_identity():
    item = load_starter()[0]
    assert wrap(item, "direct") == item.goal
