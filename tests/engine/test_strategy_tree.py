from __future__ import annotations

import pytest

from options_backtester.engine.strategy_tree import StrategyTreeNode, StrategyTreeEngine

from tests.engine.test_engine import _run_engine


def test_strategy_tree_allocates_capital_by_weights():
    leaf_a = StrategyTreeNode(name="a", weight=2.0, engine=_run_engine())
    leaf_b = StrategyTreeNode(name="b", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[leaf_a, leaf_b])
    tree = StrategyTreeEngine(root, initial_capital=900_000)

    tree.run(rebalance_freq=1)

    assert abs(tree.leaf_weights["a"] - (2.0 / 3.0)) < 1e-12
    assert abs(tree.leaf_weights["b"] - (1.0 / 3.0)) < 1e-12
    assert tree.attribution["a"]["capital"] == round(900_000 * (2.0 / 3.0))
    assert tree.attribution["b"]["capital"] == round(900_000 * (1.0 / 3.0))
    assert "total capital" in tree.balance.columns


def test_nested_tree_weight_propagation():
    leaf_a = StrategyTreeNode(name="a", weight=1.0, engine=_run_engine())
    leaf_b = StrategyTreeNode(name="b", weight=3.0, engine=_run_engine())
    branch = StrategyTreeNode(name="branch", weight=2.0, children=[leaf_a, leaf_b])
    leaf_c = StrategyTreeNode(name="c", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[branch, leaf_c])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)

    tree.run(rebalance_freq=1)

    # branch share = 2/3; inside branch, a=1/4 and b=3/4
    assert abs(tree.leaf_weights["a"] - (2.0 / 3.0) * (1.0 / 4.0)) < 1e-12
    assert abs(tree.leaf_weights["b"] - (2.0 / 3.0) * (3.0 / 4.0)) < 1e-12
    assert abs(tree.leaf_weights["c"] - (1.0 / 3.0)) < 1e-12


def test_leaf_max_share_throttles_allocation():
    leaf_a = StrategyTreeNode(name="a", weight=1.0, max_share=0.20, engine=_run_engine())
    leaf_b = StrategyTreeNode(name="b", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[leaf_a, leaf_b])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)

    tree.run(rebalance_freq=1)

    assert abs(tree.leaf_weights["a"] - 0.20) < 1e-12
    assert "a" in tree.throttles
    assert "unallocated_cash" in tree.balance.columns


# ---------------------------------------------------------------------------
# Validation (item 11)
# ---------------------------------------------------------------------------

def test_node_rejects_engine_and_children():
    """A node cannot be both a leaf (engine) and a branch (children)."""
    with pytest.raises(ValueError, match="both engine and children"):
        StrategyTreeNode(
            name="bad",
            engine=_run_engine(),
            children=[StrategyTreeNode(name="child", engine=_run_engine())],
        )


def test_empty_branch_produces_no_leaves():
    empty = StrategyTreeNode(name="empty", children=[])
    root = StrategyTreeNode(name="root", children=[empty])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)
    results = tree.run(rebalance_freq=1)
    assert len(results) == 0
    assert tree.balance.empty


# ---------------------------------------------------------------------------
# Capital restoration (item 7)
# ---------------------------------------------------------------------------

def test_engine_capital_restored_after_tree_run():
    """StrategyTreeEngine should not permanently mutate leaf engine capital."""
    engine = _run_engine()
    original_capital = engine.initial_capital
    leaf = StrategyTreeNode(name="a", weight=1.0, engine=engine)
    root = StrategyTreeNode(name="root", children=[leaf])
    tree = StrategyTreeEngine(root, initial_capital=500_000)
    tree.run(rebalance_freq=1)
    assert engine.initial_capital == original_capital


# ---------------------------------------------------------------------------
# Round vs int for capital allocation (item 5)
# ---------------------------------------------------------------------------

def test_capital_uses_round_not_truncate():
    """With 3 equal-weight leaves and 1M capital, round(1e6/3) = 333333."""
    engines = [_run_engine() for _ in range(3)]
    leaves = [StrategyTreeNode(name=f"l{i}", weight=1.0, engine=e) for i, e in enumerate(engines)]
    root = StrategyTreeNode(name="root", children=leaves)
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)
    tree.run(rebalance_freq=1)
    total_allocated = sum(tree.attribution[f"l{i}"]["capital"] for i in range(3))
    # With round(), 333333 * 3 = 999999 — only $1 lost vs $3 with int()
    assert total_allocated >= 999_999


# ---------------------------------------------------------------------------
# Balance structure
# ---------------------------------------------------------------------------

def test_balance_has_pct_change_and_accumulated_return():
    leaf = StrategyTreeNode(name="a", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[leaf])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)
    tree.run(rebalance_freq=1)
    assert "% change" in tree.balance.columns
    assert "accumulated return" in tree.balance.columns
    assert "total capital" in tree.balance.columns


def test_attribution_dict_structure():
    leaf = StrategyTreeNode(name="a", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[leaf])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)
    tree.run(rebalance_freq=1)
    assert "a" in tree.attribution
    assert "weight" in tree.attribution["a"]
    assert "capital" in tree.attribution["a"]
    assert tree.attribution["a"]["weight"] == 1.0


# ---------------------------------------------------------------------------
# to_dot() — Graphviz DOT export
# ---------------------------------------------------------------------------

def test_to_dot_single_leaf():
    leaf = StrategyTreeNode(name="leaf_a", weight=1.0, engine=_run_engine())
    dot = leaf.to_dot()
    assert "digraph StrategyTree" in dot
    assert "leaf_a" in dot
    assert "w=1.0" in dot
    assert "ellipse" in dot  # leaf → ellipse shape


def test_to_dot_nested_tree():
    leaf_a = StrategyTreeNode(name="a", weight=2.0, engine=_run_engine())
    leaf_b = StrategyTreeNode(name="b", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="root", children=[leaf_a, leaf_b])
    dot = root.to_dot()
    assert "digraph StrategyTree" in dot
    assert "root" in dot
    assert "box" in dot  # branch → box shape
    assert "->" in dot  # edges exist
    assert "w=2.0" in dot
    assert "w=1.0" in dot


def test_to_dot_max_share_shown():
    leaf = StrategyTreeNode(name="capped", weight=1.0, max_share=0.25, engine=_run_engine())
    dot = leaf.to_dot()
    assert "max=0.25" in dot


def test_engine_to_dot_delegates_to_root():
    leaf = StrategyTreeNode(name="x", weight=1.0, engine=_run_engine())
    root = StrategyTreeNode(name="top", children=[leaf])
    tree = StrategyTreeEngine(root, initial_capital=1_000_000)
    dot = tree.to_dot()
    assert "digraph StrategyTree" in dot
    assert "top" in dot
    assert "x" in dot
