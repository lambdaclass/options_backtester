from __future__ import annotations

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
    assert tree.attribution["a"]["capital"] == int(900_000 * (2.0 / 3.0))
    assert tree.attribution["b"]["capital"] == int(900_000 * (1.0 / 3.0))
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
