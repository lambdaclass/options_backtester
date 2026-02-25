"""Hierarchical strategy tree runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from options_backtester.engine.engine import BacktestEngine


@dataclass
class StrategyTreeNode:
    """Node in a capital-allocation strategy tree."""

    name: str
    weight: float = 1.0
    max_share: float | None = None
    engine: BacktestEngine | None = None
    children: list["StrategyTreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.engine is not None and self.children:
            raise ValueError(
                f"StrategyTreeNode '{self.name}' has both engine and children; "
                "a node must be either a leaf (engine) or a branch (children), not both"
            )

    def is_leaf(self) -> bool:
        return self.engine is not None

    def to_dot(self) -> str:
        """Generate Graphviz DOT string for this subtree."""
        lines = [
            "digraph StrategyTree {",
            "  rankdir=TB;",
            '  node [style=filled, fillcolor=lightyellow];',
        ]
        self._dot_recursive(lines, parent_id=None)
        lines.append("}")
        return "\n".join(lines)

    def _dot_recursive(self, lines: list[str], parent_id: str | None) -> None:
        node_id = f"n{id(self)}"
        label = f"{self.name}\\nw={self.weight}"
        if self.max_share is not None:
            label += f"\\nmax={self.max_share}"
        shape = "ellipse" if self.is_leaf() else "box"
        lines.append(f'  {node_id} [label="{label}", shape={shape}];')
        if parent_id:
            lines.append(f"  {parent_id} -> {node_id};")
        for child in self.children:
            child._dot_recursive(lines, node_id)


class StrategyTreeEngine:
    """Run leaf engines with capital shares implied by tree weights."""

    def __init__(self, root: StrategyTreeNode, initial_capital: int = 1_000_000) -> None:
        self.root = root
        self.initial_capital = initial_capital
        self.throttles: dict[str, dict[str, float]] = {}

    def to_dot(self) -> str:
        """Generate Graphviz DOT string for the strategy tree."""
        return self.root.to_dot()

    def _leaf_shares(self, node: StrategyTreeNode, parent_share: float) -> list[tuple[StrategyTreeNode, float]]:
        if node.is_leaf():
            capped = min(parent_share, node.max_share) if node.max_share is not None else parent_share
            if capped < parent_share:
                self.throttles[node.name] = {"requested_share": parent_share, "applied_share": capped}
            return [(node, capped)]
        if not node.children:
            return []
        total = sum(c.weight for c in node.children)
        if total <= 0:
            return []
        out: list[tuple[StrategyTreeNode, float]] = []
        for child in node.children:
            out.extend(self._leaf_shares(child, parent_share * (child.weight / total)))
        return out

    def run(self, rebalance_freq: int = 0, monthly: bool = False, sma_days: int | None = None) -> dict[str, pd.DataFrame]:
        leaf_allocs = self._leaf_shares(self.root, 1.0)
        results: dict[str, pd.DataFrame] = {}
        self.leaf_weights = {leaf.name: w for leaf, w in leaf_allocs}
        self.attribution = {}
        allocated_share = float(sum(w for _, w in leaf_allocs))
        unallocated_share = max(0.0, 1.0 - allocated_share)

        balances: list[pd.DataFrame] = []
        for leaf, share in leaf_allocs:
            cap = round(self.initial_capital * share)
            saved_capital = leaf.engine.initial_capital
            leaf.engine.initial_capital = cap
            trade_log = leaf.engine.run(rebalance_freq=rebalance_freq, monthly=monthly, sma_days=sma_days)
            leaf.engine.initial_capital = saved_capital
            results[leaf.name] = trade_log
            self.attribution[leaf.name] = {
                "weight": share,
                "capital": cap,
            }
            b = leaf.engine.balance[["total capital"]].rename(columns={"total capital": f"{leaf.name}_capital"})
            balances.append(b)

        if balances:
            self.balance = pd.concat(balances, axis=1)
            cap_cols = [c for c in self.balance.columns if c.endswith("_capital")]
            self.balance["unallocated_cash"] = float(self.initial_capital * unallocated_share)
            self.balance["total capital"] = self.balance[cap_cols].sum(axis=1) + self.balance["unallocated_cash"]
            self.balance["% change"] = self.balance["total capital"].pct_change()
            self.balance["accumulated return"] = (1.0 + self.balance["% change"]).cumprod()
        else:
            self.balance = pd.DataFrame()

        return results
