//! Position sizers — how many contracts an entry buys or sells.
//!
//! Mirrors Python's `options_portfolio_backtester.execution.sizer`. Every
//! sizer is capped by the entry's budget (the capital available to it).

#[derive(Debug, Clone, Default)]
pub enum Sizer {
    /// As many contracts as the budget allows.
    #[default]
    CapitalBased,
    /// A fixed number of contracts (fewer if the budget cannot pay for them).
    FixedQuantity { quantity: f64 },
    /// Contracts worth up to `amount` dollars.
    FixedDollar { amount: f64 },
    /// Contracts worth up to `pct` of total portfolio value.
    PercentOfPortfolio { pct: f64 },
}

impl Sizer {
    /// Contracts to trade at `cost_per_contract` (absolute, dollars) given
    /// the entry's `budget` and the portfolio's `total_capital`.
    pub fn size(&self, cost_per_contract: f64, budget: f64, total_capital: f64) -> f64 {
        let cost = cost_per_contract.abs();
        if cost == 0.0 || budget <= 0.0 {
            return 0.0;
        }
        let affordable = (budget / cost).floor();
        match self {
            Sizer::CapitalBased => affordable,
            Sizer::FixedQuantity { quantity } => quantity.min(affordable),
            Sizer::FixedDollar { amount } => (amount.min(budget) / cost).floor(),
            Sizer::PercentOfPortfolio { pct } => ((pct * total_capital).min(budget) / cost).floor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizers_cap_at_the_budget() {
        let (cost, budget, total) = (300.0, 10_000.0, 1_000_000.0);
        assert_eq!(Sizer::CapitalBased.size(cost, budget, total), 33.0);
        assert_eq!(Sizer::FixedQuantity { quantity: 1.0 }.size(cost, budget, total), 1.0);
        assert_eq!(Sizer::FixedQuantity { quantity: 50.0 }.size(cost, budget, total), 33.0);
        assert_eq!(Sizer::FixedDollar { amount: 1_000.0 }.size(cost, budget, total), 3.0);
        assert_eq!(Sizer::PercentOfPortfolio { pct: 0.002 }.size(cost, budget, total), 6.0);
        assert_eq!(Sizer::PercentOfPortfolio { pct: 0.5 }.size(cost, budget, total), 33.0);
        assert_eq!(Sizer::CapitalBased.size(0.0, budget, total), 0.0);
    }
}
