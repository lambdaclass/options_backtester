//! Transaction cost models for options and stocks.
//!
//! Mirrors Python's `options_portfolio_backtester.execution.cost_model`.

#[derive(Debug, Clone, Default)]
pub enum CostModel {
    /// Zero transaction costs.
    #[default]
    NoCosts,
    /// Fixed per-contract commission (e.g., $0.65/contract for IBKR).
    PerContract { rate: f64, stock_rate: f64 },
    /// Tiered commission schedule with volume discounts.
    /// Tiers are (max_contracts, rate) pairs sorted by max_contracts ascending.
    Tiered { tiers: Vec<(i64, f64)>, stock_rate: f64 },
}

impl CostModel {
    /// Compute option trade commission.
    pub fn option_cost(&self, _price: f64, quantity: f64, _spc: i64) -> f64 {
        let qty = quantity.abs();
        match self {
            CostModel::NoCosts => 0.0,
            CostModel::PerContract { rate, .. } => rate * qty,
            CostModel::Tiered { tiers, .. } => {
                let mut total = 0.0;
                let mut remaining = qty;
                let mut prev_bound: i64 = 0;
                for &(max_qty, rate) in tiers {
                    let tier_qty = remaining.min((max_qty - prev_bound) as f64);
                    if tier_qty <= 0.0 {
                        prev_bound = max_qty;
                        continue;
                    }
                    total += tier_qty * rate;
                    remaining -= tier_qty;
                    prev_bound = max_qty;
                    if remaining <= 0.0 {
                        break;
                    }
                }
                if remaining > 0.0 {
                    if let Some(&(_, last_rate)) = tiers.last() {
                        total += remaining * last_rate;
                    }
                }
                total
            }
        }
    }

    /// Compute stock trade commission.
    pub fn stock_cost(&self, _price: f64, quantity: f64) -> f64 {
        let qty = quantity.abs();
        match self {
            CostModel::NoCosts => 0.0,
            CostModel::PerContract { stock_rate, .. } => stock_rate * qty,
            CostModel::Tiered { stock_rate, .. } => stock_rate * qty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_costs() {
        let m = CostModel::NoCosts;
        assert_eq!(m.option_cost(10.0, 5.0, 100), 0.0);
        assert_eq!(m.stock_cost(150.0, 100.0), 0.0);
    }

    #[test]
    fn per_contract() {
        let m = CostModel::PerContract { rate: 0.65, stock_rate: 0.005 };
        assert!((m.option_cost(10.0, 10.0, 100) - 6.5).abs() < 1e-10);
        assert!((m.option_cost(10.0, -10.0, 100) - 6.5).abs() < 1e-10);
        assert!((m.stock_cost(150.0, 100.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn tiered_within_first_tier() {
        let m = CostModel::Tiered {
            tiers: vec![(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)],
            stock_rate: 0.005,
        };
        // 100 contracts, all in first tier
        assert!((m.option_cost(10.0, 100.0, 100) - 65.0).abs() < 1e-10);
    }

    #[test]
    fn tiered_spanning_tiers() {
        let m = CostModel::Tiered {
            tiers: vec![(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)],
            stock_rate: 0.005,
        };
        // 15000 contracts: 10000 * 0.65 + 5000 * 0.50
        let expected = 10_000.0 * 0.65 + 5_000.0 * 0.50;
        assert!((m.option_cost(10.0, 15_000.0, 100) - expected).abs() < 1e-10);
    }

    #[test]
    fn tiered_beyond_all() {
        let m = CostModel::Tiered {
            tiers: vec![(10_000, 0.65), (50_000, 0.50), (100_000, 0.25)],
            stock_rate: 0.005,
        };
        // 120_000: 10k*0.65 + 40k*0.50 + 50k*0.25 + 20k*0.25
        let expected = 10_000.0 * 0.65 + 40_000.0 * 0.50 + 50_000.0 * 0.25 + 20_000.0 * 0.25;
        assert!((m.option_cost(10.0, 120_000.0, 100) - expected).abs() < 1e-10);
    }

    #[test]
    fn tiered_stock_cost() {
        let m = CostModel::Tiered {
            tiers: vec![(10_000, 0.65)],
            stock_rate: 0.005,
        };
        assert!((m.stock_cost(150.0, 100.0) - 0.5).abs() < 1e-10);
    }
}
