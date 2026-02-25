//! Risk management — constraints checked before entering positions.
//!
//! Mirrors Python's `options_portfolio_backtester.portfolio.risk`.

use crate::types::Greeks;

#[derive(Debug, Clone)]
pub enum RiskConstraint {
    /// Reject trades that would push portfolio delta beyond a limit.
    MaxDelta { limit: f64 },
    /// Reject trades that would push portfolio vega beyond a limit.
    MaxVega { limit: f64 },
    /// Reject new entries if portfolio drawdown exceeds a threshold.
    MaxDrawdown { max_dd_pct: f64 },
}

impl RiskConstraint {
    /// Check whether a proposed trade is allowed.
    ///
    /// Returns true if the trade passes this constraint.
    pub fn check(
        &self,
        current_greeks: &Greeks,
        proposed_greeks: &Greeks,
        portfolio_value: f64,
        peak_value: f64,
    ) -> bool {
        match self {
            RiskConstraint::MaxDelta { limit } => {
                let new_delta = current_greeks.delta + proposed_greeks.delta;
                new_delta.abs() <= *limit
            }
            RiskConstraint::MaxVega { limit } => {
                let new_vega = current_greeks.vega + proposed_greeks.vega;
                new_vega.abs() <= *limit
            }
            RiskConstraint::MaxDrawdown { max_dd_pct } => {
                if peak_value <= 0.0 {
                    return true;
                }
                let dd = (peak_value - portfolio_value) / peak_value;
                dd < *max_dd_pct
            }
        }
    }
}

/// Check all constraints. Returns (allowed, failing_constraint_index).
pub fn check_all(
    constraints: &[RiskConstraint],
    current_greeks: &Greeks,
    proposed_greeks: &Greeks,
    portfolio_value: f64,
    peak_value: f64,
) -> bool {
    constraints.iter().all(|c| c.check(current_greeks, proposed_greeks, portfolio_value, peak_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_delta_allows() {
        let c = RiskConstraint::MaxDelta { limit: 100.0 };
        let current = Greeks::new(50.0, 0.0, 0.0, 0.0);
        let proposed = Greeks::new(30.0, 0.0, 0.0, 0.0);
        assert!(c.check(&current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn max_delta_rejects() {
        let c = RiskConstraint::MaxDelta { limit: 100.0 };
        let current = Greeks::new(80.0, 0.0, 0.0, 0.0);
        let proposed = Greeks::new(30.0, 0.0, 0.0, 0.0);
        assert!(!c.check(&current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn max_delta_negative() {
        let c = RiskConstraint::MaxDelta { limit: 100.0 };
        let current = Greeks::new(-80.0, 0.0, 0.0, 0.0);
        let proposed = Greeks::new(-30.0, 0.0, 0.0, 0.0);
        assert!(!c.check(&current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn max_vega_allows() {
        let c = RiskConstraint::MaxVega { limit: 50.0 };
        let current = Greeks::new(0.0, 0.0, 0.0, 20.0);
        let proposed = Greeks::new(0.0, 0.0, 0.0, 10.0);
        assert!(c.check(&current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn max_vega_rejects() {
        let c = RiskConstraint::MaxVega { limit: 50.0 };
        let current = Greeks::new(0.0, 0.0, 0.0, 40.0);
        let proposed = Greeks::new(0.0, 0.0, 0.0, 20.0);
        assert!(!c.check(&current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn max_drawdown_allows() {
        let c = RiskConstraint::MaxDrawdown { max_dd_pct: 0.20 };
        let g = Greeks::default();
        // 10% drawdown from peak
        assert!(c.check(&g, &g, 900_000.0, 1_000_000.0));
    }

    #[test]
    fn max_drawdown_rejects() {
        let c = RiskConstraint::MaxDrawdown { max_dd_pct: 0.20 };
        let g = Greeks::default();
        // 25% drawdown from peak
        assert!(!c.check(&g, &g, 750_000.0, 1_000_000.0));
    }

    #[test]
    fn max_drawdown_zero_peak() {
        let c = RiskConstraint::MaxDrawdown { max_dd_pct: 0.20 };
        let g = Greeks::default();
        assert!(c.check(&g, &g, 100.0, 0.0));
    }

    #[test]
    fn check_all_passes() {
        let constraints = vec![
            RiskConstraint::MaxDelta { limit: 100.0 },
            RiskConstraint::MaxVega { limit: 50.0 },
        ];
        let current = Greeks::new(30.0, 0.0, 0.0, 10.0);
        let proposed = Greeks::new(10.0, 0.0, 0.0, 5.0);
        assert!(super::check_all(&constraints, &current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn check_all_fails_one() {
        let constraints = vec![
            RiskConstraint::MaxDelta { limit: 100.0 },
            RiskConstraint::MaxVega { limit: 50.0 },
        ];
        let current = Greeks::new(30.0, 0.0, 0.0, 40.0);
        let proposed = Greeks::new(10.0, 0.0, 0.0, 20.0);
        // Delta OK (40), but Vega fails (60 > 50)
        assert!(!super::check_all(&constraints, &current, &proposed, 1_000_000.0, 1_000_000.0));
    }

    #[test]
    fn check_all_empty_passes() {
        let g = Greeks::default();
        assert!(super::check_all(&[], &g, &g, 1_000_000.0, 1_000_000.0));
    }
}
