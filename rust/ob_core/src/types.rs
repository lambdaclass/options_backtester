/// Core domain types mirroring Python's options_portfolio_backtester.core.types.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Buy,
    Sell,
}

impl Direction {
    pub fn sign(self) -> f64 {
        match self {
            Direction::Buy => -1.0,
            Direction::Sell => 1.0,
        }
    }

    pub fn price_column(self) -> &'static str {
        match self {
            Direction::Buy => "ask",
            Direction::Sell => "bid",
        }
    }

    pub fn invert(self) -> Direction {
        match self {
            Direction::Buy => Direction::Sell,
            Direction::Sell => Direction::Buy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn as_str(self) -> &'static str {
        match self {
            OptionType::Call => "call",
            OptionType::Put => "put",
        }
    }
}

/// Configuration for a single strategy leg.
#[derive(Debug, Clone)]
pub struct LegConfig {
    pub name: String,
    pub option_type: OptionType,
    pub direction: Direction,
    pub entry_filter_query: Option<String>,
    pub exit_filter_query: Option<String>,
    pub entry_sort_col: Option<String>,
    pub entry_sort_asc: bool,
    /// Per-leg signal selector override (None = use engine-level selector).
    pub signal_selector: Option<crate::signal_selector::SignalSelector>,
    /// Per-leg fill model override (None = use engine-level fill model).
    pub fill_model: Option<crate::fill_model::FillModel>,
}

/// Aggregated Greeks for a position or portfolio.
#[derive(Debug, Clone, Copy, Default)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64) -> Self {
        Self { delta, gamma, theta, vega }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            delta: self.delta * s,
            gamma: self.gamma * s,
            theta: self.theta * s,
            vega: self.vega * s,
        }
    }
}

impl std::ops::Add for Greeks {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            delta: self.delta + rhs.delta,
            gamma: self.gamma + rhs.gamma,
            theta: self.theta + rhs.theta,
            vega: self.vega + rhs.vega,
        }
    }
}

impl std::ops::AddAssign for Greeks {
    fn add_assign(&mut self, rhs: Self) {
        self.delta += rhs.delta;
        self.gamma += rhs.gamma;
        self.theta += rhs.theta;
        self.vega += rhs.vega;
    }
}

/// Balance row for a single date.
#[derive(Debug, Clone, Default)]
pub struct BalanceRow {
    pub cash: f64,
    pub options_qty: f64,
    pub calls_capital: f64,
    pub puts_capital: f64,
    pub stocks_qty: f64,
    pub stock_holdings: Vec<(String, f64)>,
    pub stock_qtys: Vec<(String, f64)>,
}

/// Stats summary for a backtest.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: u32,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub total_trades: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_sign() {
        assert_eq!(Direction::Buy.sign(), -1.0);
        assert_eq!(Direction::Sell.sign(), 1.0);
    }

    #[test]
    fn direction_invert() {
        assert_eq!(Direction::Buy.invert(), Direction::Sell);
        assert_eq!(Direction::Sell.invert(), Direction::Buy);
    }

    #[test]
    fn greeks_add() {
        let a = Greeks::new(1.0, 2.0, 3.0, 4.0);
        let b = Greeks::new(0.5, 0.5, 0.5, 0.5);
        let c = a + b;
        assert!((c.delta - 1.5).abs() < 1e-10);
        assert!((c.gamma - 2.5).abs() < 1e-10);
    }

    #[test]
    fn greeks_scale() {
        let g = Greeks::new(1.0, 2.0, 3.0, 4.0).scale(2.0);
        assert!((g.delta - 2.0).abs() < 1e-10);
        assert!((g.vega - 8.0).abs() < 1e-10);
    }
}
