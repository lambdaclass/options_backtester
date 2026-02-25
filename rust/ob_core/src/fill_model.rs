/// Fill models — determine the execution price for trades.
///
/// Mirrors Python's `options_backtester.execution.fill_model`.

#[derive(Debug, Clone)]
pub enum FillModel {
    /// Fill at bid (sell) or ask (buy) — matches original behavior.
    MarketAtBidAsk,
    /// Fill at the midpoint of bid and ask.
    MidPrice,
    /// Fill price adjusts for volume impact. Low volume pushes toward mid.
    VolumeAware { full_volume_threshold: i64 },
}

impl Default for FillModel {
    fn default() -> Self {
        FillModel::MarketAtBidAsk
    }
}

impl FillModel {
    /// Compute fill price given bid, ask, volume, and whether this is a buy.
    ///
    /// `is_buy`: true for BUY direction (fills at ask), false for SELL (fills at bid).
    pub fn fill_price(&self, bid: f64, ask: f64, volume: Option<f64>, is_buy: bool) -> f64 {
        match self {
            FillModel::MarketAtBidAsk => {
                if is_buy { ask } else { bid }
            }
            FillModel::MidPrice => {
                (bid + ask) / 2.0
            }
            FillModel::VolumeAware { full_volume_threshold } => {
                let mid = (bid + ask) / 2.0;
                let target = if is_buy { ask } else { bid };
                let vol = volume.unwrap_or(*full_volume_threshold as f64);

                if vol >= *full_volume_threshold as f64 {
                    return target;
                }

                let ratio = vol / *full_volume_threshold as f64;
                mid + ratio * (target - mid)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn market_at_bid_ask_buy() {
        let m = FillModel::MarketAtBidAsk;
        assert!((m.fill_price(9.0, 10.0, None, true) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn market_at_bid_ask_sell() {
        let m = FillModel::MarketAtBidAsk;
        assert!((m.fill_price(9.0, 10.0, None, false) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn mid_price() {
        let m = FillModel::MidPrice;
        assert!((m.fill_price(9.0, 11.0, None, true) - 10.0).abs() < 1e-10);
        assert!((m.fill_price(9.0, 11.0, None, false) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn volume_aware_full_volume() {
        let m = FillModel::VolumeAware { full_volume_threshold: 100 };
        // At or above threshold, same as market
        assert!((m.fill_price(9.0, 10.0, Some(100.0), true) - 10.0).abs() < 1e-10);
        assert!((m.fill_price(9.0, 10.0, Some(200.0), false) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn volume_aware_zero_volume() {
        let m = FillModel::VolumeAware { full_volume_threshold: 100 };
        // At volume=0, fill at mid
        let mid = (9.0 + 10.0) / 2.0;
        assert!((m.fill_price(9.0, 10.0, Some(0.0), true) - mid).abs() < 1e-10);
        assert!((m.fill_price(9.0, 10.0, Some(0.0), false) - mid).abs() < 1e-10);
    }

    #[test]
    fn volume_aware_half_volume() {
        let m = FillModel::VolumeAware { full_volume_threshold: 100 };
        let mid = (9.0 + 10.0) / 2.0;
        // At 50% volume: mid + 0.5 * (ask - mid) = 9.5 + 0.25 = 9.75
        let expected_buy = mid + 0.5 * (10.0 - mid);
        assert!((m.fill_price(9.0, 10.0, Some(50.0), true) - expected_buy).abs() < 1e-10);
        let expected_sell = mid + 0.5 * (9.0 - mid);
        assert!((m.fill_price(9.0, 10.0, Some(50.0), false) - expected_sell).abs() < 1e-10);
    }

    #[test]
    fn volume_aware_no_volume_data() {
        let m = FillModel::VolumeAware { full_volume_threshold: 100 };
        // Missing volume defaults to threshold -> market price
        assert!((m.fill_price(9.0, 10.0, None, true) - 10.0).abs() < 1e-10);
    }
}
