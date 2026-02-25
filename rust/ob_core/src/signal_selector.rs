/// Signal selectors — choose which contract to trade from candidates.
///
/// Mirrors Python's `options_backtester.execution.signal_selector`.

use polars::prelude::*;

#[derive(Debug, Clone)]
pub enum SignalSelector {
    /// Pick the first row (default — matches original iloc[0] behavior).
    FirstMatch,
    /// Pick the contract whose column value is closest to `target`.
    NearestDelta { target: f64, column: String },
    /// Pick the contract with the highest value in `column`.
    MaxOpenInterest { column: String },
}

impl Default for SignalSelector {
    fn default() -> Self {
        SignalSelector::FirstMatch
    }
}

impl SignalSelector {
    /// Extra columns this selector needs preserved through the entry pipeline.
    pub fn column_requirements(&self) -> Vec<&str> {
        match self {
            SignalSelector::FirstMatch => vec![],
            SignalSelector::NearestDelta { column, .. } => vec![column.as_str()],
            SignalSelector::MaxOpenInterest { column } => vec![column.as_str()],
        }
    }

    /// Select one row index from a DataFrame of candidates. Returns 0-based row index.
    pub fn select_index(&self, candidates: &DataFrame) -> usize {
        if candidates.height() == 0 {
            return 0;
        }
        match self {
            SignalSelector::FirstMatch => 0,
            SignalSelector::NearestDelta { target, column } => {
                match candidates.column(column).ok() {
                    Some(col) => {
                        match col.f64() {
                            Ok(ca) => {
                                let mut best_idx = 0;
                                let mut best_diff = f64::MAX;
                                for (i, val) in ca.into_iter().enumerate() {
                                    if let Some(v) = val {
                                        let diff = (v - target).abs();
                                        if diff < best_diff {
                                            best_diff = diff;
                                            best_idx = i;
                                        }
                                    }
                                }
                                best_idx
                            }
                            Err(_) => 0,
                        }
                    }
                    None => 0, // column not found, fall back to first
                }
            }
            SignalSelector::MaxOpenInterest { column } => {
                match candidates.column(column).ok() {
                    Some(col) => {
                        match col.f64() {
                            Ok(ca) => {
                                let mut best_idx = 0;
                                let mut best_val = f64::MIN;
                                for (i, val) in ca.into_iter().enumerate() {
                                    if let Some(v) = val {
                                        if v > best_val {
                                            best_val = v;
                                            best_idx = i;
                                        }
                                    }
                                }
                                best_idx
                            }
                            Err(_) => {
                                // Try i64 column
                                match col.i64() {
                                    Ok(ca) => {
                                        let mut best_idx = 0;
                                        let mut best_val = i64::MIN;
                                        for (i, val) in ca.into_iter().enumerate() {
                                            if let Some(v) = val {
                                                if v > best_val {
                                                    best_val = v;
                                                    best_idx = i;
                                                }
                                            }
                                        }
                                        best_idx
                                    }
                                    Err(_) => 0,
                                }
                            }
                        }
                    }
                    None => 0,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candidates() -> DataFrame {
        df!(
            "contract" => &["A", "B", "C"],
            "cost" => &[100.0, 200.0, 150.0],
            "delta" => &[-0.20, -0.30, -0.45],
            "openinterest" => &[500.0, 1200.0, 800.0],
        ).unwrap()
    }

    #[test]
    fn first_match() {
        let df = sample_candidates();
        let sel = SignalSelector::FirstMatch;
        assert_eq!(sel.select_index(&df), 0);
    }

    #[test]
    fn nearest_delta() {
        let df = sample_candidates();
        let sel = SignalSelector::NearestDelta {
            target: -0.30,
            column: "delta".into(),
        };
        assert_eq!(sel.select_index(&df), 1); // B has delta=-0.30
    }

    #[test]
    fn nearest_delta_between() {
        let df = sample_candidates();
        let sel = SignalSelector::NearestDelta {
            target: -0.35,
            column: "delta".into(),
        };
        // -0.30 is 0.05 away, -0.45 is 0.10 away → B wins
        assert_eq!(sel.select_index(&df), 1);
    }

    #[test]
    fn max_open_interest() {
        let df = sample_candidates();
        let sel = SignalSelector::MaxOpenInterest {
            column: "openinterest".into(),
        };
        assert_eq!(sel.select_index(&df), 1); // B has OI=1200
    }

    #[test]
    fn missing_column_falls_back() {
        let df = sample_candidates();
        let sel = SignalSelector::NearestDelta {
            target: -0.30,
            column: "nonexistent".into(),
        };
        assert_eq!(sel.select_index(&df), 0);
    }

    #[test]
    fn column_requirements_check() {
        assert!(SignalSelector::FirstMatch.column_requirements().is_empty());
        let sel = SignalSelector::NearestDelta { target: 0.0, column: "delta".into() };
        assert_eq!(sel.column_requirements(), vec!["delta"]);
        let sel = SignalSelector::MaxOpenInterest { column: "oi".into() };
        assert_eq!(sel.column_requirements(), vec!["oi"]);
    }
}
