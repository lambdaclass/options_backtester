//! PyO3 bindings for execution models: cost, fill, signal selection, risk.
//!
//! Exposes flat functions that call into `ob_core` implementations,
//! allowing Python classes to delegate computation to Rust.

use pyo3::prelude::*;

use ob_core::cost_model::CostModel;
use ob_core::fill_model::FillModel;
use ob_core::risk::RiskConstraint;
use ob_core::types::Greeks;

// ---------------------------------------------------------------------------
// Cost models
// ---------------------------------------------------------------------------

/// Compute option trade commission via Rust cost model.
///
/// `model_type`: "PerContract" or "Tiered"
/// `tiers`: list of (max_contracts, rate) pairs (only used for Tiered)
#[pyfunction]
#[pyo3(signature = (model_type, rate, stock_rate, tiers, price, quantity, spc))]
pub fn rust_option_cost(
    model_type: &str,
    rate: f64,
    stock_rate: f64,
    tiers: Vec<(i64, f64)>,
    price: f64,
    quantity: f64,
    spc: i64,
) -> PyResult<f64> {
    let model = match model_type {
        "PerContract" => CostModel::PerContract { rate, stock_rate },
        "Tiered" => CostModel::Tiered { tiers, stock_rate },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown cost model type: {other}"),
            ))
        }
    };
    Ok(model.option_cost(price, quantity, spc))
}

/// Compute stock trade commission via Rust cost model.
#[pyfunction]
#[pyo3(signature = (model_type, rate, stock_rate, tiers, price, quantity))]
pub fn rust_stock_cost(
    model_type: &str,
    rate: f64,
    stock_rate: f64,
    tiers: Vec<(i64, f64)>,
    price: f64,
    quantity: f64,
) -> PyResult<f64> {
    let model = match model_type {
        "PerContract" => CostModel::PerContract { rate, stock_rate },
        "Tiered" => CostModel::Tiered { tiers, stock_rate },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown cost model type: {other}"),
            ))
        }
    };
    Ok(model.stock_cost(price, quantity))
}

// ---------------------------------------------------------------------------
// Fill models
// ---------------------------------------------------------------------------

/// Compute fill price via Rust fill model.
///
/// `model_type`: "VolumeAware"
/// `threshold`: full_volume_threshold (only used for VolumeAware)
/// `volume`: None means missing volume data
#[pyfunction]
#[pyo3(signature = (model_type, threshold, bid, ask, volume, is_buy))]
pub fn rust_fill_price(
    model_type: &str,
    threshold: i64,
    bid: f64,
    ask: f64,
    volume: Option<f64>,
    is_buy: bool,
) -> PyResult<f64> {
    let model = match model_type {
        "VolumeAware" => FillModel::VolumeAware {
            full_volume_threshold: threshold,
        },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown fill model type: {other}"),
            ))
        }
    };
    Ok(model.fill_price(bid, ask, volume, is_buy))
}

// ---------------------------------------------------------------------------
// Signal selectors
// ---------------------------------------------------------------------------

/// Find the index of the value nearest to `target` in a list of f64.
/// NaN values are skipped. Returns 0 for empty input.
#[pyfunction]
#[pyo3(signature = (values, target))]
pub fn rust_nearest_delta_index(values: Vec<f64>, target: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_diff = f64::MAX;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        let diff = (v - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }
    best_idx
}

/// Find the index of the maximum value in a list of f64.
/// NaN values are skipped. Returns 0 for empty input.
#[pyfunction]
#[pyo3(signature = (values,))]
pub fn rust_max_value_index(values: Vec<f64>) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = f64::MIN;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Risk constraints
// ---------------------------------------------------------------------------

/// Check a single risk constraint via Rust.
///
/// `constraint_type`: "MaxDelta", "MaxVega", or "MaxDrawdown"
/// `limit`: the constraint limit (delta/vega limit, or max_dd_pct)
/// `current_greeks`: [delta, gamma, theta, vega]
/// `proposed_greeks`: [delta, gamma, theta, vega]
#[pyfunction]
#[pyo3(signature = (constraint_type, limit, current_greeks, proposed_greeks, portfolio_value, peak_value))]
pub fn rust_risk_check(
    constraint_type: &str,
    limit: f64,
    current_greeks: [f64; 4],
    proposed_greeks: [f64; 4],
    portfolio_value: f64,
    peak_value: f64,
) -> PyResult<bool> {
    let constraint = match constraint_type {
        "MaxDelta" => RiskConstraint::MaxDelta { limit },
        "MaxVega" => RiskConstraint::MaxVega { limit },
        "MaxDrawdown" => RiskConstraint::MaxDrawdown { max_dd_pct: limit },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown risk constraint type: {other}"),
            ))
        }
    };
    let current = Greeks::new(
        current_greeks[0],
        current_greeks[1],
        current_greeks[2],
        current_greeks[3],
    );
    let proposed = Greeks::new(
        proposed_greeks[0],
        proposed_greeks[1],
        proposed_greeks[2],
        proposed_greeks[3],
    );
    Ok(constraint.check(&current, &proposed, portfolio_value, peak_value))
}
