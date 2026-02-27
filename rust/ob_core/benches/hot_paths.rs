use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars::prelude::*;

use ob_core::entries::{compute_entry_qty, compute_leg_entries};
use ob_core::exits::threshold_exit_mask;
use ob_core::filter::CompiledFilter;
use ob_core::inventory::join_inventory_to_market;
use ob_core::stats::compute_stats;
use ob_core::types::Direction;

fn make_options_df(n: usize) -> DataFrame {
    let contracts: Vec<String> = (0..n).map(|i| format!("SPX_{i}")).collect();
    let underlyings: Vec<&str> = vec!["SPX"; n];
    let types: Vec<&str> = (0..n)
        .map(|i| if i % 2 == 0 { "put" } else { "call" })
        .collect();
    let expirations: Vec<&str> = vec!["2024-06-01"; n];
    let strikes: Vec<f64> = (0..n).map(|i| 3800.0 + i as f64 * 5.0).collect();
    let asks: Vec<f64> = (0..n).map(|i| 1.0 + (i % 50) as f64 * 0.5).collect();
    let bids: Vec<f64> = asks.iter().map(|a| a * 0.95).collect();
    let dtes: Vec<i32> = (0..n).map(|i| 30 + (i % 180) as i32).collect();

    DataFrame::new(vec![
        Column::new("optionroot".into(), contracts),
        Column::new("underlying".into(), underlyings),
        Column::new("type".into(), types),
        Column::new("expiration".into(), expirations),
        Column::new("strike".into(), strikes),
        Column::new("ask".into(), asks),
        Column::new("bid".into(), bids),
        Column::new("dte".into(), dtes),
    ])
    .unwrap()
}

fn bench_inventory_join(c: &mut Criterion) {
    let opts = make_options_df(10_000);
    let n_inv = 50;
    let contracts: Vec<String> = (0..n_inv).map(|i| format!("SPX_{i}")).collect();
    let qtys: Vec<f64> = vec![10.0; n_inv];
    let types: Vec<String> = (0..n_inv)
        .map(|i| {
            if i % 2 == 0 {
                "put".into()
            } else {
                "call".into()
            }
        })
        .collect();

    let underlyings: Vec<String> = vec!["SPX".into(); n_inv];
    let strikes: Vec<f64> = (0..n_inv).map(|i| 3800.0 + i as f64 * 5.0).collect();

    c.bench_function("inventory_join_50x10k", |b| {
        b.iter(|| {
            let result = join_inventory_to_market(
                black_box(&contracts),
                black_box(&qtys),
                black_box(&types),
                black_box(&underlyings),
                black_box(&strikes),
                black_box(&opts),
                None,
                "optionroot",
                "quotedate",
                "bid",
                None,
                None,
                Direction::Buy,
                100,
            )
            .unwrap();
            black_box(result.height());
        });
    });
}

fn bench_filter_compile_and_apply(c: &mut Criterion) {
    let df = make_options_df(10_000);

    let filter = CompiledFilter::new(
        "(type == 'put') & (ask > 0) & (underlying == 'SPX') & (dte >= 60) & (dte <= 120)",
    )
    .unwrap();

    c.bench_function("filter_apply_10k", |b| {
        b.iter(|| {
            let result = filter.apply(black_box(&df)).unwrap();
            black_box(result.height());
        });
    });
}

fn bench_filter_compile(c: &mut Criterion) {
    c.bench_function("filter_compile", |b| {
        b.iter(|| {
            let f = CompiledFilter::new(black_box(
                "(type == 'put') & (ask > 0) & (underlying == 'SPX') & (dte >= 60) & (dte <= 120)",
            ))
            .unwrap();
            black_box(&f);
        });
    });
}

fn bench_entry_computation(c: &mut Criterion) {
    let opts = make_options_df(10_000);
    let held: Vec<String> = (0..10).map(|i| format!("SPX_{i}")).collect();
    let filter = CompiledFilter::new("(type == 'put') & (ask > 0) & (dte >= 60)").unwrap();

    c.bench_function("entry_compute_10k", |b| {
        b.iter(|| {
            let result = compute_leg_entries(
                black_box(&opts),
                black_box(&held),
                black_box(&filter),
                "optionroot",
                "ask",
                Some("strike"),
                true,
                100,
                false,
            )
            .unwrap();
            black_box(result.height());
        });
    });
}

fn bench_exit_mask(c: &mut Criterion) {
    let n = 1000;
    let entries: Vec<f64> = (0..n).map(|i| 100.0 + (i % 50) as f64).collect();
    let currents: Vec<f64> = (0..n).map(|i| 80.0 + (i % 80) as f64).collect();

    let entry_s = Series::new("entry".into(), &entries);
    let current_s = Series::new("current".into(), &currents);

    c.bench_function("exit_mask_1k", |b| {
        b.iter(|| {
            let mask = threshold_exit_mask(
                black_box(&entry_s),
                black_box(&current_s),
                Some(0.5),
                Some(0.2),
            )
            .unwrap();
            black_box(mask.len());
        });
    });
}

fn bench_stats_computation(c: &mut Criterion) {
    let n = 2520; // ~10 years of trading days
    let returns: Vec<f64> = (0..n).map(|i| ((i as f64 * 0.1).sin()) * 0.02).collect();
    let pnls: Vec<f64> = (0..100)
        .map(|i| if i % 3 == 0 { -50.0 } else { 100.0 })
        .collect();

    c.bench_function("stats_10yr", |b| {
        b.iter(|| {
            let s = compute_stats(black_box(&returns), black_box(&pnls), 0.02);
            black_box(s);
        });
    });
}

fn bench_entry_qty(c: &mut Criterion) {
    let n = 5000;
    let costs: Vec<f64> = (0..n).map(|i| 50.0 + (i % 200) as f64).collect();
    let series = Series::new("cost".into(), &costs);

    c.bench_function("entry_qty_5k", |b| {
        b.iter(|| {
            let qty = compute_entry_qty(black_box(&series), 1_000_000.0).unwrap();
            black_box(qty.len());
        });
    });
}

criterion_group!(
    benches,
    bench_inventory_join,
    bench_filter_compile,
    bench_filter_compile_and_apply,
    bench_entry_computation,
    bench_exit_mask,
    bench_stats_computation,
    bench_entry_qty,
);
criterion_main!(benches);
