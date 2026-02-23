use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars::prelude::*;

use ob_core::filter::CompiledFilter;
use ob_core::stats::compute_stats;

fn bench_filter_compile_and_apply(c: &mut Criterion) {
    let n = 10_000;
    let types: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "put" } else { "call" }).collect();
    let asks: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.1).collect();
    let dtes: Vec<i64> = (0..n).map(|i| (i % 365) as i64).collect();
    let underlyings: Vec<&str> = (0..n).map(|i| if i % 3 == 0 { "SPX" } else { "AAPL" }).collect();

    let df = df!(
        "type" => types,
        "ask" => asks,
        "dte" => dtes,
        "underlying" => underlyings,
    )
    .unwrap();

    let filter = CompiledFilter::new("(type == 'put') & (ask > 0) & (underlying == 'SPX') & (dte >= 60) & (dte <= 120)").unwrap();

    c.bench_function("filter_apply_10k", |b| {
        b.iter(|| {
            let result = filter.apply(black_box(&df)).unwrap();
            black_box(result.height());
        });
    });
}

fn bench_stats_computation(c: &mut Criterion) {
    let n = 2520; // ~10 years of trading days
    let returns: Vec<f64> = (0..n).map(|i| ((i as f64 * 0.1).sin()) * 0.02).collect();
    let pnls: Vec<f64> = (0..100).map(|i| if i % 3 == 0 { -50.0 } else { 100.0 }).collect();

    c.bench_function("stats_10yr", |b| {
        b.iter(|| {
            let s = compute_stats(black_box(&returns), black_box(&pnls), 0.02);
            black_box(s);
        });
    });
}

criterion_group!(benches, bench_filter_compile_and_apply, bench_stats_computation);
criterion_main!(benches);
