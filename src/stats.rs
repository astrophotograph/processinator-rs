//! Order statistics matching numpy's conventions.
//!
//! The Python processinator leans on `np.median` (which averages the two
//! middle order statistics for even-length input) and `np.percentile`
//! (linear interpolation between order statistics). Matching those exactly
//! keeps this port numerically aligned with the reference implementation.
//! All selection is quickselect (`select_nth_unstable`), so each statistic
//! costs O(n) rather than a full sort.

use std::cmp::Ordering;

pub(crate) fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// Median with numpy semantics. Reorders `data`; returns 0.0 when empty.
pub(crate) fn median_in_place(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (below, mid_val, _) = data.select_nth_unstable_by(mid, cmp_f64);
    let hi = *mid_val;
    if n % 2 == 1 {
        hi
    } else {
        let lo = below.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// Median of a slice without disturbing it.
pub(crate) fn median_of(data: &[f64]) -> f64 {
    let mut buf = data.to_vec();
    median_in_place(&mut buf)
}

/// Single linearly-interpolated percentile (numpy default method).
/// NaNs are ignored, matching the Python code's `nanpercentile` fallback.
pub(crate) fn percentile_in_place(data: &mut [f64], p: f64) -> f64 {
    percentile_pair_in_place(data, p, p).0
}

/// Two percentiles from one buffer, sharing partition work. `lo_p <= hi_p`.
pub(crate) fn percentile_pair_in_place(data: &mut [f64], lo_p: f64, hi_p: f64) -> (f64, f64) {
    let data = drop_nans(data);
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (data[0], data[0]);
    }

    let rank = |p: f64| {
        let r = (p / 100.0).clamp(0.0, 1.0) * (n - 1) as f64;
        (r, r.floor() as usize, (r.ceil() as usize).min(n - 1))
    };
    let (rl, fl, cl) = rank(lo_p);
    let (rh, fh, ch) = rank(hi_p);

    let mut wanted = vec![fl, cl, fh, ch];
    wanted.sort_unstable();
    wanted.dedup();
    let values = order_stats(data, &wanted);
    let value_at = |k: usize| values[wanted.binary_search(&k).unwrap()];
    let interp = |r: f64, f: usize, c: usize| {
        let vf = value_at(f);
        vf + (r - f as f64) * (value_at(c) - vf)
    };
    (interp(rl, fl, cl), interp(rh, fh, ch))
}

/// Order statistics at the given sorted, deduplicated ranks. Each selection
/// runs on the still-unpartitioned tail of the previous one.
fn order_stats(data: &mut [f64], ranks: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ranks.len());
    let mut offset = 0usize;
    let mut slice = data;
    for &r in ranks {
        let (_, value, rest) = slice.select_nth_unstable_by(r - offset, cmp_f64);
        out.push(*value);
        offset = r + 1;
        slice = rest;
    }
    out
}

/// Compact non-NaN values to the front and return that prefix.
fn drop_nans(data: &mut [f64]) -> &mut [f64] {
    let mut n = data.len();
    let mut i = 0;
    while i < n {
        if data[i].is_nan() {
            n -= 1;
            data.swap(i, n);
        } else {
            i += 1;
        }
    }
    &mut data[..n]
}
