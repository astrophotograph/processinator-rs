//! Parity tests for the display-stretch contract.
//!
//! Astra's WebGL preview applies [`mtf_display_solution`] per pixel in a
//! fragment shader instead of running the CPU pipeline. These tests pin the
//! two halves of that contract:
//!
//! 1. Applying the solution the way the shader does (shadow/scale clamp →
//!    MTF → green removal → saturation) reproduces [`process`] exactly.
//! 2. The frontend recomputes the solution per slider change from
//!    serialized per-channel stats plus a histogram of the reference
//!    channel (the post-subtraction median can't be had analytically).
//!    That reconstruction — transcribed here — must match the exact
//!    solution to display precision.

mod common;

use processinator::{
    mtf_display_solution, mtf_stats_channel, mtf_stats_linked, prepare, process, remove_green,
    saturate, Image, MtfSolution, MtfStats, PipelineConfig, StretchAlgorithm,
};

const BG: f64 = 0.15;
const SIGMA: f64 = 3.0;

/// Number of histogram bins the astra payload uses.
const HIST_BINS: usize = 1 << 16;

fn config(bg_percent: f64, sigma: f64) -> PipelineConfig {
    PipelineConfig {
        stretch: StretchAlgorithm::Mtf {
            bg_percent,
            sigma,
            linked: true,
        },
        ..Default::default()
    }
}

/// Per-pixel application of the solution, mirroring the GLSL fragment
/// shader: `clamp((v - shadow) * scale)` → MTF → green removal → saturation.
fn shader_stretch(prepared: &Image, sol: &MtfSolution, cfg: &PipelineConfig) -> Image {
    let mut out = prepared.clone();
    for (c, ch) in out.channels_mut().iter_mut().enumerate() {
        let shadow = sol.shadows[c];
        for v in ch.iter_mut() {
            let x = ((*v - shadow) * sol.scale).clamp(0.0, 1.0);
            let denom = (2.0 * sol.midtone - 1.0) * x - sol.midtone;
            *v = if denom.abs() < 1e-10 {
                x
            } else {
                ((sol.midtone - 1.0) * x / denom).clamp(0.0, 1.0)
            };
        }
    }
    remove_green(&mut out, cfg.green_removal);
    saturate(&mut out, cfg.saturation);
    out
}

fn max_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.num_channels(), b.num_channels());
    a.channels()
        .iter()
        .zip(b.channels())
        .flat_map(|(ca, cb)| ca.iter().zip(cb.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn shader_path_matches_pipeline_rgb() {
    let field = common::rgb_field();
    let cfg = config(BG, SIGMA);
    let expected = process(field.data.clone(), &cfg);

    let prepared = prepare(field.data.clone(), &cfg);
    let sol = mtf_display_solution(&prepared, BG, SIGMA);
    let actual = shader_stretch(&prepared, &sol, &cfg);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-12, "shader path diverged from pipeline: {diff}");
}

#[test]
fn shader_path_matches_pipeline_mono() {
    let field = common::gradient_field();
    let cfg = config(BG, SIGMA);
    let expected = process(field.data.clone(), &cfg);

    let prepared = prepare(field.data.clone(), &cfg);
    let sol = mtf_display_solution(&prepared, BG, SIGMA);
    let actual = shader_stretch(&prepared, &sol, &cfg);

    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-12, "shader path diverged from pipeline: {diff}");
}

#[test]
fn prepare_then_stretch_equals_process() {
    // prepare() must stay the exact pre-stretch prefix of process()
    let field = common::rgb_field();
    let cfg = config(BG, SIGMA);
    let expected = process(field.data.clone(), &cfg);

    let prepared = prepare(field.data.clone(), &cfg);
    let stretched = processinator::stretch(
        prepared.clone(),
        &processinator::StretchOptions {
            algorithm: cfg.stretch.clone(),
            autocrop: false,
            pre_normalized: true,
        },
    );
    let mut finished = stretched;
    remove_green(&mut finished, cfg.green_removal);
    saturate(&mut finished, cfg.saturation);

    assert!(max_abs_diff(&finished, &expected) == 0.0);
}

// ---------------------------------------------------------------------------
// Frontend solution reconstruction (stats + histogram), transcribed from
// src/lib/stretch/mtf-solution.ts in astra. Keep the two in sync.
// ---------------------------------------------------------------------------

fn histogram(data: &[f64]) -> Vec<u32> {
    let mut hist = vec![0u32; HIST_BINS];
    for &v in data {
        if v > 0.0 {
            let bin = ((v * HIST_BINS as f64) as usize).min(HIST_BINS - 1);
            hist[bin] += 1;
        }
    }
    hist
}

/// Median of the histogrammed values strictly above `threshold`, assuming a
/// uniform distribution inside each bin. None when nothing lies above.
fn conditional_median(hist: &[u32], threshold: f64) -> Option<f64> {
    let bins = hist.len() as f64;
    let t = threshold.max(0.0);
    let tb = ((t * bins) as usize).min(hist.len() - 1);
    let bin_lo = tb as f64 / bins;
    let bin_hi = (tb as f64 + 1.0) / bins;
    let frac_above = if t <= bin_lo {
        1.0
    } else {
        ((bin_hi - t) * bins).max(0.0)
    };

    let first = hist[tb] as f64 * frac_above;
    let total: f64 = first + hist[tb + 1..].iter().map(|&c| c as f64).sum::<f64>();
    if total <= 0.0 {
        return None;
    }

    let target = total / 2.0;
    if first >= target {
        return Some(t + (target / first) * (bin_hi - t));
    }
    let mut acc = first;
    for (i, &c) in hist.iter().enumerate().skip(tb + 1) {
        let c = c as f64;
        if acc + c >= target {
            return Some((i as f64 + (target - acc) / c) / bins);
        }
        acc += c;
    }
    Some((bins - 0.5) / bins)
}

fn midtone_for_background(median: f64, bg_percent: f64) -> f64 {
    if median > 0.0 && median < 1.0 && bg_percent > 0.0 {
        let m = median * (bg_percent - 1.0) / (2.0 * bg_percent * median - bg_percent - median);
        m.clamp(1e-4, 0.9999)
    } else {
        0.5
    }
}

/// The frontend's linked-color solution from serialized stats + histogram.
fn frontend_linked_solution(
    stats: &[MtfStats],
    hist: &[u32],
    bg_percent: f64,
    sigma: f64,
) -> MtfSolution {
    let k = sigma * 1.4826;
    let residual = stats
        .iter()
        .map(|s| (s.median - s.p25) + k * s.mad)
        .fold(f64::INFINITY, f64::min);
    let shadows: Vec<f64> = stats
        .iter()
        .map(|s| (s.median - residual).max(0.0))
        .collect();
    let max_shadow = shadows.iter().copied().fold(0.0, f64::max);
    let scale = 1.0 / (1.0 - max_shadow).max(1e-6);

    let ref_idx = std::cmp::min(1, stats.len() - 1);
    let ref_median = match conditional_median(hist, shadows[ref_idx]) {
        Some(mc) => ((mc - shadows[ref_idx]) * scale).clamp(0.0, 1.0),
        None => 0.0,
    };
    MtfSolution {
        shadows,
        scale,
        midtone: midtone_for_background(ref_median, bg_percent),
    }
}

/// The frontend's mono solution — analytic, stats only.
fn frontend_mono_solution(stats: &MtfStats, bg_percent: f64, sigma: f64) -> MtfSolution {
    if stats.count == 0 {
        return MtfSolution {
            shadows: vec![0.0],
            scale: 1.0,
            midtone: 0.5,
        };
    }
    let shadow = (stats.p25 - sigma * stats.mad * 1.4826).max(0.0);
    let range = 1.0 - shadow;
    let median_norm = (stats.median - shadow) / range;
    MtfSolution {
        shadows: vec![shadow],
        scale: 1.0 / range,
        midtone: midtone_for_background(median_norm, bg_percent),
    }
}

#[test]
fn frontend_solution_matches_exact_rgb() {
    let field = common::rgb_field();
    let cfg = config(BG, SIGMA);
    let prepared = prepare(field.data.clone(), &cfg);

    let stats: Vec<MtfStats> = prepared
        .channels()
        .iter()
        .map(|ch| mtf_stats_linked(ch))
        .collect();
    let hist = histogram(prepared.channel(1));

    for (bg, sigma) in [
        (0.10, 3.0),
        (0.15, 3.0),
        (0.25, 2.0),
        (0.40, 1.0),
        (0.15, 0.0),
    ] {
        let exact = mtf_display_solution(&prepared, bg, sigma);
        let front = frontend_linked_solution(&stats, &hist, bg, sigma);

        // Shadows and scale are pure scalar math over the same stats — exact
        assert_eq!(front.shadows, exact.shadows, "bg={bg} sigma={sigma}");
        assert!((front.scale - exact.scale).abs() < 1e-15);

        // The midtone goes through the histogram approximation
        assert!(
            (front.midtone - exact.midtone).abs() < 1e-3,
            "midtone diverged at bg={bg} sigma={sigma}: {} vs {}",
            front.midtone,
            exact.midtone
        );

        // And the approximation must be invisible in the rendered image
        let img_exact = shader_stretch(&prepared, &exact, &cfg);
        let img_front = shader_stretch(&prepared, &front, &cfg);
        let diff = max_abs_diff(&img_exact, &img_front);
        assert!(
            diff < 2.0 / 255.0,
            "histogram midtone visibly changed the image at bg={bg} sigma={sigma}: {diff}"
        );
    }
}

#[test]
fn frontend_solution_matches_exact_mono() {
    let field = common::mono_field();
    let cfg = config(BG, SIGMA);
    let prepared = prepare(field.data.clone(), &cfg);

    let stats = mtf_stats_channel(prepared.channel(0));

    for (bg, sigma) in [(0.10, 3.0), (0.15, 3.0), (0.25, 2.0), (0.40, 1.0)] {
        let exact = mtf_display_solution(&prepared, bg, sigma);
        let front = frontend_mono_solution(&stats, bg, sigma);
        // Mono is fully analytic — identical
        assert_eq!(front, exact, "bg={bg} sigma={sigma}");
    }
}

#[test]
fn empty_channel_gets_identity_solution() {
    let flat = Image::new_mono(16, 16, vec![0.0; 256]);
    let sol = mtf_display_solution(&flat, BG, SIGMA);
    assert_eq!(sol.shadows, vec![0.0]);
    assert_eq!(sol.scale, 1.0);
    assert_eq!(sol.midtone, 0.5);

    // ...and the identity solution really is the identity map
    let data: Vec<f64> = (0..256).map(|i| i as f64 / 255.0).collect();
    let img = Image::new_mono(16, 16, data.clone());
    let out = shader_stretch(&img, &sol, &config(BG, SIGMA));
    assert_eq!(out.channel(0), &data[..]);
}
