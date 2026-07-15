//! Tests for starlet wavelet denoising, ported from test_denoise.py.
//!
//! The key "astronomically safe" properties: noise goes down, star
//! photometry and large-scale structure survive.
//!
//! Note on modes: hard thresholding (the default) preserves point-source
//! photometry exactly, which is why it is the astronomy standard. Soft
//! thresholding gives a smoother background but systematically dims stars,
//! so its quality tests use a starless field.

mod common;

use common::*;
use processinator::synthetic::{make_test_image, SyntheticImage, SyntheticParams};
use processinator::{denoise, DenoiseParams, ThresholdMode};

/// Star field with nebulosity and noise.
fn noisy() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        nebula_amplitude: 300.0,
        seed: 11,
        ..Default::default()
    })
}

/// Pure background + nebulosity — no point sources.
fn starless() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        n_stars: 0,
        nebula_amplitude: 300.0,
        seed: 11,
        ..Default::default()
    })
}

// --- Noise reduction --------------------------------------------------------

/// Denoised image must be much closer to the noise-free truth.
#[test]
fn mse_against_clean_improves() {
    let field = noisy();
    let result = denoise(&field.data, &DenoiseParams::default());
    let mse_before = mse(&field.data, &field.clean);
    let mse_after = mse(&result, &field.clean);
    assert!(
        mse_after < 0.5 * mse_before,
        "mse {mse_before} -> {mse_after}"
    );
}

/// In a starless field the residual noise should shrink a lot.
#[test]
fn background_noise_suppressed() {
    let field = starless();
    let result = denoise(&field.data, &DenoiseParams::default());
    let std_before = diff_std(&field.data, &field.clean);
    let std_after = diff_std(&result, &field.clean);
    assert!(std_after < 0.5 * std_before);
}

/// Soft thresholding shines when there are no point sources.
#[test]
fn soft_mode_on_starless_field() {
    let field = starless();
    let result = denoise(
        &field.data,
        &DenoiseParams {
            mode: ThresholdMode::Soft,
            ..Default::default()
        },
    );
    let mse_before = mse(&field.data, &field.clean);
    let mse_after = mse(&result, &field.clean);
    assert!(mse_after < 0.2 * mse_before);
}

#[test]
fn higher_threshold_smooths_more() {
    let field = starless();
    let gentle = denoise(
        &field.data,
        &DenoiseParams {
            threshold: 2.0,
            ..Default::default()
        },
    );
    let strong = denoise(
        &field.data,
        &DenoiseParams {
            threshold: 5.0,
            ..Default::default()
        },
    );
    let resid_gentle = diff_std(&gentle, &field.clean);
    let resid_strong = diff_std(&strong, &field.clean);
    // Stronger threshold removes more noise in a starless field
    assert!(resid_strong <= resid_gentle);
}

// --- Signal preservation ----------------------------------------------------

/// Hard thresholding (default) keeps large coefficients bit-for-bit.
///
/// Exact preservation only holds where the star is the locally dominant
/// source: on the shoulder of a brighter neighbor, the finest-scale
/// coefficient at the catalog position is legitimately near zero and gets
/// thresholded. The Python suite satisfies this implicitly through its
/// RNG; with this generator's star field the isolation filter is explicit.
#[test]
fn star_peaks_preserved_exactly_by_default() {
    let field = noisy();
    let result = denoise(&field.data, &DenoiseParams::default());
    let bright: Vec<_> = field
        .stars
        .iter()
        .filter(|s| s.amplitude > 100.0 * field.noise_sigma)
        .filter(|s| {
            !field.stars.iter().any(|o| {
                o.amplitude > s.amplitude
                    && ((o.x - s.x).powi(2) + (o.y - s.y).powi(2)).sqrt() < 10.0
            })
        })
        .collect();
    assert!(!bright.is_empty());
    for star in bright {
        let xi = star.x.round() as usize;
        let yi = star.y.round() as usize;
        let observed = field.data.get(xi, yi, 0);
        let denoised = result.get(xi, yi, 0);
        assert!(
            (denoised - observed).abs() <= 1e-5 * observed.abs(),
            "star at ({xi}, {yi}): {observed} -> {denoised}"
        );
    }
}

/// Soft mode dims stars by ~the threshold sum — small for bright stars.
#[test]
fn soft_mode_star_loss_is_bounded() {
    let field = noisy();
    let result = denoise(
        &field.data,
        &DenoiseParams {
            mode: ThresholdMode::Soft,
            ..Default::default()
        },
    );
    let bright: Vec<_> = field
        .stars
        .iter()
        .filter(|s| s.amplitude > 200.0 * field.noise_sigma)
        .collect();
    assert!(!bright.is_empty());
    for star in bright {
        let xi = star.x.round() as usize;
        let yi = star.y.round() as usize;
        assert!(result.get(xi, yi, 0) > 0.9 * field.clean.get(xi, yi, 0));
    }
}

/// Large-scale structure lives in the untouched residual.
#[test]
fn nebulosity_preserved() {
    let field = noisy();
    let result = denoise(&field.data, &DenoiseParams::default());
    const BLOCK: usize = 32;
    let w = field.clean.width();
    let h = field.clean.height();
    let block_mean = |img: &processinator::Image, by: usize, bx: usize| {
        let ch = img.channel(0);
        let mut sum = 0.0;
        for y in by * BLOCK..(by + 1) * BLOCK {
            for x in bx * BLOCK..(bx + 1) * BLOCK {
                sum += ch[y * w + x];
            }
        }
        sum / (BLOCK * BLOCK) as f64
    };
    for by in 0..h / BLOCK {
        for bx in 0..w / BLOCK {
            let clean = block_mean(&field.clean, by, bx);
            let denoised = block_mean(&result, by, bx);
            assert!(
                (denoised - clean).abs() <= 0.05 * clean.abs(),
                "block ({by}, {bx}): {clean} vs {denoised}"
            );
        }
    }
}

#[test]
fn noise_free_image_nearly_unchanged() {
    let img = make_test_image(&SyntheticParams {
        noise_sigma: 0.0,
        seed: 5,
        ..Default::default()
    });
    let result = denoise(&img.data, &DenoiseParams::default());
    // Nothing to remove — output should track the input closely
    let abs_diffs: Vec<f64> = img
        .data
        .channel(0)
        .iter()
        .zip(result.channel(0))
        .map(|(a, b)| (a - b).abs())
        .collect();
    assert!(median(&abs_diffs) < 1e-4 * image_max(&img.data));
}

// --- Interface --------------------------------------------------------------

#[test]
fn shape_mono() {
    let field = noisy();
    let result = denoise(&field.data, &DenoiseParams::default());
    assert_eq!(result.width(), field.data.width());
    assert_eq!(result.height(), field.data.height());
    assert_eq!(result.num_channels(), 1);
}

#[test]
fn rgb() {
    let img = make_test_image(&SyntheticParams {
        rgb: true,
        seed: 11,
        ..Default::default()
    });
    let result = denoise(&img.data, &DenoiseParams::default());
    assert_eq!(result.num_channels(), 3);
    assert_eq!(result.width(), img.data.width());
    assert_eq!(result.height(), img.data.height());
    assert!(mse(&result, &img.clean) < mse(&img.data, &img.clean));
}

#[test]
fn small_image_clamps_scales() {
    let img = make_test_image(&SyntheticParams {
        height: 48,
        width: 48,
        n_stars: 5,
        seed: 2,
        ..Default::default()
    });
    let result = denoise(
        &img.data,
        &DenoiseParams {
            n_scales: 8,
            ..Default::default()
        },
    );
    assert_eq!(result.width(), 48);
    assert_eq!(result.height(), 48);
}

#[test]
fn deterministic() {
    let field = noisy();
    let a = denoise(&field.data, &DenoiseParams::default());
    let b = denoise(&field.data, &DenoiseParams::default());
    assert_eq!(a, b);
}
