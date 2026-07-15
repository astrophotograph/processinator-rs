//! Tests for stretching algorithms, ported from test_stretch.py.

mod common;

use common::*;
use processinator::{stretch, Image, StretchAlgorithm, StretchOptions};

/// Simulate a typical FITS image: mostly dark with a few bright pixels.
fn grayscale_image() -> Image {
    let (w, h) = (256, 256);
    let mut rng = TestRng::new(42);
    let mut data: Vec<f64> = (0..w * h)
        .map(|_| rng.normal(100.0, 10.0).clamp(0.0, 65535.0))
        .collect();
    data[50 * w + 50] = 50000.0; // bright star
    data[100 * w + 150] = 30000.0; // medium star
    data[200 * w + 200] = 65000.0; // very bright star
    Image::new_mono(w, h, data)
}

/// RGB version of the test image.
fn rgb_image() -> Image {
    let gray = grayscale_image();
    let base = gray.channel(0);
    Image::new_rgb(
        gray.width(),
        gray.height(),
        [
            base.to_vec(),
            base.iter().map(|&v| v * 0.8).collect(),
            base.iter().map(|&v| v * 0.6).collect(),
        ],
    )
}

fn all_algorithms() -> Vec<StretchAlgorithm> {
    vec![
        StretchAlgorithm::mtf(),
        StretchAlgorithm::arcsinh(),
        StretchAlgorithm::log(),
        StretchAlgorithm::linear(),
        StretchAlgorithm::statistical(),
    ]
}

// --- Output range: all algorithms should produce output in [0, 1] ---------

#[test]
fn output_range_grayscale() {
    let img = grayscale_image();
    for algo in all_algorithms() {
        let result = stretch(&img, &StretchOptions::new(algo.clone()));
        assert_unit_range(&result);
    }
}

#[test]
fn output_range_rgb() {
    let img = rgb_image();
    for algo in all_algorithms() {
        let result = stretch(&img, &StretchOptions::new(algo.clone()));
        assert_unit_range(&result);
    }
}

// --- Shape: output shape should match input shape --------------------------

#[test]
fn shape_preserved_grayscale() {
    let img = grayscale_image();
    for algo in all_algorithms() {
        let result = stretch(&img, &StretchOptions::new(algo));
        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
        assert_eq!(result.num_channels(), img.num_channels());
    }
}

#[test]
fn shape_preserved_rgb() {
    let img = rgb_image();
    for algo in all_algorithms() {
        let result = stretch(&img, &StretchOptions::new(algo));
        assert_eq!(result.width(), img.width());
        assert_eq!(result.height(), img.height());
        assert_eq!(result.num_channels(), 3);
    }
}

// --- Behavior: stretching actually changes the data distribution -----------

#[test]
fn mtf_reveals_background() {
    let result = stretch(&grayscale_image(), &StretchOptions::default());
    // After MTF stretch, the median should be closer to bg_percent (0.15)
    let median = image_median(&result);
    assert!(median > 0.01 && median < 0.5, "median = {median}");
}

#[test]
fn linear_uses_percentiles() {
    let result = stretch(
        &grayscale_image(),
        &StretchOptions::new(StretchAlgorithm::linear()),
    );
    // Most values should be in the middle range after linear stretch
    assert!(image_median(&result) > 0.1);
}

#[test]
fn constant_image_produces_zeros() {
    let img = Image::new_mono(64, 64, vec![1000.0; 64 * 64]);
    let result = stretch(&img, &StretchOptions::default());
    assert!(result.channel(0).iter().all(|&v| v == 0.0));
}

#[test]
fn default_algorithm_is_mtf() {
    let img = grayscale_image();
    let default_result = stretch(&img, &StretchOptions::default());
    let mtf_result = stretch(&img, &StretchOptions::new(StretchAlgorithm::mtf()));
    assert_eq!(default_result, mtf_result);
}

// --- Color behavior ---------------------------------------------------------

/// The linked MTF must not rebalance channels: a nebula that is brightest
/// in red keeps R > G > B through the stretch (regression test for the old
/// multiplicative background equalization, which crushed red and shifted
/// real captures green). Background and stars are identical across
/// channels so the per-channel normalization stays comparable; only the
/// nebula amplitude differs.
#[test]
fn linked_mtf_preserves_nebula_color() {
    let (w, h) = (256, 256);
    let mut rng = TestRng::new(42);
    let base: Vec<f64> = (0..w * h).map(|_| rng.normal(100.0, 10.0)).collect();

    let mut channels = [base.clone(), base.clone(), base];
    for (ch, nebula_amp) in channels.iter_mut().zip([300.0, 150.0, 75.0]) {
        // Shared bright stars anchor each channel's normalization ceiling
        ch[50 * w + 50] = 50000.0;
        ch[200 * w + 200] = 65000.0;
        // H-alpha-ish nebula: strongest in red
        for y in 100..140 {
            for x in 100..140 {
                ch[y * w + x] += nebula_amp;
            }
        }
    }
    let [r, g, b] = channels;
    let img = Image::new_rgb(w, h, [r, g, b]);

    let result = stretch(&img, &StretchOptions::default());
    let nebula_mean = |c: usize| {
        let ch = result.channel(c);
        let mut sum = 0.0;
        for y in 110..130 {
            for x in 110..130 {
                sum += ch[y * w + x];
            }
        }
        sum / 400.0
    };
    let (r, g, b) = (nebula_mean(0), nebula_mean(1), nebula_mean(2));
    assert!(
        r > g + 0.02 && g > b + 0.02,
        "nebula color lost: r={r} g={g} b={b}"
    );
}

#[test]
fn saturate_scales_chroma_around_luminance() {
    let mut img = Image::new_rgb(1, 1, [vec![0.5], vec![0.3], vec![0.2]]);
    processinator::saturate(&mut img, 1.5);
    let lum = (0.5 + 0.3 + 0.2) / 3.0;
    for (c, orig) in [0.5, 0.3, 0.2].iter().enumerate() {
        let expected = lum + 1.5 * (orig - lum);
        assert!((img.channel(c)[0] - expected).abs() < 1e-12);
    }
}

#[test]
fn saturate_noop_for_factor_one_and_mono() {
    let mut rgb = Image::new_rgb(1, 1, [vec![0.5], vec![0.3], vec![0.2]]);
    processinator::saturate(&mut rgb, 1.0);
    assert_eq!(rgb.channel(0)[0], 0.5);

    let mut mono = Image::new_mono(1, 1, vec![0.5]);
    processinator::saturate(&mut mono, 2.0);
    assert_eq!(mono.channel(0)[0], 0.5);
}
