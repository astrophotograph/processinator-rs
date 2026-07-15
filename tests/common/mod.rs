//! Shared fixtures and helpers for the integration tests, ported from the
//! Python suite's conftest.py.

#![allow(dead_code)]

use processinator::synthetic::{make_test_image, SyntheticImage, SyntheticParams};
use processinator::Image;

/// Plain mono star field with noise.
pub fn mono_field() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        seed: 42,
        ..Default::default()
    })
}

/// RGB star field with a color cast.
pub fn rgb_field() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        rgb: true,
        seed: 42,
        ..Default::default()
    })
}

/// Mono field with a strong sky gradient and some nebulosity.
pub fn gradient_field() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        gradient_amplitude: 400.0,
        nebula_amplitude: 200.0,
        seed: 7,
        ..Default::default()
    })
}

/// Mono field with dark stacking edges on three sides.
pub fn edged_field() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        dark_edges: (12, 8, 20, 0),
        seed: 3,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// numpy-style helpers
// ---------------------------------------------------------------------------

/// Median (numpy convention: even-length inputs average the middle pair).
pub fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut buf = values.to_vec();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = buf.len() / 2;
    if buf.len() % 2 == 1 {
        buf[mid]
    } else {
        0.5 * (buf[mid - 1] + buf[mid])
    }
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation (numpy default, ddof = 0).
pub fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
}

/// All samples of all channels, concatenated.
pub fn all_values(img: &Image) -> Vec<f64> {
    img.channels().iter().flatten().copied().collect()
}

pub fn image_min(img: &Image) -> f64 {
    all_values(img).into_iter().fold(f64::INFINITY, f64::min)
}

pub fn image_max(img: &Image) -> f64 {
    all_values(img)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max)
}

pub fn image_median(img: &Image) -> f64 {
    median(&all_values(img))
}

/// Mean squared difference over all channels.
pub fn mse(a: &Image, b: &Image) -> f64 {
    let av = all_values(a);
    let bv = all_values(b);
    assert_eq!(av.len(), bv.len());
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / av.len() as f64
}

/// Std of the elementwise difference over all channels.
pub fn diff_std(a: &Image, b: &Image) -> f64 {
    let av = all_values(a);
    let bv = all_values(b);
    assert_eq!(av.len(), bv.len());
    let diffs: Vec<f64> = av.iter().zip(&bv).map(|(x, y)| x - y).collect();
    std_dev(&diffs)
}

/// New image with every sample multiplied by `scale`.
pub fn scaled(img: &Image, scale: f64) -> Image {
    let channels = img
        .channels()
        .iter()
        .map(|ch| ch.iter().map(|&v| v * scale).collect())
        .collect();
    Image::from_channels(img.width(), img.height(), channels)
}

/// Assert every sample lies in [0, 1].
pub fn assert_unit_range(img: &Image) {
    assert!(image_min(img) >= 0.0, "min {} < 0", image_min(img));
    assert!(image_max(img) <= 1.0, "max {} > 1", image_max(img));
}

/// Deterministic RNG mirroring the library's internal one, for building
/// ad-hoc test data (SplitMix64 + Box-Muller).
pub struct TestRng {
    state: u64,
}

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self, mean: f64, sigma: f64) -> f64 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        mean + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
