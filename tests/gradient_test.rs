//! Tests for background gradient removal, ported from test_gradient.py.

mod common;

use common::*;
use processinator::synthetic::{make_test_image, SyntheticParams};
use processinator::{remove_gradient, GradientParams, Image};

/// Spread of row/column medians — a star-robust flatness metric.
fn background_tilt(img: &Image) -> f64 {
    let w = img.width();
    let h = img.height();
    let ch = img.channel(0);
    let row_medians: Vec<f64> = (0..h).map(|y| median(&ch[y * w..(y + 1) * w])).collect();
    let col_medians: Vec<f64> = (0..w)
        .map(|x| {
            let col: Vec<f64> = (0..h).map(|y| ch[y * w + x]).collect();
            median(&col)
        })
        .collect();
    let ptp = |v: &[f64]| {
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - v.iter().copied().fold(f64::INFINITY, f64::min)
    };
    ptp(&row_medians) + ptp(&col_medians)
}

#[test]
fn flattens_background() {
    let field = gradient_field();
    let data = scaled(&field.data, 1.0 / image_max(&field.data));
    let result = remove_gradient(&data, &GradientParams::default());
    assert!(background_tilt(&result) < 0.35 * background_tilt(&data));
}

/// On an already-flat image the fitted surface should be near-constant.
#[test]
fn no_gradient_is_gentle() {
    let field = mono_field();
    let data = scaled(&field.data, 1.0 / image_max(&field.data));
    let result = remove_gradient(&data, &GradientParams::default());
    // The output is background-shifted toward 0 but should stay flat
    assert!(background_tilt(&result) < background_tilt(&data) + 0.01);
}

/// Star contrast above local background must be preserved.
#[test]
fn stars_survive() {
    let field = gradient_field();
    let scale = image_max(&field.data);
    let data = scaled(&field.data, 1.0 / scale);
    let result = remove_gradient(&data, &GradientParams::default());

    let bright: Vec<_> = field
        .stars
        .iter()
        .filter(|s| s.amplitude > 5000.0)
        .collect();
    assert!(!bright.is_empty());
    for star in bright {
        let xi = star.x.round() as usize;
        let yi = star.y.round() as usize;
        let ch = result.channel(0);
        let w = result.width();
        let h = result.height();
        let mut window = Vec::new();
        for y in yi.saturating_sub(12)..(yi + 12).min(h) {
            for x in xi.saturating_sub(12)..(xi + 12).min(w) {
                window.push(ch[y * w + x]);
            }
        }
        let local_bg = median(&window);
        let contrast = ch[yi * w + xi] - local_bg;
        // Expected contrast in normalized units (clipping may trim the top)
        let expected = star.amplitude / scale;
        assert!(
            contrast > 0.5 * expected.min(1.0 - local_bg),
            "star at ({xi}, {yi}): contrast {contrast}"
        );
    }
}

#[test]
fn shape_preserved_mono_and_rgb() {
    for rgb in [false, true] {
        let img = make_test_image(&SyntheticParams {
            height: 128,
            width: 128,
            rgb,
            gradient_amplitude: 300.0,
            seed: 6,
            ..Default::default()
        });
        let data = scaled(&img.data, 1.0 / image_max(&img.data));
        let result = remove_gradient(&data, &GradientParams::default());
        assert_eq!(result.width(), data.width());
        assert_eq!(result.height(), data.height());
        assert_eq!(result.num_channels(), data.num_channels());
    }
}

#[test]
fn output_clipped_to_unit_range() {
    let field = gradient_field();
    let data = scaled(&field.data, 1.0 / image_max(&field.data));
    let result = remove_gradient(&data, &GradientParams::default());
    assert_unit_range(&result);
}
