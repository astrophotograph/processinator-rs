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
    let mut result = data.clone();
    remove_gradient(&mut result, &GradientParams::default());
    assert!(background_tilt(&result) < 0.35 * background_tilt(&data));
}

/// On an already-flat image the fitted surface should be near-constant.
#[test]
fn no_gradient_is_gentle() {
    let field = mono_field();
    let data = scaled(&field.data, 1.0 / image_max(&field.data));
    let mut result = data.clone();
    remove_gradient(&mut result, &GradientParams::default());
    // The output is background-shifted toward 0 but should stay flat
    assert!(background_tilt(&result) < background_tilt(&data) + 0.01);
}

/// Star contrast above local background must be preserved.
#[test]
fn stars_survive() {
    let field = gradient_field();
    let scale = image_max(&field.data);
    let data = scaled(&field.data, 1.0 / scale);
    let mut result = data.clone();
    remove_gradient(&mut result, &GradientParams::default());

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
        let mut result = data.clone();
        remove_gradient(&mut result, &GradientParams::default());
        assert_eq!(result.width(), data.width());
        assert_eq!(result.height(), data.height());
        assert_eq!(result.num_channels(), data.num_channels());
    }
}

#[test]
fn output_clipped_to_unit_range() {
    let field = gradient_field();
    let data = scaled(&field.data, 1.0 / image_max(&field.data));
    let mut result = data.clone();
    remove_gradient(&mut result, &GradientParams::default());
    assert_unit_range(&result);
}

/// A dark lane (dust) below the sky must keep its depth and must not be
/// clamped to zero: the surface is subtracted by its shape only.
#[test]
fn dark_lane_keeps_its_depth_and_is_not_crushed() {
    let (w, h) = (128usize, 128usize);
    let mut data = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let sky = 0.30 + 0.12 * x as f64 / (w - 1) as f64 + 0.05 * y as f64 / (h - 1) as f64;
            let lane = if (50..70).contains(&y) { 0.85 } else { 1.0 };
            data[y * w + x] = sky * lane;
        }
    }
    let mut img = Image::new_mono(w, h, data);
    remove_gradient(&mut img, &GradientParams::default());
    let ch = img.channel(0);
    let lane: Vec<f64> = (55..65).flat_map(|y| ch[y * w..(y + 1) * w].to_vec()).collect();
    let sky: Vec<f64> = (90..100).flat_map(|y| ch[y * w..(y + 1) * w].to_vec()).collect();
    let (lane_med, sky_med) = (median(&lane), median(&sky));
    assert!(sky_med > 0.2, "background level kept, not re-zeroed: {sky_med}");
    assert!(lane_med > 0.05, "lane must not be clamped to black: {lane_med}");
    let depth = (sky_med - lane_med) / sky_med;
    assert!((depth - 0.15).abs() < 0.05, "lane depth {depth} drifted from 0.15");
}
