//! End-to-end tests for the processing pipeline, ported from
//! test_pipeline.py.

mod common;

use common::*;
use processinator::synthetic::{make_test_image, SyntheticImage, SyntheticParams};
use processinator::{process, PipelineConfig, StretchAlgorithm};

/// The works: gradient, nebula, dark edges, hot pixels, RGB color cast.
fn full_frame() -> SyntheticImage {
    make_test_image(&SyntheticParams {
        rgb: true,
        gradient_amplitude: 300.0,
        nebula_amplitude: 150.0,
        dark_edges: (10, 0, 14, 0),
        hot_pixels: 20,
        seed: 13,
        ..Default::default()
    })
}

#[test]
fn default_config() {
    let frame = full_frame();
    let result = process(frame.data.clone(), &PipelineConfig::default());
    assert_eq!(result.width(), frame.data.width());
    assert_eq!(result.height(), frame.data.height());
    assert_eq!(result.num_channels(), frame.data.num_channels());
    assert_unit_range(&result);
}

#[test]
fn all_steps_enabled() {
    let frame = full_frame();
    let config = PipelineConfig {
        denoise: true,
        ..Default::default()
    };
    let result = process(frame.data.clone(), &config);
    assert_eq!(result.width(), frame.data.width());
    assert_eq!(result.height(), frame.data.height());
    assert_unit_range(&result);
    // A stretched image should have a visible (non-black) background
    assert!(image_median(&result) > 0.01);
}

#[test]
fn all_steps_disabled() {
    let field = mono_field();
    let config = PipelineConfig {
        autocrop: false,
        gradient_removal: false,
        denoise: false,
        ..Default::default()
    };
    let result = process(field.data.clone(), &config);
    assert_eq!(result.width(), field.data.width());
    assert_eq!(result.height(), field.data.height());
    assert_unit_range(&result);
}

/// The synthetic RGB image has a strong color cast; linked MTF should
/// equalize the per-channel background medians.
#[test]
fn linked_mtf_neutralizes_color_cast() {
    let frame = full_frame();
    let result = process(frame.data.clone(), &PipelineConfig::default());
    let w = result.width();
    let h = result.height();
    let medians: Vec<f64> = (0..3)
        .map(|c| {
            let ch = result.channel(c);
            let mut interior = Vec::new();
            for y in 24..h - 24 {
                for x in 24..w - 24 {
                    interior.push(ch[y * w + x]);
                }
            }
            median(&interior)
        })
        .collect();
    let spread = medians.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - medians.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(spread < 0.05, "channel medians {medians:?}");
}

#[test]
fn denoise_smooths_output() {
    let frame = full_frame();
    let plain = process(
        frame.data.clone(),
        &PipelineConfig {
            denoise: false,
            ..Default::default()
        },
    );
    let smoothed = process(
        frame.data.clone(),
        &PipelineConfig {
            denoise: true,
            ..Default::default()
        },
    );
    // Compare local roughness (std of vertical first differences) in a
    // star-free region of the green channel; denoised must be smoother
    let roughness = |img: &processinator::Image| {
        let w = img.width();
        let ch = img.channel(1);
        let mut diffs = Vec::new();
        for y in 20..79 {
            for x in 20..80 {
                diffs.push(ch[(y + 1) * w + x] - ch[y * w + x]);
            }
        }
        std_dev(&diffs)
    };
    assert!(roughness(&smoothed) < roughness(&plain));
}

#[test]
fn alternate_algorithm() {
    let field = mono_field();
    let config = PipelineConfig {
        stretch: StretchAlgorithm::arcsinh(),
        ..Default::default()
    };
    let result = process(field.data.clone(), &config);
    assert_unit_range(&result);
}

#[test]
fn stretch_params_passthrough() {
    let field = mono_field();
    let with_bg = |bg_percent: f64| PipelineConfig {
        stretch: StretchAlgorithm::Mtf {
            bg_percent,
            sigma: 3.0,
            linked: true,
        },
        ..Default::default()
    };
    let bright = process(field.data.clone(), &with_bg(0.4));
    let dim = process(field.data.clone(), &with_bg(0.05));
    assert!(image_median(&bright) > image_median(&dim));
}
