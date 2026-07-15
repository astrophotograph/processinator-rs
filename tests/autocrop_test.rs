//! Tests for dark stacking-edge detection and cropping, ported from
//! test_autocrop.py.

mod common;

use common::*;
use processinator::synthetic::{make_test_image, SyntheticParams};
use processinator::{autocrop, AutocropParams};

#[test]
fn detects_configured_edges() {
    let field = edged_field();
    let (_, crop) = autocrop(&field.data, &AutocropParams::default());
    assert_eq!(crop, field.dark_edges);
}

#[test]
fn cropped_shape_matches() {
    let field = edged_field();
    let (cropped, (top, bottom, left, right)) = autocrop(&field.data, &AutocropParams::default());
    assert_eq!(cropped.height(), field.data.height() - top - bottom);
    assert_eq!(cropped.width(), field.data.width() - left - right);
}

#[test]
fn clean_image_not_cropped() {
    let field = mono_field();
    let (cropped, crop) = autocrop(&field.data, &AutocropParams::default());
    assert_eq!(crop, (0, 0, 0, 0));
    assert_eq!(cropped.width(), field.data.width());
    assert_eq!(cropped.height(), field.data.height());
}

#[test]
fn rgb_edges_detected() {
    let img = make_test_image(&SyntheticParams {
        rgb: true,
        dark_edges: (10, 0, 0, 16),
        seed: 9,
        ..Default::default()
    });
    let (cropped, crop) = autocrop(&img.data, &AutocropParams::default());
    assert_eq!(crop, (10, 0, 0, 16));
    assert!(cropped.is_color());
}

/// Edges thinner than min_crop_fraction (1%) should not trigger a crop.
#[test]
fn subthreshold_edge_ignored() {
    let img = make_test_image(&SyntheticParams {
        dark_edges: (1, 0, 0, 0),
        seed: 4,
        ..Default::default()
    });
    let (_, crop) = autocrop(&img.data, &AutocropParams::default());
    assert_eq!(crop, (0, 0, 0, 0));
}

#[test]
fn interior_content_unchanged() {
    let field = edged_field();
    let (cropped, (top, _bottom, left, _right)) = autocrop(&field.data, &AutocropParams::default());
    for y in 0..cropped.height() {
        for x in 0..cropped.width() {
            assert_eq!(
                cropped.get(x, y, 0),
                field.data.get(x + left, y + top, 0),
                "mismatch at ({x}, {y})"
            );
        }
    }
}
