//! Sanity tests for the synthetic image generator itself, ported from
//! test_synthetic.py.

mod common;

use common::*;
use processinator::synthetic::{make_test_image, SyntheticParams};
use processinator::{read_fits, write_fits};

#[test]
fn mono_shape() {
    let img = make_test_image(&SyntheticParams {
        height: 128,
        width: 200,
        seed: 1,
        ..Default::default()
    });
    assert_eq!((img.data.height(), img.data.width()), (128, 200));
    assert_eq!((img.clean.height(), img.clean.width()), (128, 200));
}

#[test]
fn rgb_shape() {
    let img = make_test_image(&SyntheticParams {
        height: 128,
        width: 200,
        rgb: true,
        seed: 1,
        ..Default::default()
    });
    assert_eq!((img.data.height(), img.data.width()), (128, 200));
    assert_eq!(img.data.num_channels(), 3);
}

#[test]
fn deterministic() {
    let a = make_test_image(&SyntheticParams {
        seed: 99,
        ..Default::default()
    });
    let b = make_test_image(&SyntheticParams {
        seed: 99,
        ..Default::default()
    });
    assert_eq!(a.data, b.data);
}

#[test]
fn stars_present_and_recorded() {
    let img = make_test_image(&SyntheticParams {
        n_stars: 30,
        seed: 1,
        ..Default::default()
    });
    assert_eq!(img.stars.len(), 30);
    // Brightest star should tower over the background
    let brightest = img
        .stars
        .iter()
        .max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap())
        .unwrap();
    let xi = brightest.x.round() as usize;
    let yi = brightest.y.round() as usize;
    assert!(img.clean.get(xi, yi, 0) > img.background + 0.8 * brightest.amplitude);
}

#[test]
fn noise_matches_config() {
    let img = make_test_image(&SyntheticParams {
        noise_sigma: 25.0,
        n_stars: 0,
        seed: 1,
        ..Default::default()
    });
    let residual_std = diff_std(&img.data, &img.clean);
    assert!(
        residual_std > 22.0 && residual_std < 28.0,
        "residual std = {residual_std}"
    );
}

#[test]
fn dark_edges_are_dark() {
    let img = make_test_image(&SyntheticParams {
        dark_edges: (15, 0, 0, 0),
        seed: 1,
        ..Default::default()
    });
    let w = img.data.width();
    let ch = img.data.channel(0);
    let edge_median = median(&ch[..15 * w]);
    let interior_median = median(&ch[30 * w..(img.data.height() - 30) * w]);
    assert!(edge_median < 0.1 * interior_median);
}

#[test]
fn data_non_negative() {
    let img = make_test_image(&SyntheticParams {
        noise_sigma: 100.0,
        seed: 1,
        ..Default::default()
    });
    assert!(image_min(&img.data) >= 0.0);
}

// --- FITS round trip --------------------------------------------------------

fn assert_close(a: &processinator::Image, b: &processinator::Image, rtol: f64) {
    assert_eq!(a.num_channels(), b.num_channels());
    for c in 0..a.num_channels() {
        for (x, y) in a.channel(c).iter().zip(b.channel(c)) {
            assert!(
                (x - y).abs() <= rtol * y.abs(),
                "value mismatch: {x} vs {y}"
            );
        }
    }
}

#[test]
fn mono_roundtrip() {
    let dir = std::env::temp_dir().join("processinator-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mono.fits");

    let img = make_test_image(&SyntheticParams {
        height: 96,
        width: 128,
        seed: 8,
        ..Default::default()
    });
    write_fits(&img.data, &path).unwrap();
    let loaded = read_fits(&path).unwrap();
    assert_eq!((loaded.height(), loaded.width()), (96, 128));
    assert_close(&loaded, &img.data, 1e-6);
}

#[test]
fn rgb_roundtrip() {
    let dir = std::env::temp_dir().join("processinator-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rgb.fits");

    let img = make_test_image(&SyntheticParams {
        height: 96,
        width: 128,
        rgb: true,
        seed: 8,
        ..Default::default()
    });
    write_fits(&img.data, &path).unwrap();
    let loaded = read_fits(&path).unwrap();
    // Stored channels-first, read back as three planes
    assert_eq!((loaded.height(), loaded.width()), (96, 128));
    assert_eq!(loaded.num_channels(), 3);
    assert_close(&loaded, &img.data, 1e-6);
}
