//! Generate example FITS test images (and stretched PNG previews) into a
//! directory for manual inspection and experimentation. The test suite
//! generates its own images in memory and does not depend on these files.
//!
//! Usage:
//!     cargo run --release --example make_test_images [output_dir]

use std::path::PathBuf;

use processinator::synthetic::{make_test_image, SyntheticParams};
use processinator::{fits_to_image, write_fits, PipelineConfig};

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/images"));
    std::fs::create_dir_all(&out_dir).expect("create output directory");

    let base = SyntheticParams {
        height: 512,
        width: 512,
        ..Default::default()
    };
    let variants: Vec<(&str, SyntheticParams)> = vec![
        (
            "starfield_mono",
            SyntheticParams {
                seed: 42,
                ..base.clone()
            },
        ),
        (
            "starfield_rgb",
            SyntheticParams {
                rgb: true,
                seed: 42,
                ..base.clone()
            },
        ),
        (
            "gradient_nebula",
            SyntheticParams {
                gradient_amplitude: 400.0,
                nebula_amplitude: 250.0,
                seed: 7,
                ..base.clone()
            },
        ),
        (
            "stacking_edges",
            SyntheticParams {
                dark_edges: (12, 8, 20, 0),
                seed: 3,
                ..base.clone()
            },
        ),
        (
            "kitchen_sink_rgb",
            SyntheticParams {
                rgb: true,
                gradient_amplitude: 300.0,
                nebula_amplitude: 150.0,
                dark_edges: (10, 0, 14, 0),
                hot_pixels: 20,
                seed: 13,
                ..base.clone()
            },
        ),
        (
            "noisy",
            SyntheticParams {
                noise_sigma: 60.0,
                nebula_amplitude: 300.0,
                seed: 11,
                ..base.clone()
            },
        ),
    ];

    let n_variants = variants.len();
    for (name, params) in variants {
        let img = make_test_image(&params);
        let fits_path = out_dir.join(format!("{name}.fits"));
        write_fits(&img.data, &fits_path).expect("write FITS");

        // Also emit a stretched preview so the FITS can be eyeballed
        let png_path = out_dir.join(format!("{name}.png"));
        fits_to_image(&fits_path, Some(&png_path), &PipelineConfig::default())
            .expect("stretch preview");

        println!(
            "  wrote {}  ({}x{}, {} channel{})",
            fits_path.display(),
            img.data.height(),
            img.data.width(),
            img.data.num_channels(),
            if img.data.is_color() { "s" } else { "" },
        );
    }

    println!("\n{} example images in {}/", n_variants, out_dir.display());
}
