//! Smoke tests over real captures dropped in `showcase/input/` (e.g. Seestar
//! FITS stacks). These validate the pipeline against actual telescope data
//! rather than synthetic frames. When no real images are present the tests
//! pass trivially, so CI stays green without committing binary fixtures.

use std::path::PathBuf;

use processinator::{process, read_fits, to_dynamic_image, PipelineConfig};

fn real_fits_files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("showcase/input");
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| {
                            let e = e.to_ascii_lowercase();
                            e == "fit" || e == "fits" || e == "fts"
                        })
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// Every real capture survives the full pipeline (gradient removal +
/// denoise + stretch) and produces plausible display data.
#[test]
fn real_images_survive_full_pipeline() {
    let files = real_fits_files();
    if files.is_empty() {
        eprintln!("no real images in showcase/input — skipping");
        return;
    }

    let config = PipelineConfig {
        denoise: true,
        ..Default::default()
    };

    for path in files {
        let data = read_fits(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let out = process(data.clone(), &config);

        assert_eq!(
            out.width(),
            data.width(),
            "{}: width changed",
            path.display()
        );
        assert_eq!(
            out.height(),
            data.height(),
            "{}: height changed",
            path.display()
        );
        assert_eq!(
            out.num_channels(),
            data.num_channels(),
            "{}: channel count changed",
            path.display()
        );

        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for ch in out.channels() {
            for &v in ch {
                assert!(v.is_finite(), "{}: non-finite output pixel", path.display());
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        assert!(
            (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi),
            "{}: output outside [0, 1] (range {lo}..{hi})",
            path.display()
        );
        assert!(
            hi - lo > 0.1,
            "{}: output nearly constant (range {lo}..{hi})",
            path.display()
        );

        // The 8-bit conversion used by the showcase must also hold up.
        let img = to_dynamic_image(&out);
        assert_eq!(img.width() as usize, data.width());
        assert_eq!(img.height() as usize, data.height());
    }
}
