# processinator

Astronomy image processing library. Converts linear FITS data into visually useful images using nonlinear stretch algorithms.

A Rust port of the Python [processinator](../processinator) package, seeded from the native stretch pipeline in the [astra](../astra) desktop app (`src-tauri/src/stretch/`). Compute-heavy stages are parallelized with rayon; there is no GPU/JAX backend — the Rust implementation fills that performance role.

## Usage

```rust
use processinator::{fits_to_image, PipelineConfig};
use std::path::Path;

// High-level: FITS file → PNG (stretch only, like the Python defaults)
let config = PipelineConfig { gradient_removal: false, ..Default::default() };
fits_to_image("my_image.fits", Some(Path::new("stretched.png")), &config)?;

// With gradient removal and denoising
let config = PipelineConfig { denoise: true, ..Default::default() };
fits_to_image("my_image.fits", Some(Path::new("stretched.png")), &config)?;

// Low-level: in-memory data → stretched data
use processinator::{read_fits, stretch, StretchOptions, StretchAlgorithm};

let data = read_fits("my_image.fits")?;
let stretched = stretch(&data, &StretchOptions::new(StretchAlgorithm::mtf()));

// Full pipeline control
use processinator::process;

let result = process(&data, &PipelineConfig {
    gradient_removal: true,
    denoise: true,
    denoise_threshold: 3.0,
    ..Default::default()
});
```

Images are the planar [`Image`] type: one or three row-major `f64` channel planes, mirroring the `(H, W)` / `(H, W, 3)` numpy layouts of the Python library and the channel-first layout of astro FITS files.

## Algorithms

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| **MTF** (default) | General use | Midtones Transfer Function with offset background neutralization |
| **Arcsinh** | Color preservation | Inverse hyperbolic sine, maintains color ratios |
| **Log** | High dynamic range | Logarithmic stretch |
| **Linear** | Quick preview | Simple percentile-based clip and scale |
| **Statistical** | Consistent output | Gamma correction targeting a specific median |

Algorithm parameters live on the enum variants:

```rust
let algo = StretchAlgorithm::Mtf { bg_percent: 0.25, sigma: 2.0, linked: true };
let result = stretch(&data, &StretchOptions::new(algo));
```

The linked MTF is tuned on real one-shot-color captures (Seestar stacks):
per-channel background offsets are chosen so the sky comes out neutral
without rescaling channels (a multiplicative rescale rebalances signal by
noise ratios and kills H-alpha red), the shadow clip is anchored on the
lower quartile so frame-filling glow like the LMC survives, and the
pipeline finishes with SCNR-style green suppression (`green_removal`) and
a mild saturation boost (`saturation`) — both `PipelineConfig` fields,
also available directly as [`remove_green`] and [`saturate`].

## Denoising

`denoise()` implements starlet (à trous B3-spline) wavelet thresholding —
the standard astronomy approach (Starck & Murtagh). The noise level is
estimated automatically from the finest wavelet scale, so there is no
noise parameter to tune.

```rust
use processinator::{denoise, DenoiseParams, ThresholdMode};

let quiet = denoise(&data, &DenoiseParams::default());          // hard thresholding
let smooth = denoise(&data, &DenoiseParams {
    mode: ThresholdMode::Soft,                                  // smoother, for starless data
    ..Default::default()
});
let gentle = denoise(&data, &DenoiseParams {
    threshold: 2.0,                                             // threshold in noise sigmas
    ..Default::default()
});
```

Why it's astronomically safe:

- Stars and real structure produce wavelet coefficients far above the
  noise threshold and pass through untouched — hard mode preserves
  bright-star photometry exactly.
- The coarse residual (faint nebulosity, sky background) is never
  thresholded, so large-scale signal survives.

In the pipeline, denoising runs on the *stretched* image rather than the
linear data. Normalization and the MTF anchor their black point and
shadow clip on the noise width; denoising first collapses those
statistics, the black point climbs into the faintest real signal, and
dark nebulosity comes out inky. Post-stretch, the tone mapping is
identical with or without denoising.

## Synthetic test images

`processinator::synthetic` generates linear test frames with known ground
truth (star catalog, noise-free reference) — useful for testing any
processing code, not just this library's:

```rust
use processinator::synthetic::{make_test_image, SyntheticParams};
use processinator::write_fits;

let img = make_test_image(&SyntheticParams {
    height: 512,
    width: 512,
    rgb: true,
    gradient_amplitude: 300.0,  // sky gradient
    nebula_amplitude: 150.0,    // large-scale nebulosity
    dark_edges: (10, 0, 14, 0), // stacking artifacts
    hot_pixels: 20,
    seed: 13,
    ..Default::default()
});
img.data;   // observed frame (signal + noise)
img.clean;  // noise-free truth
img.stars;  // star catalog: x, y, peak amplitude
write_fits(&img.data, "test.fits")?;
```

The generator uses its own small deterministic RNG (SplitMix64), so a given
seed produces bit-identical images on every platform and library version.
Seeds do **not** reproduce the numpy images from the Python package.

`cargo run --release --example make_test_images` writes a set of example
FITS files (plus stretched PNG previews) to `examples/images/` for manual
experimentation.

## Showcase — real-capture comparison pages

`cargo run --release --example showcase` builds static comparison pages
from real captures dropped in `showcase/input/` (e.g. a Seestar FITS plus
its on-device JPEG): an interactive slider comparing any two renditions,
a side-by-side pipeline-stage grid, and a FITS-header-style processing
log with real timings from your machine. `tests/showcase_test.rs` also
runs those captures through the full pipeline as a smoke test. See
[showcase/README.md](showcase/README.md).

## Relationship to astra

Astra's `src-tauri/src/stretch/` module is the ancestor of this crate: its
MTF, gradient-removal, and autocrop code was brought over nearly verbatim
(medians/percentiles now use numpy-compatible interpolation to match the
Python reference). Astra's `generate_preview(fits, out, params)` maps to:

```rust
fits_to_image(fits_path, Some(out_path), &PipelineConfig {
    autocrop: true,
    gradient_removal: true,
    stretch: StretchAlgorithm::Mtf { bg_percent, sigma, linked: true },
    ..Default::default()
})
```

## License

AGPL-3.0
