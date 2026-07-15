# Showcase — capability pages from real captures

Static comparison pages built from real telescope data (e.g. Seestar
stacks), for eyeballing the library against the camera's own processing
and for blog/video material. Each target gets an interactive slider
comparison, a side-by-side stage grid, and a FITS-header-style processing
log with real timings from this machine.

## Usage

1. Drop captures into `showcase/input/`:
   - `<target>.fit` (or `.fits` / `.fts`) — the raw stacked FITS
   - `<target>.jpg` (optional) — the Seestar's own on-device JPEG, matched
     to the FITS by filename stem
2. Generate the pages:

   ```sh
   cargo run --release --example showcase
   ```

3. Open `showcase/index.html` in a browser.

Generated artifacts: `index.html`, one `<target>.html` per capture,
rendered PNGs plus copied JPEGs under `img/`, and machine-readable
timings in `results.json`.

## Pipeline variants

Each FITS is processed four ways, each timed end to end (pipeline +
8-bit conversion; PNG encoding timed separately):

| Stage | Config |
|---|---|
| linear (no stretch) | percentile-linear scaling only — what the sensor recorded |
| MTF stretch | default midtones transfer function stretch |
| + gradient removal | order-2 polynomial background removal, then MTF |
| + starlet denoise | gradient removal, starlet wavelet denoise, then MTF |

## As a test

`tests/showcase_test.rs` runs every FITS in `showcase/input/` through the
full pipeline and sanity-checks the output (dimensions preserved, finite,
in [0, 1], non-constant). With no images present it passes trivially, so
CI does not need the binary fixtures.

Keep large captures out of version control if the repo is meant to stay
lean — `input/` and `img/` are the directories that grow.
