//! Synthetic astrophotography test images with known ground truth.
//!
//! Generates realistic linear "FITS-like" data — star field, nebulosity,
//! sky gradient, read noise, dark stacking edges, hot pixels — so that
//! stretching, autocrop, gradient removal, and denoising can be tested
//! against a noise-free reference.

use std::f64::consts::TAU;

use crate::image::Image;

/// One generated star: position and peak amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    pub x: f64,
    pub y: f64,
    pub amplitude: f64,
}

/// A generated test image plus the ground truth used to build it.
#[derive(Debug, Clone)]
pub struct SyntheticImage {
    /// Observed image (signal + noise + artifacts). Linear, in raw
    /// ADU-like units.
    pub data: Image,
    /// Noise-free signal (stars + nebula + background + gradient, with
    /// dark edges applied). Same shape as `data`.
    pub clean: Image,
    /// Star catalog with per-star peak amplitude.
    pub stars: Vec<Star>,
    /// Sky background level in ADU.
    pub background: f64,
    /// Gaussian read-noise sigma in ADU.
    pub noise_sigma: f64,
    /// `(top, bottom, left, right)` dark edge widths in pixels.
    pub dark_edges: (usize, usize, usize, usize),
}

/// Configuration for [`make_test_image`]. Defaults match the Python
/// processinator; override fields with struct-update syntax:
///
/// ```
/// use processinator::synthetic::{make_test_image, SyntheticParams};
///
/// let img = make_test_image(&SyntheticParams {
///     rgb: true,
///     gradient_amplitude: 300.0,
///     seed: 13,
///     ..Default::default()
/// });
/// ```
#[derive(Debug, Clone)]
pub struct SyntheticParams {
    pub height: usize,
    pub width: usize,
    /// Produce an RGB image with a color cast set by `channel_balance`
    /// (exercises linked-channel background neutralization).
    pub rgb: bool,
    /// Number of stars, log-uniform amplitudes.
    pub n_stars: usize,
    /// Gaussian PSF sigma in pixels.
    pub psf_sigma: f64,
    /// `(min, max)` star peak amplitude in ADU.
    pub star_amplitude: (f64, f64),
    /// Sky background level in ADU.
    pub background: f64,
    /// Gaussian read-noise sigma in ADU. 0 disables noise.
    pub noise_sigma: f64,
    /// Peak amplitude of a smooth linear+bilinear sky gradient in ADU.
    /// 0 disables.
    pub gradient_amplitude: f64,
    /// Peak amplitude of smooth large-scale nebulosity in ADU. 0 disables.
    pub nebula_amplitude: f64,
    /// `(top, bottom, left, right)` widths of dark stacking edges to
    /// simulate (signal multiplied by 0.02 there).
    pub dark_edges: (usize, usize, usize, usize),
    /// Number of isolated hot pixels added to the observed data (not the
    /// clean reference).
    pub hot_pixels: usize,
    /// Per-channel signal multipliers for RGB images.
    pub channel_balance: (f64, f64, f64),
    /// RNG seed; the same seed always produces the same image.
    pub seed: u64,
}

impl Default for SyntheticParams {
    fn default() -> Self {
        Self {
            height: 256,
            width: 256,
            rgb: false,
            n_stars: 80,
            psf_sigma: 1.6,
            star_amplitude: (200.0, 40000.0),
            background: 800.0,
            noise_sigma: 15.0,
            gradient_amplitude: 0.0,
            nebula_amplitude: 0.0,
            dark_edges: (0, 0, 0, 0),
            hot_pixels: 0,
            channel_balance: (1.0, 0.85, 0.7),
            seed: 0,
        }
    }
}

/// Generate a synthetic linear astrophotograph with known ground truth.
pub fn make_test_image(params: &SyntheticParams) -> SyntheticImage {
    let mut rng = Rng::new(params.seed);
    let h = params.height;
    let w = params.width;
    let n = w * h;

    // Star field: positions uniform (margin keeps PSF cores in-frame),
    // amplitudes log-uniform so a few stars dominate — like real frames.
    let margin = 4.0 * params.psf_sigma;
    let xs: Vec<f64> = (0..params.n_stars)
        .map(|_| rng.uniform(margin, w as f64 - margin))
        .collect();
    let ys: Vec<f64> = (0..params.n_stars)
        .map(|_| rng.uniform(margin, h as f64 - margin))
        .collect();
    let (lo, hi) = params.star_amplitude;
    let stars: Vec<Star> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| Star {
            x,
            y,
            amplitude: rng.uniform(lo.ln(), hi.ln()).exp(),
        })
        .collect();

    let mut signal = render_gaussians(w, h, &stars, params.psf_sigma);

    if params.nebula_amplitude > 0.0 {
        let nebula = render_nebula(w, h, params.nebula_amplitude, &mut rng);
        for (s, v) in signal.iter_mut().zip(nebula) {
            *s += v;
        }
    }

    if params.gradient_amplitude > 0.0 {
        let gradient = render_gradient(w, h, params.gradient_amplitude);
        for (s, v) in signal.iter_mut().zip(gradient) {
            *s += v;
        }
    }

    for s in signal.iter_mut() {
        *s += params.background;
    }

    // Expand to RGB with a color cast
    let mut clean = if params.rgb {
        let (br, bg, bb) = params.channel_balance;
        Image::new_rgb(
            w,
            h,
            [
                signal.iter().map(|&v| v * br).collect(),
                signal.iter().map(|&v| v * bg).collect(),
                signal.iter().map(|&v| v * bb).collect(),
            ],
        )
    } else {
        Image::new_mono(w, h, signal)
    };

    // Dark stacking edges attenuate everything, including the noise floor
    let (top, bottom, left, right) = params.dark_edges;
    if top + bottom + left + right > 0 {
        let mut mask = vec![1.0f64; n];
        for y in 0..h {
            for x in 0..w {
                if y < top || y >= h - bottom || x < left || x >= w - right {
                    mask[y * w + x] = 0.02;
                }
            }
        }
        for ch in clean.channels_mut() {
            for (v, m) in ch.iter_mut().zip(&mask) {
                *v *= m;
            }
        }
    }

    let mut data = clean.clone();
    if params.noise_sigma > 0.0 {
        for ch in data.channels_mut() {
            for v in ch.iter_mut() {
                *v += rng.normal(params.noise_sigma);
            }
        }
    }

    if params.hot_pixels > 0 {
        let hys: Vec<usize> = (0..params.hot_pixels).map(|_| rng.below(h)).collect();
        let hxs: Vec<usize> = (0..params.hot_pixels).map(|_| rng.below(w)).collect();
        let hot_value = params.star_amplitude.1 * 1.5;
        for (&hy, &hx) in hys.iter().zip(&hxs) {
            for c in 0..data.num_channels() {
                data.set(hx, hy, c, hot_value);
            }
        }
    }

    for ch in data.channels_mut() {
        for v in ch.iter_mut() {
            *v = v.max(0.0);
        }
    }

    SyntheticImage {
        data,
        clean,
        stars,
        background: params.background,
        noise_sigma: params.noise_sigma,
        dark_edges: params.dark_edges,
    }
}

/// Render Gaussian point sources onto a zero image. Each source is drawn
/// on a small stamp (±4 sigma) for speed.
fn render_gaussians(w: usize, h: usize, stars: &[Star], sigma: f64) -> Vec<f64> {
    let mut img = vec![0.0f64; w * h];
    let r = (4.0 * sigma).ceil() as i64;
    let denom = 2.0 * sigma * sigma;
    for star in stars {
        let xi = star.x as i64;
        let yi = star.y as i64;
        let x0 = (xi - r).max(0) as usize;
        let x1 = ((xi + r + 1).min(w as i64)).max(0) as usize;
        let y0 = (yi - r).max(0) as usize;
        let y1 = ((yi + r + 1).min(h as i64)).max(0) as usize;
        for y in y0..y1 {
            let dy2 = (y as f64 - star.y) * (y as f64 - star.y);
            for x in x0..x1 {
                let dx2 = (x as f64 - star.x) * (x as f64 - star.x);
                img[y * w + x] += star.amplitude * (-(dx2 + dy2) / denom).exp();
            }
        }
    }
    img
}

/// Smooth large-scale nebulosity: a handful of wide Gaussian blobs.
fn render_nebula(w: usize, h: usize, amplitude: f64, rng: &mut Rng) -> Vec<f64> {
    const N_BLOBS: usize = 5;
    let xs: Vec<f64> = (0..N_BLOBS)
        .map(|_| rng.uniform(0.2 * w as f64, 0.8 * w as f64))
        .collect();
    let ys: Vec<f64> = (0..N_BLOBS)
        .map(|_| rng.uniform(0.2 * h as f64, 0.8 * h as f64))
        .collect();
    let amps: Vec<f64> = (0..N_BLOBS).map(|_| rng.uniform(0.3, 1.0)).collect();
    let min_dim = w.min(h) as f64;
    let sigmas: Vec<f64> = (0..N_BLOBS)
        .map(|_| rng.uniform(0.08, 0.25) * min_dim)
        .collect();

    let mut nebula = vec![0.0f64; w * h];
    for i in 0..N_BLOBS {
        let denom = 2.0 * sigmas[i] * sigmas[i];
        for y in 0..h {
            let dy2 = (y as f64 - ys[i]) * (y as f64 - ys[i]);
            for x in 0..w {
                let dx2 = (x as f64 - xs[i]) * (x as f64 - xs[i]);
                nebula[y * w + x] += amps[i] * (-(dx2 + dy2) / denom).exp();
            }
        }
    }
    let peak = nebula.iter().copied().fold(0.0f64, f64::max);
    if peak > 0.0 {
        let scale = amplitude / peak;
        for v in nebula.iter_mut() {
            *v *= scale;
        }
    }
    nebula
}

/// Smooth sky gradient: linear tilt plus a bilinear term, peak = amplitude.
fn render_gradient(w: usize, h: usize, amplitude: f64) -> Vec<f64> {
    let w_max = (w.saturating_sub(1)).max(1) as f64;
    let h_max = (h.saturating_sub(1)).max(1) as f64;
    let mut surface = vec![0.0f64; w * h];
    let mut max = f64::NEG_INFINITY;
    for y in 0..h {
        let yn = y as f64 / h_max;
        for x in 0..w {
            let xn = x as f64 / w_max;
            let v = 0.5 * xn + 0.3 * yn + 0.2 * xn * yn;
            surface[y * w + x] = v;
            max = max.max(v);
        }
    }
    for v in surface.iter_mut() {
        *v = amplitude * (*v / max);
    }
    surface
}

/// Deterministic RNG (SplitMix64 + Box-Muller). Not cryptographic —
/// chosen over the `rand` crate so the same seed yields bit-identical
/// test images on every platform and library version, forever.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 bits of precision.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Zero-mean Gaussian via Box-Muller.
    fn normal(&mut self, sigma: f64) -> f64 {
        let u1 = 1.0 - self.unit(); // (0, 1] keeps ln finite
        let u2 = self.unit();
        sigma * (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Uniform integer in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
