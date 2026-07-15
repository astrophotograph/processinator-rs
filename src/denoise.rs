//! Astronomically-safe denoising via starlet wavelet thresholding.
//!
//! Uses the isotropic undecimated wavelet transform (the "starlet" /
//! à trous transform with a B3-spline kernel) that is standard in
//! astronomy image processing (Starck & Murtagh, "Astronomical Image and
//! Data Analysis").
//!
//! Why it is safe for astrophotos:
//! - Stars and real structure produce wavelet coefficients far above the
//!   noise threshold and pass through (nearly) untouched.
//! - The coarse residual — faint nebulosity, sky background — is never
//!   thresholded, so no large-scale signal is destroyed.
//! - The noise level is estimated from the data itself (MAD of the finest
//!   wavelet scale), so no manual noise parameter is needed.
//!
//! Works on linear or stretched data; the estimate is scale-free.

use rayon::prelude::*;

use crate::image::Image;
use crate::stats;

/// B3-spline scaling kernel used by the à trous transform.
const B3: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Noise standard deviation of each starlet scale for unit-variance
/// Gaussian input (2D, B3 spline). Precomputed constants from
/// Starck & Murtagh.
const NOISE_LEVELS: [f64; 7] = [0.8907, 0.2007, 0.0855, 0.0412, 0.0203, 0.0102, 0.0051];

/// How detail coefficients below the threshold are treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThresholdMode {
    /// Keep-or-kill. Preserves star photometry exactly — the standard
    /// choice for astronomy.
    #[default]
    Hard,
    /// Shrink every coefficient. Visually smoother background but
    /// systematically dims point sources — fine for starless or
    /// background-dominated data.
    Soft,
}

/// Tuning for [`denoise`].
#[derive(Debug, Clone)]
pub struct DenoiseParams {
    /// Number of wavelet detail scales to threshold. More scales remove
    /// lower-frequency noise but risk touching real structure. Clamped
    /// automatically for small images.
    pub n_scales: usize,
    /// Threshold in units of the estimated noise sigma. The finest scale
    /// uses `threshold + 1` (standard practice — it is almost pure noise).
    /// Higher = stronger smoothing.
    pub threshold: f64,
    pub mode: ThresholdMode,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            n_scales: 4,
            threshold: 3.0,
            mode: ThresholdMode::Hard,
        }
    }
}

/// Denoise image data using starlet wavelet thresholding.
///
/// Decomposes each channel into wavelet detail scales plus a smooth
/// residual, estimates the noise sigma from the finest scale, thresholds
/// the detail coefficients, and reconstructs. Coefficients well above the
/// noise (stars, structure) survive; the smooth residual (nebulosity,
/// background) is never touched. Values are not clipped or rescaled.
pub fn denoise(image: &Image, params: &DenoiseParams) -> Image {
    let w = image.width();
    let h = image.height();
    let channels: Vec<Vec<f64>> = image
        .channels()
        .iter()
        .map(|ch| denoise_channel(ch, w, h, params))
        .collect();
    Image::from_channels(w, h, channels)
}

fn denoise_channel(
    channel: &[f64],
    width: usize,
    height: usize,
    params: &DenoiseParams,
) -> Vec<f64> {
    // The kernel span at scale j is 4 * 2^j + 1 pixels; keep it inside the image
    let min_dim = std::cmp::min(width, height);
    let mut max_scales = 1;
    while 4 * (1usize << max_scales) + 1 < min_dim && max_scales < NOISE_LEVELS.len() - 1 {
        max_scales += 1;
    }
    let n_scales = params.n_scales.clamp(1, max_scales);

    // À trous decomposition: detail scales + smooth residual
    let mut c = channel.to_vec();
    let mut details = Vec::with_capacity(n_scales);
    for j in 0..n_scales {
        let smooth = b3_smooth(&c, 1 << j, width, height);
        let detail: Vec<f64> = c.iter().zip(&smooth).map(|(a, b)| a - b).collect();
        details.push(detail);
        c = smooth;
    }

    // Estimate noise sigma from the finest detail scale via MAD. Detail
    // coefficients are zero-mean; stars are sparse enough that the median
    // of |w| is set by the noise.
    let mut abs_finest: Vec<f64> = details[0].iter().map(|v| v.abs()).collect();
    let sigma = stats::median_in_place(&mut abs_finest) / 0.6745 / NOISE_LEVELS[0];
    if sigma <= 0.0 {
        // No detectable noise — return the input unchanged
        return channel.to_vec();
    }

    // Threshold detail scales and reconstruct (residual passes untouched)
    let mut result = c;
    for (j, detail) in details.iter().enumerate() {
        let k = if j == 0 {
            params.threshold + 1.0
        } else {
            params.threshold
        };
        let t = k * sigma * noise_level(j);
        match params.mode {
            ThresholdMode::Soft => {
                result
                    .par_iter_mut()
                    .zip(detail)
                    .for_each(|(r, &d)| *r += d.signum() * (d.abs() - t).max(0.0));
            }
            ThresholdMode::Hard => {
                result
                    .par_iter_mut()
                    .zip(detail)
                    .for_each(|(r, &d)| *r += if d.abs() > t { d } else { 0.0 });
            }
        }
    }
    result
}

/// Noise std of a starlet scale for unit-variance input noise.
fn noise_level(scale: usize) -> f64 {
    if scale < NOISE_LEVELS.len() {
        NOISE_LEVELS[scale]
    } else {
        // Beyond the table each scale roughly halves
        NOISE_LEVELS[NOISE_LEVELS.len() - 1] * 0.5f64.powi((scale - NOISE_LEVELS.len() + 1) as i32)
    }
}

/// Separable B3-spline smoothing with holes ("à trous") of size `step`.
fn b3_smooth(data: &[f64], step: usize, width: usize, height: usize) -> Vec<f64> {
    let vertical = b3_pass_vertical(data, step, width, height);
    b3_pass_horizontal(&vertical, step, width, height)
}

/// Vertical pass: out[y][x] = sum_k B3[k] * in[reflect(y + (k-2)*step)][x].
fn b3_pass_vertical(data: &[f64], step: usize, width: usize, height: usize) -> Vec<f64> {
    let mut out = vec![0.0; data.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let sources: Vec<&[f64]> = (0..5)
            .map(|k| {
                let sy = reflect(y as isize + (k as isize - 2) * step as isize, height);
                &data[sy * width..(sy + 1) * width]
            })
            .collect();
        for x in 0..width {
            row[x] = B3[0] * sources[0][x]
                + B3[1] * sources[1][x]
                + B3[2] * sources[2][x]
                + B3[3] * sources[3][x]
                + B3[4] * sources[4][x];
        }
    });
    out
}

/// Horizontal pass: out[y][x] = sum_k B3[k] * in[y][reflect(x + (k-2)*step)].
fn b3_pass_horizontal(data: &[f64], step: usize, width: usize, height: usize) -> Vec<f64> {
    let _ = height;
    let mut out = vec![0.0; data.len()];
    out.par_chunks_mut(width)
        .zip(data.par_chunks(width))
        .for_each(|(row_out, row_in)| {
            for (x, out_v) in row_out.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (k, &weight) in B3.iter().enumerate() {
                    let sx = reflect(x as isize + (k as isize - 2) * step as isize, width);
                    acc += weight * row_in[sx];
                }
                *out_v = acc;
            }
        });
    out
}

/// Mirror an index into [0, n) without repeating the edge sample —
/// numpy's `pad(mode="reflect")`: `[3 2 | 1 2 3 | 2 1]`.
fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}
