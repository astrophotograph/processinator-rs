//! Image stretching algorithms for astronomy images.
//!
//! Converts linear FITS data (where most detail sits in low pixel values)
//! into visually useful images by applying nonlinear transfer functions.
//! The MTF implementation is brought over from astra's `stretch/mtf.rs`;
//! the remaining algorithms are ported from the Python processinator.

use rayon::prelude::*;

use crate::autocrop::{self, AutocropParams, CropBounds};
use crate::image::Image;
use crate::stats;

/// A stretch algorithm together with its tuning parameters.
///
/// Construct defaults with [`StretchAlgorithm::mtf`] and friends, then
/// adjust fields with struct-update syntax where needed.
#[derive(Debug, Clone, PartialEq)]
pub enum StretchAlgorithm {
    /// Midtones Transfer Function (GraXpert-style). Default. Good
    /// all-around choice; `linked` subtracts each channel's background
    /// level and applies one shared stretch, so channel ratios — and
    /// therefore color — survive.
    Mtf {
        /// Target background level (0-1).
        bg_percent: f64,
        /// Sigmas above background for shadow clipping.
        sigma: f64,
        /// Use the same stretch parameters for all channels.
        linked: bool,
    },
    /// Inverse hyperbolic sine. Preserves color ratios well.
    Arcsinh {
        /// Stretch aggressiveness; smaller = more aggressive.
        factor: f64,
    },
    /// Logarithmic stretch. Good for high dynamic range images.
    Log {
        /// Stretch aggressiveness; smaller = more aggressive.
        factor: f64,
    },
    /// Simple percentile-based linear stretch.
    Linear {
        low_percentile: f64,
        high_percentile: f64,
    },
    /// Percentile clip then gamma correction targeting a median brightness.
    Statistical {
        /// Desired median value after the stretch.
        target_median: f64,
        low_percentile: f64,
        high_percentile: f64,
    },
}

impl StretchAlgorithm {
    pub fn mtf() -> Self {
        Self::Mtf {
            bg_percent: 0.22,
            sigma: 2.0,
            linked: true,
        }
    }

    pub fn arcsinh() -> Self {
        Self::Arcsinh { factor: 0.15 }
    }

    pub fn log() -> Self {
        Self::Log { factor: 0.15 }
    }

    pub fn linear() -> Self {
        Self::Linear {
            low_percentile: 1.0,
            high_percentile: 99.0,
        }
    }

    pub fn statistical() -> Self {
        Self::Statistical {
            target_median: 0.15,
            low_percentile: 0.5,
            high_percentile: 99.9,
        }
    }
}

impl Default for StretchAlgorithm {
    fn default() -> Self {
        Self::mtf()
    }
}

/// Options for [`stretch`].
#[derive(Debug, Clone)]
pub struct StretchOptions {
    pub algorithm: StretchAlgorithm,
    /// Detect dark stacking edges and exclude them from the normalization
    /// statistics. The output keeps the full frame (nothing is cropped
    /// away). Ignored when `pre_normalized` is set.
    pub autocrop: bool,
    /// The data is already normalized to [0, 1]; skip the internal
    /// normalization pass (used by the pipeline to avoid redundant work).
    pub pre_normalized: bool,
}

impl Default for StretchOptions {
    fn default() -> Self {
        Self {
            algorithm: StretchAlgorithm::default(),
            autocrop: true,
            pre_normalized: false,
        }
    }
}

impl StretchOptions {
    pub fn new(algorithm: StretchAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }
}

/// Stretch image data from linear to nonlinear for display.
///
/// Input values may be in their original FITS range; the output is
/// normalized to [0, 1] with the same shape. Consumes the input — full
/// planes of a modern sensor run to hundreds of MB, so callers that need
/// the original clone explicitly rather than paying for a hidden copy.
pub fn stretch(mut image: Image, options: &StretchOptions) -> Image {
    if !options.pre_normalized {
        // Detect dark stacking edges and compute normalization statistics
        // from the interior only; the full frame is kept so pixel
        // coordinates stay aligned with the FITS.
        let bounds = if options.autocrop {
            autocrop::detect_edges(&image, &AutocropParams::default())
        } else {
            (0, 0, 0, 0)
        };
        normalize_to_01(&mut image, bounds);
    }

    match options.algorithm {
        StretchAlgorithm::Mtf {
            bg_percent,
            sigma,
            linked,
        } => stretch_mtf(&mut image, bg_percent, sigma, linked),
        StretchAlgorithm::Arcsinh { factor } => stretch_arcsinh(&mut image, factor),
        StretchAlgorithm::Log { factor } => stretch_log(&mut image, factor),
        StretchAlgorithm::Linear {
            low_percentile,
            high_percentile,
        } => stretch_linear(&mut image, low_percentile, high_percentile),
        StretchAlgorithm::Statistical {
            target_median,
            low_percentile,
            high_percentile,
        } => stretch_statistical(&mut image, target_median, low_percentile, high_percentile),
    }
    image
}

/// Normalize raw FITS data to [0, 1] in place, per channel, with
/// percentiles (0.1, 99.99) computed from the interior left by
/// `stats_bounds`. Percentile clipping keeps hot pixels and bright stars
/// from compressing the useful range; a constant channel comes back all
/// zero.
pub(crate) fn normalize_to_01(image: &mut Image, stats_bounds: CropBounds) {
    // One channel's interior scratch at a time — three parallel copies of
    // a 26 MP plane is real memory; the selection is O(n) so the apply
    // loop below dominates anyway
    let ranges: Vec<(f64, f64)> = (0..image.num_channels())
        .map(|c| {
            let mut interior = image.channel_interior(c, stats_bounds);
            stats::percentile_pair_in_place(&mut interior, 0.1, 99.99)
        })
        .collect();
    image
        .channels_mut()
        .par_iter_mut()
        .zip(ranges)
        .for_each(|(ch, (vmin, vmax))| {
            let range = vmax - vmin;
            if range > 0.0 {
                for v in ch.iter_mut() {
                    *v = ((*v - vmin) / range).clamp(0.0, 1.0);
                }
            } else {
                ch.fill(0.0);
            }
        });
}

// ---------------------------------------------------------------------------
// MTF (Midtones Transfer Function) — brought over from astra
// ---------------------------------------------------------------------------

/// Order statistics driving an MTF stretch solution.
///
/// Parameter-independent: computed once per channel from the pre-stretch
/// data, they determine the shadows and midtone for *any*
/// `(bg_percent, sigma)` — which is what lets a GPU display path recompute
/// the stretch per slider change without touching the pixels again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtfStats {
    /// Median of the valid samples.
    pub median: f64,
    /// Lower quartile of the valid samples.
    pub p25: f64,
    /// Median absolute deviation around the median.
    pub mad: f64,
    /// Number of valid samples the statistics were computed from.
    /// 0 means the channel had no usable data and the stretch is a no-op.
    pub count: usize,
}

/// The scalar solution an MTF stretch applies to the pixels:
/// `v' = MTF(midtone, clamp((v - shadows[c]) * scale, 0, 1))`.
///
/// Produced by [`mtf_display_solution`]; the same values drive the CPU
/// pipeline and any external (e.g. GPU shader) implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct MtfSolution {
    /// Per-channel shadow offset (one entry for mono).
    pub shadows: Vec<f64>,
    /// Common rescale applied after shadow subtraction.
    pub scale: f64,
    /// Shared midtone balance parameter.
    pub midtone: f64,
}

/// Statistics for the linked (color) MTF path: positive samples only,
/// MAD floored at 1e-6. A channel with no positive samples reports
/// `{median: 0, p25: 0, mad: 0.01, count: 0}` (the historical fallback).
pub fn mtf_stats_linked(data: &[f64]) -> MtfStats {
    match stats_from_valid(data.iter().copied().filter(|&v| v > 0.0).collect()) {
        Some((p25, median, mad, count)) => MtfStats {
            median,
            p25,
            mad: mad.max(1e-6),
            count,
        },
        None => MtfStats {
            median: 0.0,
            p25: 0.0,
            mad: 0.01,
            count: 0,
        },
    }
}

/// Statistics for the per-channel (mono/unlinked) MTF path: samples in
/// (0, 1) exclusive, no MAD floor. `count == 0` means the stretch is a
/// no-op for this channel.
pub fn mtf_stats_channel(data: &[f64]) -> MtfStats {
    match stats_from_valid(
        data.iter()
            .copied()
            .filter(|&v| v > 0.0 && v < 1.0)
            .collect(),
    ) {
        Some((p25, median, mad, count)) => MtfStats {
            median,
            p25,
            mad,
            count,
        },
        None => MtfStats {
            median: 0.0,
            p25: 0.0,
            mad: 0.0,
            count: 0,
        },
    }
}

/// (p25, median, mad, count) of the pre-filtered samples, or None if empty.
fn stats_from_valid(mut valid: Vec<f64>) -> Option<(f64, f64, f64, usize)> {
    if valid.is_empty() {
        return None;
    }
    let count = valid.len();
    let (p25, median) = stats::percentile_pair_in_place(&mut valid, 25.0, 50.0);
    for v in valid.iter_mut() {
        *v = (*v - median).abs();
    }
    let mad = stats::median_in_place(&mut valid);
    Some((p25, median, mad, count))
}

/// Midtone balance that lands a background at `median` on `bg_percent`.
fn midtone_for_background(median: f64, bg_percent: f64) -> f64 {
    if median > 0.0 && median < 1.0 && bg_percent > 0.0 {
        let m = median * (bg_percent - 1.0) / (2.0 * bg_percent * median - bg_percent - median);
        m.clamp(1e-4, 0.9999)
    } else {
        0.5
    }
}

/// Compute the MTF solution the pipeline would apply to this pre-stretch
/// image: the linked solution for color, the single-channel solution for
/// mono. The input must already be normalized to [0, 1] (the output of
/// [`crate::pipeline::prepare`]).
///
/// Applying `MTF(midtone, clamp((v - shadows[c]) * scale, 0, 1))` per pixel
/// reproduces `StretchAlgorithm::Mtf { linked: true, .. }` exactly — this is
/// the contract the WebGL display stretch in astra is built on.
pub fn mtf_display_solution(image: &Image, bg_percent: f64, sigma: f64) -> MtfSolution {
    if image.is_color() {
        mtf_linked_solution(image.channels(), bg_percent, sigma)
    } else {
        mtf_channel_solution(image.channel(0), bg_percent, sigma)
    }
}

fn stretch_mtf(image: &mut Image, bg_percent: f64, sigma: f64, linked: bool) {
    if linked && image.is_color() {
        stretch_mtf_linked(image.channels_mut(), bg_percent, sigma);
    } else {
        image
            .channels_mut()
            .par_iter_mut()
            .for_each(|ch| stretch_mtf_channel(ch, bg_percent, sigma));
    }
}

/// Linked RGB mode: subtract each channel's background level (offset
/// neutralization), then apply the same stretch to every channel.
///
/// The neutralization is offset-only on purpose: an earlier version also
/// rescaled channels multiplicatively to equalize their post-subtraction
/// medians, but those medians are set by per-channel *noise* width, so the
/// rescale rebalanced real signal by noise ratios — on real OSC captures it
/// crushed H-alpha red and shifted everything green. Subtracting the
/// per-channel background offset flattens the sky cast while leaving
/// channel ratios (color) intact.
fn stretch_mtf_linked(channels: &mut [Vec<f64>], bg_percent: f64, sigma: f64) {
    let sol = mtf_linked_solution(channels, bg_percent, sigma);
    channels.par_iter_mut().enumerate().for_each(|(i, ch)| {
        apply_shadow_scale(ch, sol.shadows[i], sol.scale);
        apply_mtf(ch, sol.midtone);
    });
}

/// Solution for the linked path — see [`stretch_mtf_linked`]'s docs for the
/// reasoning behind each step.
fn mtf_linked_solution(channels: &[Vec<f64>], bg_percent: f64, sigma: f64) -> MtfSolution {
    // Step 1: Per-channel statistics
    let stats: Vec<MtfStats> = channels.par_iter().map(|ch| mtf_stats_linked(ch)).collect();

    // Step 2: Shadow offsets that land every channel's median at the same
    // residual level, so the sky comes out neutral. Each channel's natural
    // residual is (median - p25) + sigma * MAD * 1.4826 — quartile-anchored
    // rather than median-anchored, so frame-filling glow (galaxies, large
    // nebulae) inflates the residual and survives the clip instead of being
    // subtracted as "background". Taking the smallest residual across
    // channels as the shared target removes the sky color cast by offset
    // alone; signal above the sky keeps its channel deltas exactly.
    let k = sigma * 1.4826;
    let residual = stats
        .iter()
        .map(|s| (s.median - s.p25) + k * s.mad)
        .fold(f64::INFINITY, f64::min);
    let shadows: Vec<f64> = stats
        .iter()
        .map(|s| (s.median - residual).max(0.0))
        .collect();

    // Step 3: One common rescale after shadow subtraction so the channels
    // keep their relative scale
    let max_shadow = shadows.iter().copied().fold(0.0, f64::max);
    let scale = 1.0 / (1.0 - max_shadow).max(1e-6);

    // Step 4: Shared midtone from the reference channel (green), as it
    // looks after shadow subtraction
    let ref_idx = std::cmp::min(1, channels.len() - 1);
    let mut mapped: Vec<f64> = channels[ref_idx]
        .iter()
        .map(|&v| ((v - shadows[ref_idx]) * scale).clamp(0.0, 1.0))
        .filter(|&v| v > 0.0)
        .collect();
    let ref_median = if mapped.is_empty() {
        0.0
    } else {
        stats::median_in_place(&mut mapped)
    };
    let midtone = midtone_for_background(ref_median, bg_percent);

    MtfSolution {
        shadows,
        scale,
        midtone,
    }
}

/// SCNR-style green suppression (average-neutral): pull green down toward
/// the mean of red and blue where it dominates. Almost nothing in deep sky
/// is truly green, so a green cast is sensor response, not signal. `amount`
/// blends between untouched (0.0) and fully suppressed (1.0); mono images
/// are returned untouched.
pub fn remove_green(image: &mut Image, amount: f64) {
    if !image.is_color() || amount <= 0.0 {
        return;
    }
    let amount = amount.min(1.0);
    if let [r, g, b] = image.channels_mut() {
        for i in 0..r.len() {
            let neutral = (r[i] + b[i]) * 0.5;
            if g[i] > neutral {
                g[i] -= amount * (g[i] - neutral);
            }
        }
    }
}

/// Scale chroma around per-pixel luminance: `factor` > 1 boosts saturation,
/// < 1 mutes it, 1.0 is a no-op. Values stay clamped to [0, 1]; mono images
/// are returned untouched.
///
/// Nonlinear stretches compress channel ratios, so stretched astro images
/// come out grayer than the linear data; a modest boost (1.2-1.4) after
/// stretching restores the color.
pub fn saturate(image: &mut Image, factor: f64) {
    if !image.is_color() || factor == 1.0 {
        return;
    }
    let n = image.pixels_per_channel();
    let lum: Vec<f64> = {
        let (r, g, b) = (image.channel(0), image.channel(1), image.channel(2));
        (0..n).map(|i| (r[i] + g[i] + b[i]) / 3.0).collect()
    };
    image.channels_mut().par_iter_mut().for_each(|ch| {
        for (v, &l) in ch.iter_mut().zip(&lum) {
            *v = (l + factor * (*v - l)).clamp(0.0, 1.0);
        }
    });
}

/// Unlinked mode (or grayscale): stretch one channel independently.
fn stretch_mtf_channel(data: &mut [f64], bg_percent: f64, sigma: f64) {
    let sol = mtf_channel_solution(data, bg_percent, sigma);
    apply_shadow_scale(data, sol.shadows[0], sol.scale);
    apply_mtf(data, sol.midtone);
}

/// Solution for one independent channel. A channel with no valid samples
/// gets the identity solution (shadow 0, scale 1, midtone 0.5 — MTF at
/// 0.5 is the identity map).
fn mtf_channel_solution(data: &[f64], bg_percent: f64, sigma: f64) -> MtfSolution {
    let stats = mtf_stats_channel(data);
    if stats.count == 0 {
        return MtfSolution {
            shadows: vec![0.0],
            scale: 1.0,
            midtone: 0.5,
        };
    }

    // Lower-quartile anchor — see mtf_linked_solution for why not the median
    let shadow_clip = (stats.p25 - sigma * stats.mad * 1.4826).max(0.0);
    let range = 1.0 - shadow_clip;

    // Midtone balance that lands the background at bg_percent
    let median_norm = (stats.median - shadow_clip) / range;
    MtfSolution {
        shadows: vec![shadow_clip],
        scale: 1.0 / range,
        midtone: midtone_for_background(median_norm, bg_percent),
    }
}

/// `v' = clamp((v - shadow) * scale, 0, 1)` — the pre-MTF linear map.
fn apply_shadow_scale(data: &mut [f64], shadow: f64, scale: f64) {
    for v in data.iter_mut() {
        *v = ((*v - shadow) * scale).clamp(0.0, 1.0);
    }
}

/// MTF(m, x) = (m - 1) * x / ((2m - 1) * x - m)
#[inline]
fn apply_mtf(data: &mut [f64], m: f64) {
    let m_minus_1 = m - 1.0;
    let two_m_minus_1 = 2.0 * m - 1.0;

    for v in data.iter_mut() {
        let x = *v;
        let denom = two_m_minus_1 * x - m;
        *v = if denom.abs() < 1e-10 {
            x
        } else {
            (m_minus_1 * x / denom).clamp(0.0, 1.0)
        };
    }
}

// ---------------------------------------------------------------------------
// Arcsinh stretch
// ---------------------------------------------------------------------------

fn stretch_arcsinh(image: &mut Image, factor: f64) {
    let scale = 1.0 / factor;
    let denom = scale.asinh();
    image.channels_mut().par_iter_mut().for_each(|ch| {
        for v in ch.iter_mut() {
            *v = ((*v * scale).asinh() / denom).clamp(0.0, 1.0);
        }
    });
}

// ---------------------------------------------------------------------------
// Log stretch
// ---------------------------------------------------------------------------

fn stretch_log(image: &mut Image, factor: f64) {
    let offset = factor * 0.01;
    let denom = (1.0 / offset).ln_1p();
    image.channels_mut().par_iter_mut().for_each(|ch| {
        for v in ch.iter_mut() {
            *v = ((*v / offset).ln_1p() / denom).clamp(0.0, 1.0);
        }
    });
}

// ---------------------------------------------------------------------------
// Linear stretch
// ---------------------------------------------------------------------------

fn stretch_linear(image: &mut Image, low_percentile: f64, high_percentile: f64) {
    // Percentiles span all channels together, matching numpy operating on
    // the full (H, W, 3) array
    let mut all: Vec<f64> = image
        .channels()
        .iter()
        .flat_map(|ch| ch.iter().copied())
        .collect();
    let (vmin, vmax) = stats::percentile_pair_in_place(&mut all, low_percentile, high_percentile);
    // The flattened copy is a full extra image; release it before mutating
    drop(all);
    let range = vmax - vmin;
    if range == 0.0 {
        return;
    }
    image.channels_mut().par_iter_mut().for_each(|ch| {
        for v in ch.iter_mut() {
            *v = ((*v - vmin) / range).clamp(0.0, 1.0);
        }
    });
}

// ---------------------------------------------------------------------------
// Statistical stretch (gamma correction)
// ---------------------------------------------------------------------------

fn stretch_statistical(
    image: &mut Image,
    target_median: f64,
    low_percentile: f64,
    high_percentile: f64,
) {
    let mut all: Vec<f64> = image
        .channels()
        .iter()
        .flat_map(|ch| ch.iter().copied())
        .collect();
    let (vmin, vmax) = stats::percentile_pair_in_place(&mut all, low_percentile, high_percentile);
    // The flattened copy is a full extra image; release it before mutating
    drop(all);
    let range = vmax - vmin;
    if range == 0.0 {
        return;
    }

    image.channels_mut().par_iter_mut().for_each(|ch| {
        for v in ch.iter_mut() {
            *v = ((*v - vmin) / range).clamp(0.0, 1.0);
        }
    });

    // Gamma correction to bring the median of the clipped result to target
    let mut positives: Vec<f64> = image
        .channels()
        .iter()
        .flat_map(|ch| ch.iter().copied())
        .filter(|&v| v > 0.0)
        .collect();
    if positives.is_empty() {
        return;
    }
    let current_median = stats::median_in_place(&mut positives);
    drop(positives);
    if current_median > 0.0 && current_median != target_median {
        let gamma = (target_median.ln() / current_median.ln()).clamp(0.2, 5.0);
        image.channels_mut().par_iter_mut().for_each(|ch| {
            for v in ch.iter_mut() {
                *v = v.powf(gamma).clamp(0.0, 1.0);
            }
        });
    }
}
