//! Processing pipeline for astronomy images.
//!
//! Composes processing steps (gradient removal, denoising, stretching)
//! into a configurable pipeline. Each step can be enabled or disabled
//! independently.
//!
//! Stage order: autocrop detection → normalize → gradient removal →
//! renormalize → stretch → denoise → green removal → saturation. All
//! stages after normalization operate on [0, 1] data, so they compose
//! without rescaling tricks. Denoising runs on the stretched image so the
//! tone mapping (anchored on noise statistics) is identical with or
//! without it.

use crate::autocrop::{self, AutocropParams, CropBounds};
use crate::denoise::{self, DenoiseParams, ThresholdMode};
use crate::enhance;
use crate::gradient::{self, GradientParams};
use crate::image::Image;
use crate::stretch::{self, StretchAlgorithm, StretchOptions};

/// Configuration for the processing pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Detect dark stacking edges and exclude them from all statistics
    /// (the frame itself is never cropped).
    pub autocrop: bool,
    /// Enable background gradient removal.
    pub gradient_removal: bool,
    /// Polynomial order for the gradient model (1-3).
    pub gradient_order: usize,
    /// Sigma clip for gradient background sampling.
    pub gradient_sigma: f64,
    /// Enable starlet wavelet denoising.
    pub denoise: bool,
    /// Number of wavelet scales to threshold.
    pub denoise_scales: usize,
    /// Denoise threshold in noise sigmas.
    pub denoise_threshold: f64,
    /// Hard or soft thresholding.
    pub denoise_mode: ThresholdMode,
    /// Corner-sampled background color neutralization before the stretch
    /// (RGB only) — the Python processing flow's "color calibration".
    pub color_calibration: bool,
    /// Stretch algorithm (with its parameters) to finish with.
    pub stretch: StretchAlgorithm,
    /// Post-stretch mean-anchored contrast boost; 1.0 disables.
    pub contrast: f64,
    /// Post-stretch star dimming to emphasize nebulosity.
    pub star_reduction: bool,
    /// Luminance threshold for star detection (0-1).
    pub star_threshold: f64,
    /// SCNR-style green suppression after the stretch, 0.0 (off) to 1.0
    /// (full). Deep-sky signal is almost never green; this removes the
    /// green cast OSC sensors leave behind.
    pub green_removal: f64,
    /// Post-stretch saturation boost for color images (1.0 = off).
    /// Nonlinear stretches compress channel ratios; a modest boost restores
    /// the color the linear data had.
    pub saturation: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            autocrop: true,
            gradient_removal: true,
            gradient_order: 2,
            gradient_sigma: 2.5,
            denoise: false,
            denoise_scales: 4,
            denoise_threshold: 3.0,
            denoise_mode: ThresholdMode::Hard,
            color_calibration: false,
            stretch: StretchAlgorithm::default(),
            contrast: 1.0,
            star_reduction: false,
            star_threshold: 0.8,
            green_removal: 1.0,
            saturation: 1.25,
        }
    }
}

/// The pre-stretch prefix of [`process`]: autocrop edge detection,
/// normalization to [0, 1], optional gradient removal, renormalization.
///
/// The result is what the stretch consumes — pass it to
/// [`crate::stretch::stretch`] with `pre_normalized: true`, or hand it to a
/// display stretch (e.g. astra's WebGL preview) together with
/// [`crate::stretch::mtf_display_solution`].
///
/// Consumes the input and transforms it in place: full planes of a modern
/// sensor run to hundreds of MB, so callers that need the original clone
/// explicitly rather than paying for a hidden copy.
pub fn prepare(mut image: Image, config: &PipelineConfig) -> Image {
    // Detect dark stacking edges once; use the interior for all statistics
    let crop: CropBounds = if config.autocrop {
        autocrop::detect_edges(&image, &AutocropParams::default())
    } else {
        (0, 0, 0, 0)
    };

    // Normalize to [0, 1]; percentiles come from the interior
    stretch::normalize_to_01(&mut image, crop);

    if config.gradient_removal {
        gradient::remove_gradient(
            &mut image,
            &GradientParams {
                order: config.gradient_order,
                sigma_clip: config.gradient_sigma,
                ..Default::default()
            },
        );
        // Gradient removal shifts the data range; renormalize so the
        // stretch sees the full [0, 1] span (keeps bright stars near white)
        stretch::normalize_to_01(&mut image, crop);
    }

    image
}

/// Run the processing pipeline on raw FITS image data.
///
/// Returns the processed image normalized to [0, 1], same shape as the
/// input. Consumes the input (see [`prepare`]).
pub fn process(image: Image, config: &PipelineConfig) -> Image {
    let mut result = prepare(image, config);

    // Color calibration operates on the linear (pre-stretch) data, after
    // gradient removal — matching the Python processing flow's order
    if config.color_calibration {
        enhance::color_calibrate(&mut result);
    }

    let mut stretched = stretch::stretch(
        result,
        &StretchOptions {
            algorithm: config.stretch.clone(),
            autocrop: false,
            pre_normalized: true,
        },
    );

    // Denoise the stretched image, not the linear data: normalization and
    // the stretch anchor their statistics on the noise width (percentile
    // black point, MAD shadow clip), so denoising first collapses those
    // statistics and the black point climbs into the faintest real signal —
    // dark nebulosity came out inky. After the stretch the tone mapping is
    // identical with or without denoising, and the smoothing also takes the
    // post-stretch chroma speckle with it.
    if config.denoise {
        stretched = denoise::denoise(
            &stretched,
            &DenoiseParams {
                n_scales: config.denoise_scales,
                threshold: config.denoise_threshold,
                mode: config.denoise_mode,
            },
        );
        // Starlet reconstruction can overshoot slightly at hard edges
        for ch in stretched.channels_mut() {
            for v in ch.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
        }
    }

    // Post-stretch enhancements in the Python flow's order: contrast, then
    // star reduction, before the color cosmetics
    if config.contrast > 1.0 {
        enhance::contrast(&mut stretched, config.contrast);
    }
    if config.star_reduction {
        enhance::reduce_stars(&mut stretched, config.star_threshold);
    }

    stretch::remove_green(&mut stretched, config.green_removal);
    stretch::saturate(&mut stretched, config.saturation);
    stretched
}
