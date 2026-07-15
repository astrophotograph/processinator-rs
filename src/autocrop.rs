//! Auto-crop dark or noisy stacking edges from astrophotography images.
//!
//! Alt-az mounts and poor guiding produce dark bars or noisy edges after
//! stacking. This module detects them so stretching and gradient removal
//! can compute statistics from the clean interior. Brought over from
//! astra's `stretch/autocrop.rs`, extended with the crop application and
//! safety checks of the Python processinator.

use rayon::prelude::*;

use crate::image::Image;
use crate::stats;

/// Pixels detected as dark edge on each side: `(top, bottom, left, right)`.
pub type CropBounds = (usize, usize, usize, usize);

/// Tuning for dark-edge detection.
#[derive(Debug, Clone)]
pub struct AutocropParams {
    /// An edge is dark if its median is below this fraction of the
    /// interior median.
    pub threshold: f64,
    /// Minimum fraction of a dimension a dark run must span to count
    /// (avoids cropping single-pixel noise).
    pub min_crop_fraction: f64,
    /// Maximum fraction of a dimension to crop from each edge.
    pub max_crop_fraction: f64,
}

impl Default for AutocropParams {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            min_crop_fraction: 0.01,
            max_crop_fraction: 0.20,
        }
    }
}

/// Detect and crop dark/noisy stacking edges.
///
/// Returns the cropped image and the pixels removed per edge. The input is
/// returned unchanged (with zero bounds) when no significant edges are
/// found or when cropping would remove more than half of a dimension.
pub fn autocrop(image: &Image, params: &AutocropParams) -> (Image, CropBounds) {
    let bounds = detect_edges(image, params);
    if bounds == (0, 0, 0, 0) {
        (image.clone(), bounds)
    } else {
        (image.crop(bounds), bounds)
    }
}

/// Detect dark stacking edges without materializing the crop.
///
/// Examines row and column medians (of the channel-mean luminance for RGB)
/// and reports contiguous runs from each edge that fall below
/// `threshold * interior_median`.
pub fn detect_edges(image: &Image, params: &AutocropParams) -> CropBounds {
    let w = image.width();
    let h = image.height();
    let mono = image.luminance();

    let row_medians: Vec<f64> = (0..h)
        .into_par_iter()
        .map(|y| stats::median_of(&mono[y * w..(y + 1) * w]))
        .collect();
    let col_medians: Vec<f64> = (0..w)
        .into_par_iter()
        .map(|x| {
            let mut col: Vec<f64> = (0..h).map(|y| mono[y * w + x]).collect();
            stats::median_in_place(&mut col)
        })
        .collect();

    // Interior reference: median over the central 60% of rows and columns
    let interior_rows = &row_medians[(h as f64 * 0.2) as usize..(h as f64 * 0.8) as usize];
    let interior_cols = &col_medians[(w as f64 * 0.2) as usize..(w as f64 * 0.8) as usize];
    if interior_rows.is_empty() || interior_cols.is_empty() {
        return (0, 0, 0, 0);
    }

    let mut interior: Vec<f64> = interior_rows.to_vec();
    interior.extend_from_slice(interior_cols);
    let interior_median = stats::median_in_place(&mut interior);
    if interior_median <= 0.0 {
        return (0, 0, 0, 0);
    }

    let dark_threshold = interior_median * params.threshold;
    let min_rows = std::cmp::max(1, (h as f64 * params.min_crop_fraction) as usize);
    let min_cols = std::cmp::max(1, (w as f64 * params.min_crop_fraction) as usize);
    let max_rows = (h as f64 * params.max_crop_fraction) as usize;
    let max_cols = (w as f64 * params.max_crop_fraction) as usize;

    let top = find_edge(row_medians.iter(), dark_threshold, min_rows, max_rows);
    let bottom = find_edge(row_medians.iter().rev(), dark_threshold, min_rows, max_rows);
    let left = find_edge(col_medians.iter(), dark_threshold, min_cols, max_cols);
    let right = find_edge(col_medians.iter().rev(), dark_threshold, min_cols, max_cols);

    if top + bottom + left + right == 0 {
        return (0, 0, 0, 0);
    }

    // Safety: never crop away more than half of either dimension
    if ((h - top - bottom) as f64) < h as f64 * 0.5 || ((w - left - right) as f64) < w as f64 * 0.5
    {
        return (0, 0, 0, 0);
    }

    (top, bottom, left, right)
}

/// Length of the contiguous dark run at the start of `medians`, or 0 if it
/// is shorter than `min_px`.
fn find_edge<'a>(
    medians: impl Iterator<Item = &'a f64>,
    dark_threshold: f64,
    min_px: usize,
    max_px: usize,
) -> usize {
    let mut count = 0;
    for (i, &v) in medians.take(max_px).enumerate() {
        if v < dark_threshold {
            count = i + 1;
        } else {
            break;
        }
    }
    if count >= min_px {
        count
    } else {
        0
    }
}
