//! Enhancement operations ported from astra's Python `image_process`:
//! mean-anchored contrast, corner-sampled color calibration, and star
//! reduction. These fill the capability gaps so the astra daemon can run
//! the full desktop processing flow natively (no Python).
//!
//! Ports are semantic, not bit-exact: neighborhood filters use
//! clamp-to-edge boundaries where scipy defaults to reflection — an
//! invisible difference for the masks and blurs involved.

use rayon::prelude::*;

use crate::image::Image;
use crate::stats;

/// Mean-anchored contrast boost: `v' = mean + (v - mean) * strength`,
/// clipped to [0, 1]. The mean is global across all channels (matching
/// `np.mean` over the full array). `strength <= 1.0` is a no-op.
pub fn contrast(image: &mut Image, strength: f64) {
    if strength <= 1.0 {
        return;
    }
    let total = (image.pixels_per_channel() * image.num_channels()) as f64;
    if total == 0.0 {
        return;
    }
    let sum: f64 = image
        .channels()
        .iter()
        .map(|ch| ch.iter().sum::<f64>())
        .sum();
    let mean = sum / total;
    image.channels_mut().par_iter_mut().for_each(|ch| {
        for v in ch.iter_mut() {
            *v = (mean + (*v - mean) * strength).clamp(0.0, 1.0);
        }
    });
}

/// Background color neutralization for RGB images: sample the four frame
/// corners, take each channel's median, and rescale channels so their
/// backgrounds meet at the mean level. Mono images are returned untouched.
pub fn color_calibrate(image: &mut Image) {
    if !image.is_color() {
        return;
    }
    let (w, h) = (image.width(), image.height());
    let corner = (h.min(w) / 20).max(10).min(w).min(h);
    if corner == 0 {
        return;
    }

    let medians: Vec<f64> = (0..3)
        .map(|c| {
            let ch = image.channel(c);
            let mut samples = Vec::with_capacity(corner * corner * 4);
            for (y0, y1, x0, x1) in [
                (0, corner, 0, corner),
                (0, corner, w - corner, w),
                (h - corner, h, 0, corner),
                (h - corner, h, w - corner, w),
            ] {
                for y in y0..y1 {
                    for x in x0..x1 {
                        samples.push(ch[y * w + x]);
                    }
                }
            }
            stats::median_in_place(&mut samples)
        })
        .collect();

    let target = medians.iter().sum::<f64>() / 3.0;
    if target <= 0.0 {
        return;
    }
    for (c, &m) in medians.iter().enumerate() {
        if m > 0.0 {
            let scale = target / m;
            for v in image.channel_mut(c) {
                *v = (*v * scale).clamp(0.0, 1.0);
            }
        }
    }
}

/// Dim bright stars to emphasize nebulosity: detect luminance peaks above
/// `threshold` (5×5 local maxima), grow them into star regions (3 rounds of
/// 4-connected dilation), build a reduction map (stars → 0.7), smooth it
/// (gaussian σ=2), and multiply it into every channel.
pub fn reduce_stars(image: &mut Image, threshold: f64) {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return;
    }
    let lum = image.luminance();

    let local_max = maximum_filter(&lum, w, h, 2);
    let mut mask: Vec<bool> = lum
        .iter()
        .zip(&local_max)
        .map(|(&v, &m)| v == m && v > threshold)
        .collect();
    for _ in 0..3 {
        mask = dilate4(&mask, w, h);
    }

    let reduction: Vec<f64> = mask.iter().map(|&m| if m { 0.7 } else { 1.0 }).collect();
    let reduction = gaussian_blur(&reduction, w, h, 2.0);

    image.channels_mut().par_iter_mut().for_each(|ch| {
        for (v, r) in ch.iter_mut().zip(&reduction) {
            *v = (*v * r).clamp(0.0, 1.0);
        }
    });
}

/// Square (2r+1)×(2r+1) maximum filter, clamp-to-edge. Separable: rows
/// then columns.
fn maximum_filter(data: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut rows = vec![f64::NEG_INFINITY; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let mut m = f64::NEG_INFINITY;
            for xi in x0..x1 {
                m = m.max(data[y * w + xi]);
            }
            rows[y * w + x] = m;
        }
    }
    let mut out = vec![f64::NEG_INFINITY; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for x in 0..w {
            let mut m = f64::NEG_INFINITY;
            for yi in y0..y1 {
                m = m.max(rows[yi * w + x]);
            }
            out[y * w + x] = m;
        }
    }
    out
}

/// One round of 4-connected binary dilation.
fn dilate4(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                continue;
            }
            let hit = (x > 0 && mask[y * w + x - 1])
                || (x + 1 < w && mask[y * w + x + 1])
                || (y > 0 && mask[(y - 1) * w + x])
                || (y + 1 < h && mask[(y + 1) * w + x]);
            if hit {
                out[y * w + x] = true;
            }
        }
    }
    out
}

/// Separable gaussian blur with clamp-to-edge boundaries.
pub(crate) fn gaussian_blur(data: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }
    let r = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = (0..=2 * r)
        .map(|i| {
            let d = i as f64 - r as f64;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let norm: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|k| k / norm).collect();

    let mut rows = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let xi = (x + i).saturating_sub(r).min(w - 1);
                acc += data[y * w + xi] * k;
            }
            rows[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let yi = (y + i).saturating_sub(r).min(h - 1);
                acc += rows[yi * w + x] * k;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_mono(w: usize, h: usize) -> Image {
        let data: Vec<f64> = (0..w * h).map(|i| i as f64 / (w * h) as f64).collect();
        Image::new_mono(w, h, data)
    }

    #[test]
    fn contrast_widens_spread_and_is_noop_at_one() {
        let mut img = gradient_mono(16, 16);
        let before = img.clone();
        contrast(&mut img, 1.0);
        assert_eq!(img.channel(0), before.channel(0));

        contrast(&mut img, 1.5);
        let spread = |im: &Image| {
            let ch = im.channel(0);
            let mean = ch.iter().sum::<f64>() / ch.len() as f64;
            ch.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ch.len() as f64
        };
        assert!(spread(&img) > spread(&before));
        assert!(img.channel(0).iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn color_calibrate_neutralizes_corner_cast() {
        // Flat background with a strong red cast
        let n = 64 * 64;
        let mut img = Image::new_rgb(64, 64, [vec![0.4; n], vec![0.2; n], vec![0.1; n]]);
        color_calibrate(&mut img);
        let bg = |c: usize| img.get(2, 2, c);
        assert!((bg(0) - bg(1)).abs() < 1e-9);
        assert!((bg(1) - bg(2)).abs() < 1e-9);
    }

    #[test]
    fn reduce_stars_dims_peaks_not_background() {
        let n = 64 * 64;
        let mut data = vec![0.1; n];
        data[32 * 64 + 32] = 1.0; // one bright star
        let mut img = Image::new_mono(64, 64, data);
        reduce_stars(&mut img, 0.8);
        assert!(img.get(32, 32, 0) < 1.0, "star should be dimmed");
        // Far corner untouched (blurred mask never reaches it)
        assert!((img.get(2, 2, 0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn gaussian_blur_preserves_constant_field() {
        let blurred = gaussian_blur(&vec![0.5; 100], 10, 10, 2.0);
        assert!(blurred.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }
}
