//! Enhancement operations ported from astra's Python `image_process`:
//! mean-anchored contrast, corner-sampled color calibration, and star
//! reduction. These fill the capability gaps so the astra daemon can run
//! the full desktop processing flow natively (no Python).
//!
//! Ports are semantic, not bit-exact: neighborhood filters use
//! clamp-to-edge boundaries where scipy defaults to reflection — an
//! invisible difference for the masks and blurs involved.

use rayon::prelude::*;

use crate::autocrop::CropBounds;
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

/// Background color neutralization for RGB images: bring every channel's
/// sky to the same level by **offset**, never by gain.
///
/// The sky is each channel's median over the autocrop interior (`bounds`,
/// as [`crate::autocrop::detect_edges`] reports them): the frame corners
/// of a stacked image are the dark stacking edges, which is why sampling
/// them produced arbitrary per-channel scalings. The offsets are add-only
/// (channels are lifted to the highest background, nothing is pushed
/// below zero), so signal above the sky keeps its channel differences
/// exactly — the same neutralization the linked MTF stretch performs
/// through its shadow offsets. A multiplicative balance is deliberately
/// not applied here: after per-channel normalization the data carries no
/// white reference a gain could be derived from, and scaling channels to
/// equalize a background that gradient removal already pulled to the
/// noise floor only amplified noise differences into a color cast.
/// Mono images are returned untouched.
pub fn color_calibrate(image: &mut Image, bounds: CropBounds) {
    if !image.is_color() {
        return;
    }
    let (top, bottom, left, right) = bounds;
    if top + bottom >= image.height() || left + right >= image.width() {
        return;
    }
    let backgrounds: Vec<f64> = (0..3)
        .map(|c| {
            let mut interior = image.channel_interior(c, bounds);
            stats::median_in_place(&mut interior)
        })
        .collect();
    if backgrounds.iter().any(|v| !v.is_finite()) {
        return;
    }
    let target = backgrounds
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for (c, &bg) in backgrounds.iter().enumerate() {
        let offset = target - bg;
        if offset > 0.0 {
            for v in image.channel_mut(c) {
                *v = (*v + offset).min(1.0);
            }
        }
    }
}

/// Dim bright stars to emphasize nebulosity: detect luminance peaks above
/// `threshold` (5×5 local maxima) that also stand at least
/// [`STAR_PROMINENCE`] above their surroundings, grow them into star
/// regions (3 rounds of 4-connected dilation), build a reduction map
/// (stars → 0.7), smooth it (gaussian σ=2), and multiply it into every
/// channel.
///
/// The prominence test is what keeps a bright nebula core intact: inside
/// a near-saturated core every noise ripple is a local maximum above the
/// threshold, and dimming those speckled the core with gray dots. A star
/// rises well above the 13×13 neighbourhood minimum; a ripple on a
/// plateau does not.
pub fn reduce_stars(image: &mut Image, threshold: f64) {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return;
    }
    let lum = image.luminance();

    let local_max = extremum_filter(&lum, w, h, 2, f64::max, f64::NEG_INFINITY);
    let local_min = extremum_filter(&lum, w, h, 6, f64::min, f64::INFINITY);
    let mut mask: Vec<bool> = (0..w * h)
        .map(|i| {
            let v = lum[i];
            v == local_max[i] && v > threshold && v - local_min[i] > STAR_PROMINENCE
        })
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

/// How far a peak must rise above its 13×13 neighbourhood minimum to count
/// as a star rather than a ripple on bright nebulosity (display units).
const STAR_PROMINENCE: f64 = 0.25;

/// Square (2r+1)×(2r+1) running extremum (`pick` = `f64::max` or
/// `f64::min`, `identity` its neutral element), clamp-to-edge. Separable:
/// rows then columns.
fn extremum_filter(
    data: &[f64],
    w: usize,
    h: usize,
    r: usize,
    pick: fn(f64, f64) -> f64,
    identity: f64,
) -> Vec<f64> {
    let mut rows = vec![identity; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let mut m = identity;
            for xi in x0..x1 {
                m = pick(m, data[y * w + xi]);
            }
            rows[y * w + x] = m;
        }
    }
    let mut out = vec![identity; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for x in 0..w {
            let mut m = identity;
            for yi in y0..y1 {
                m = pick(m, rows[yi * w + x]);
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
    fn color_calibrate_neutralizes_cast_by_offset() {
        // Flat background with a strong red cast and one white star
        let n = 64 * 64;
        let mut img = Image::new_rgb(64, 64, [vec![0.4; n], vec![0.2; n], vec![0.1; n]]);
        for c in 0..3 {
            img.set(32, 32, c, img.get(32, 32, c) + 0.3);
        }
        color_calibrate(&mut img, (0, 0, 0, 0));
        let bg = |c: usize| img.get(2, 2, c);
        assert!((bg(0) - bg(1)).abs() < 1e-9);
        assert!((bg(1) - bg(2)).abs() < 1e-9);
        // Add-only: channels rise to the red level, nothing is scaled
        assert!((bg(0) - 0.4).abs() < 1e-9);
        for c in 0..3 {
            assert!((img.get(32, 32, c) - bg(c) - 0.3).abs() < 1e-9, "star delta channel {c}");
        }
    }

    #[test]
    fn color_calibrate_samples_the_interior_not_the_stacking_edges() {
        // Neutral sky, but the top 8 rows and left 10 columns are dark
        // stacking edges — and the edge is darker in blue than in red,
        // exactly the kind of corner that used to drive the gains.
        let (w, h) = (64usize, 64usize);
        let mut img = Image::new_rgb(w, h, [vec![0.20; w * h], vec![0.20; w * h], vec![0.20; w * h]]);
        for y in 0..h {
            for x in 0..w {
                if y < 8 || x < 10 {
                    img.set(x, y, 0, 0.010);
                    img.set(x, y, 1, 0.004);
                    img.set(x, y, 2, 0.001);
                }
            }
        }
        let before = img.clone();
        color_calibrate(&mut img, (8, 0, 10, 0));
        // Interior was already neutral: nothing should move
        for c in 0..3 {
            assert_eq!(img.channel(c), before.channel(c), "channel {c} changed");
        }
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
    fn reduce_stars_leaves_a_bright_nebula_core_alone() {
        // A saturated core (0.9 plateau with ±0.02 ripples) next to sky with
        // one real star: only the star is dimmed
        let (w, h) = (96usize, 64usize);
        let mut data = vec![0.1; w * h];
        for y in 8..56 {
            for x in 8..48 {
                data[y * w + x] = 0.9 + if (x + y) % 2 == 0 { 0.02 } else { -0.02 };
            }
        }
        data[32 * w + 80] = 1.0;
        let mut img = Image::new_mono(w, h, data.clone());
        reduce_stars(&mut img, 0.8);
        let out = img.channel(0);
        assert!(out[32 * w + 80] < 1.0, "isolated star dimmed");
        // The plateau's rim does look like a star edge (prominence against
        // the sky), so check well inside it, beyond dilation + blur reach
        for y in 24..42 {
            for x in 24..34 {
                // Within 1%: the rim detections' blur tail reaches this far
                assert!(
                    (out[y * w + x] - data[y * w + x]).abs() < 0.01,
                    "core pixel ({x},{y}) changed: {} -> {}",
                    data[y * w + x],
                    out[y * w + x]
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_preserves_constant_field() {
        let blurred = gaussian_blur(&vec![0.5; 100], 10, 10, 2.0);
        assert!(blurred.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }
}
