//! Background gradient removal for astrophotography images.
//!
//! Models the sky background as a low-order 2D polynomial surface per
//! channel (sampled on a grid with sigma-clipping to reject stars),
//! evaluates it at reduced resolution, upsamples, and subtracts. This
//! removes light pollution gradients and vignetting while preserving
//! astronomical signal. Brought over from astra's `stretch/gradient.rs`.

use rayon::prelude::*;

use crate::image::Image;
use crate::stats;

/// Tuning for the background surface fit.
#[derive(Debug, Clone)]
pub struct GradientParams {
    /// Polynomial order: 1 = linear tilt, 2 = quadratic (vignetting), 3 = cubic.
    pub order: usize,
    /// Sigma threshold for rejecting bright samples (stars/objects) from
    /// the background model.
    pub sigma_clip: f64,
    /// Grid divisions for background sampling. Higher = more accurate but
    /// slower.
    pub sample_grid: usize,
}

impl Default for GradientParams {
    fn default() -> Self {
        Self {
            order: 2,
            sigma_clip: 2.5,
            sample_grid: 32,
        }
    }
}

/// Remove the background gradient from each channel.
///
/// Expects data in the [0, 1] range; the output is clipped back to [0, 1]
/// with the darkest background region shifted near zero.
pub fn remove_gradient(image: &Image, params: &GradientParams) -> Image {
    let w = image.width();
    let h = image.height();
    let channels: Vec<Vec<f64>> = image
        .channels()
        .par_iter()
        .map(|ch| remove_gradient_channel(ch, w, h, params))
        .collect();
    Image::from_channels(w, h, channels)
}

/// Remove the gradient from a single row-major channel.
fn remove_gradient_channel(
    data: &[f64],
    width: usize,
    height: usize,
    params: &GradientParams,
) -> Vec<f64> {
    let sample_grid = params.sample_grid.max(2);
    let order = params.order;

    // Sample background on a grid using patch medians for robustness
    let patch_h = std::cmp::max(1, height / (sample_grid * 2));
    let patch_w = std::cmp::max(1, width / (sample_grid * 2));

    let mut sample_y = Vec::with_capacity(sample_grid * sample_grid);
    let mut sample_x = Vec::with_capacity(sample_grid * sample_grid);
    let mut sample_v = Vec::with_capacity(sample_grid * sample_grid);

    for gy in 0..sample_grid {
        let y = gy * (height - 1) / (sample_grid - 1);
        for gx in 0..sample_grid {
            let x = gx * (width - 1) / (sample_grid - 1);

            let y0 = y.saturating_sub(patch_h);
            let y1 = std::cmp::min(height, y + patch_h + 1);
            let x0 = x.saturating_sub(patch_w);
            let x1 = std::cmp::min(width, x + patch_w + 1);

            let mut patch = Vec::with_capacity((y1 - y0) * (x1 - x0));
            for py in y0..y1 {
                patch.extend_from_slice(&data[py * width + x0..py * width + x1]);
            }

            sample_y.push(y as f64);
            sample_x.push(x as f64);
            sample_v.push(stats::median_in_place(&mut patch));
        }
    }

    // Sigma-clip to reject stars and bright objects
    for _ in 0..3 {
        let med = stats::median_of(&sample_v);
        let deviations: Vec<f64> = sample_v.iter().map(|v| (v - med).abs()).collect();
        let mad = stats::median_of(&deviations);
        let std_est = mad * 1.4826;
        if std_est < 1e-10 {
            break;
        }
        let limit = params.sigma_clip * std_est;
        let mask: Vec<bool> = sample_v.iter().map(|v| (v - med).abs() < limit).collect();
        if mask.iter().filter(|&&m| m).count() < 6 {
            break;
        }
        let filter = |vals: &[f64]| -> Vec<f64> {
            mask.iter()
                .zip(vals)
                .filter(|(&m, _)| m)
                .map(|(_, &v)| v)
                .collect()
        };
        sample_y = filter(&sample_y);
        sample_x = filter(&sample_x);
        sample_v = filter(&sample_v);
    }

    // Normalize coordinates to [-1, 1] for numerical stability
    let h_max = std::cmp::max(1, height - 1) as f64;
    let w_max = std::cmp::max(1, width - 1) as f64;
    let yn: Vec<f64> = sample_y.iter().map(|&y| y / h_max * 2.0 - 1.0).collect();
    let xn: Vec<f64> = sample_x.iter().map(|&x| x / w_max * 2.0 - 1.0).collect();

    // Build design matrix and solve least squares
    let terms = poly_terms_2d(&xn, &yn, order);
    let n_terms = terms.len();

    let coeffs = match lstsq(&terms, &sample_v, n_terms) {
        Some(c) => c,
        None => return data.to_vec(),
    };

    // Evaluate the model at reduced resolution — a low-order polynomial is
    // smooth, so there is no need to evaluate at every pixel
    let eval_size = std::cmp::min(256, std::cmp::min(width, height)).max(2);
    let mut small_model = vec![0.0f64; eval_size * eval_size];

    for ey in 0..eval_size {
        let yn_val = ey as f64 / (eval_size - 1) as f64 * 2.0 - 1.0;
        for ex in 0..eval_size {
            let xn_val = ex as f64 / (eval_size - 1) as f64 * 2.0 - 1.0;
            let mut val = 0.0;
            let mut ci = 0;
            for total in 0..=order {
                for xpow in (0..=total).rev() {
                    let ypow = total - xpow;
                    val += coeffs[ci] * xn_val.powi(xpow as i32) * yn_val.powi(ypow as i32);
                    ci += 1;
                }
            }
            small_model[ey * eval_size + ex] = val;
        }
    }

    // Bilinear upsample to full resolution and subtract (parallel by row)
    let es_f = (eval_size - 1) as f64;
    let mut result: Vec<f64> = (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            let sy = y as f64 / h_max * es_f;
            let sy0 = sy.floor() as usize;
            let sy1 = std::cmp::min(sy0 + 1, eval_size - 1);
            let fy = sy - sy0 as f64;
            let fy_inv = 1.0 - fy;
            let small = &small_model;

            (0..width).map(move |x| {
                let sx = x as f64 / w_max * es_f;
                let sx0 = sx.floor() as usize;
                let sx1 = std::cmp::min(sx0 + 1, eval_size - 1);
                let fx = sx - sx0 as f64;

                let model_val = small[sy0 * eval_size + sx0] * (1.0 - fx) * fy_inv
                    + small[sy0 * eval_size + sx1] * fx * fy_inv
                    + small[sy1 * eval_size + sx0] * (1.0 - fx) * fy
                    + small[sy1 * eval_size + sx1] * fx * fy;

                data[y * width + x] - model_val
            })
        })
        .collect();

    // Shift so the darkest background region (1st percentile) sits near zero
    let mut scratch = result.clone();
    let bg_level = stats::percentile_in_place(&mut scratch, 1.0);
    drop(scratch);

    result.par_iter_mut().for_each(|v| {
        *v = (*v - bg_level).clamp(0.0, 1.0);
    });

    result
}

/// Polynomial term columns up to `order`; for order 2:
/// `[1, x, y, x^2, xy, y^2]`, each evaluated at every sample.
fn poly_terms_2d(x: &[f64], y: &[f64], order: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut columns = Vec::new();
    for total in 0..=order {
        for xpow in (0..=total).rev() {
            let ypow = total - xpow;
            let col: Vec<f64> = (0..n)
                .map(|i| x[i].powi(xpow as i32) * y[i].powi(ypow as i32))
                .collect();
            columns.push(col);
        }
    }
    columns
}

/// Least squares via normal equations: `(A^T A) x = A^T b`.
fn lstsq(columns: &[Vec<f64>], b: &[f64], n_cols: usize) -> Option<Vec<f64>> {
    let mut ata = vec![0.0; n_cols * n_cols];
    let mut atb = vec![0.0; n_cols];

    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    for i in 0..n_cols {
        for j in 0..n_cols {
            ata[i * n_cols + j] = dot(&columns[i], &columns[j]);
        }
        atb[i] = dot(&columns[i], b);
    }

    solve_symmetric(&mut ata, &mut atb, n_cols)
}

/// Gaussian elimination with partial pivoting.
fn solve_symmetric(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col * n + k] * x[k];
        }
        x[col] = sum / a[col * n + col];
    }
    Some(x)
}
