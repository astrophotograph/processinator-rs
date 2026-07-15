//! Planar image container shared by all processing stages.

use crate::autocrop::CropBounds;

/// A mono or RGB image stored as planar channels.
///
/// Each channel is a row-major `width * height` plane of `f64` samples.
/// This mirrors the numpy layouts the Python processinator accepts —
/// `(H, W)` for mono and `(H, W, 3)` for RGB — stored here the way astro
/// FITS files lay channels out on disk (plane after plane).
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    width: usize,
    height: usize,
    channels: Vec<Vec<f64>>,
}

impl Image {
    /// Single-channel image from a row-major plane.
    pub fn new_mono(width: usize, height: usize, data: Vec<f64>) -> Self {
        Self::from_channels(width, height, vec![data])
    }

    /// Three-channel image from row-major R, G, B planes.
    pub fn new_rgb(width: usize, height: usize, channels: [Vec<f64>; 3]) -> Self {
        Self::from_channels(width, height, channels.into())
    }

    /// Build from 1 (mono) or 3 (RGB) row-major planes.
    pub fn from_channels(width: usize, height: usize, channels: Vec<Vec<f64>>) -> Self {
        assert!(
            channels.len() == 1 || channels.len() == 3,
            "expected 1 or 3 channels, got {}",
            channels.len()
        );
        for ch in &channels {
            assert_eq!(
                ch.len(),
                width * height,
                "channel length must equal width * height"
            );
        }
        Self {
            width,
            height,
            channels,
        }
    }

    /// All-zero image with the same dimensions and channel count as `self`.
    pub fn zeros_like(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            channels: vec![vec![0.0; self.width * self.height]; self.channels.len()],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    pub fn is_color(&self) -> bool {
        self.channels.len() == 3
    }

    /// Samples per channel (`width * height`).
    pub fn pixels_per_channel(&self) -> usize {
        self.width * self.height
    }

    pub fn channel(&self, i: usize) -> &[f64] {
        &self.channels[i]
    }

    pub fn channel_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.channels[i]
    }

    pub fn channels(&self) -> &[Vec<f64>] {
        &self.channels
    }

    pub fn channels_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.channels
    }

    pub fn into_channels(self) -> Vec<Vec<f64>> {
        self.channels
    }

    pub fn get(&self, x: usize, y: usize, c: usize) -> f64 {
        self.channels[c][y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: usize, value: f64) {
        self.channels[c][y * self.width + x] = value;
    }

    /// Channel-mean luminance plane (what edge detection runs on for RGB).
    pub fn luminance(&self) -> Vec<f64> {
        if self.channels.len() == 1 {
            return self.channels[0].clone();
        }
        let n = self.pixels_per_channel();
        let scale = 1.0 / self.channels.len() as f64;
        (0..n)
            .map(|i| self.channels.iter().map(|ch| ch[i]).sum::<f64>() * scale)
            .collect()
    }

    /// Copy of the interior after removing `(top, bottom, left, right)`
    /// pixels from the edges.
    pub fn crop(&self, bounds: CropBounds) -> Self {
        let (top, bottom, left, right) = bounds;
        let new_w = self.width - left - right;
        let new_h = self.height - top - bottom;
        let channels = self
            .channels
            .iter()
            .map(|ch| {
                let mut out = Vec::with_capacity(new_w * new_h);
                for y in top..self.height - bottom {
                    let row = &ch[y * self.width..(y + 1) * self.width];
                    out.extend_from_slice(&row[left..self.width - right]);
                }
                out
            })
            .collect();
        Self {
            width: new_w,
            height: new_h,
            channels,
        }
    }

    /// Interior samples of one channel as a flat buffer (for statistics).
    pub(crate) fn channel_interior(&self, c: usize, bounds: CropBounds) -> Vec<f64> {
        let (top, bottom, left, right) = bounds;
        if bounds == (0, 0, 0, 0) {
            return self.channels[c].clone();
        }
        let ch = &self.channels[c];
        let mut out =
            Vec::with_capacity((self.height - top - bottom) * (self.width - left - right));
        for y in top..self.height - bottom {
            let row = &ch[y * self.width..(y + 1) * self.width];
            out.extend_from_slice(&row[left..self.width - right]);
        }
        out
    }
}
