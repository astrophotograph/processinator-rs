//! FITS file reading/writing and displayable image output.
//!
//! The reader is brought over from astra's `stretch/pipeline.rs`
//! (`read_fits_pixels`), extended to scan all HDUs for image data the way
//! the Python processinator does.

use std::path::Path;

use fitrs::{Fits, FitsData, Hdu};
use image::{DynamicImage, GrayImage, RgbImage};

use crate::error::Error;
use crate::image::Image;
use crate::pipeline::{self, PipelineConfig};

/// Read a FITS file as a mono or RGB [`Image`].
///
/// Handles the common layouts: `(H, W)` grayscale and channel-first RGB
/// (`NAXIS3 = 3`, the layout most astro software produces).
pub fn read_fits(path: impl AsRef<Path>) -> Result<Image, Error> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(Error::FileNotFound(path.to_path_buf()));
    }

    let fits = Fits::open(path).map_err(|e| Error::Fits(e.to_string()))?;

    for hdu in fits.iter() {
        let (shape, pixels) = match hdu.read_data() {
            FitsData::FloatingPoint32(arr) => {
                let pixels = arr.data.iter().map(|&x| x as f64).collect();
                (arr.shape.clone(), pixels)
            }
            FitsData::FloatingPoint64(arr) => (arr.shape.clone(), arr.data.clone()),
            FitsData::IntegersI32(arr) => {
                let pixels = arr.data.iter().map(|x| x.unwrap_or(0) as f64).collect();
                (arr.shape.clone(), pixels)
            }
            FitsData::IntegersU32(arr) => {
                let pixels = arr.data.iter().map(|x| x.unwrap_or(0) as f64).collect();
                (arr.shape.clone(), pixels)
            }
            FitsData::Characters(_) => continue,
        };

        if shape.len() < 2 || shape.iter().product::<usize>() == 0 {
            continue;
        }

        return image_from_planar(&shape, pixels);
    }

    Err(Error::NoImageData(path.to_path_buf()))
}

/// Build an [`Image`] from FITS-ordered data (`shape` is
/// `[NAXIS1, NAXIS2, ...]` = `[width, height, channels?]`, planes stored
/// sequentially, rows fastest).
fn image_from_planar(shape: &[usize], mut pixels: Vec<f64>) -> Result<Image, Error> {
    let width = shape[0];
    let height = shape[1];
    let plane = width * height;

    if pixels.len() < plane {
        return Err(Error::Fits(format!(
            "truncated FITS data: expected at least {} samples, got {}",
            plane,
            pixels.len()
        )));
    }

    if pixels.len() >= plane * 3 {
        let b = pixels.split_off(plane * 2);
        let g = pixels.split_off(plane);
        let mut r = pixels;
        r.truncate(plane);
        let mut b = b;
        b.truncate(plane);
        Ok(Image::new_rgb(width, height, [r, g, b]))
    } else {
        pixels.truncate(plane);
        Ok(Image::new_mono(width, height, pixels))
    }
}

/// Write image data to a FITS file (32-bit float, RGB stored
/// channels-first — the layout most astro software produces and
/// [`read_fits`] expects). Overwrites any existing file.
pub fn write_fits(image: &Image, path: impl AsRef<Path>) -> Result<(), Error> {
    let path = path.as_ref();
    let mut data: Vec<f32> = Vec::with_capacity(image.pixels_per_channel() * image.num_channels());
    for ch in image.channels() {
        data.extend(ch.iter().map(|&v| v as f32));
    }

    let shape: Vec<usize> = if image.is_color() {
        vec![image.width(), image.height(), 3]
    } else {
        vec![image.width(), image.height()]
    };

    Fits::create(path, Hdu::new(&shape, data)).map_err(|e| Error::Fits(e.to_string()))?;
    Ok(())
}

/// Read a FITS file, run the processing pipeline, and produce a
/// displayable 8-bit image.
///
/// The Python `fits_to_image` defaults (stretch only, no gradient removal
/// or denoising) correspond to
/// `PipelineConfig { gradient_removal: false, ..Default::default() }`.
///
/// When `output_path` is given the image is also saved there; the format
/// follows the file extension, with JPEG encoded at quality 95.
pub fn fits_to_image(
    fits_path: impl AsRef<Path>,
    output_path: Option<&Path>,
    config: &PipelineConfig,
) -> Result<DynamicImage, Error> {
    let data = read_fits(fits_path)?;
    let stretched = pipeline::process(&data, config);
    let img = to_dynamic_image(&stretched);

    if let Some(out) = output_path {
        save_image(&img, out)?;
    }

    Ok(img)
}

/// Convert stretched [0, 1] data to an 8-bit `image` crate image
/// (grayscale for mono input, RGB otherwise).
pub fn to_dynamic_image(image: &Image) -> DynamicImage {
    let w = image.width() as u32;
    let h = image.height() as u32;
    let to_u8 = |v: f64| (v * 255.0).clamp(0.0, 255.0) as u8;

    if image.is_color() {
        let n = image.pixels_per_channel();
        let mut rgb = Vec::with_capacity(n * 3);
        let (r, g, b) = (image.channel(0), image.channel(1), image.channel(2));
        for i in 0..n {
            rgb.push(to_u8(r[i]));
            rgb.push(to_u8(g[i]));
            rgb.push(to_u8(b[i]));
        }
        DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, rgb).expect("buffer sized to w*h*3"))
    } else {
        let gray: Vec<u8> = image.channel(0).iter().map(|&v| to_u8(v)).collect();
        DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, gray).expect("buffer sized to w*h"))
    }
}

/// Save with JPEG quality 95 for .jpg/.jpeg, default encoding otherwise.
fn save_image(img: &DynamicImage, path: &Path) -> Result<(), Error> {
    let is_jpeg = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        .unwrap_or(false);

    if is_jpeg {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, 95);
        // JPEG has no grayscale-with-alpha pitfalls here; both Luma8 and
        // Rgb8 encode directly
        img.write_with_encoder(encoder)?;
    } else {
        img.save(path)?;
    }
    Ok(())
}
