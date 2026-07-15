use std::path::PathBuf;

/// Errors from FITS reading/writing and image output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("FITS file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("no image data found in FITS file: {0}")]
    NoImageData(PathBuf),

    #[error("FITS error: {0}")]
    Fits(String),

    #[error("unsupported FITS layout: {0}")]
    UnsupportedLayout(String),

    #[error("image encoding error: {0}")]
    Image(#[from] image::ImageError),
}
