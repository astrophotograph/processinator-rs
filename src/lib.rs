//! Processinator — astronomy image processing library.
//!
//! Converts linear FITS data into visually useful images using nonlinear
//! stretch algorithms, with optional background gradient removal, starlet
//! wavelet denoising, and dark stacking-edge detection. A Rust port of the
//! Python `processinator` package, seeded from the native pipeline in the
//! astra desktop app.
//!
//! ```no_run
//! use processinator::{fits_to_image, PipelineConfig};
//!
//! // High-level: FITS file → PNG, with gradient removal and denoising
//! let config = PipelineConfig {
//!     denoise: true,
//!     ..Default::default()
//! };
//! let image = fits_to_image(
//!     "my_image.fits",
//!     Some(std::path::Path::new("stretched.png")),
//!     &config,
//! )?;
//! # Ok::<(), processinator::Error>(())
//! ```
//!
//! Lower-level entry points: [`stretch`] for a plain stretch, [`process`]
//! for full pipeline control over in-memory data, and [`make_test_image`]
//! for synthetic frames with known ground truth.

pub mod autocrop;
pub mod denoise;
mod error;
pub mod fits_io;
pub mod gradient;
pub mod image;
pub mod pipeline;
mod stats;
pub mod stretch;
pub mod synthetic;

pub use self::autocrop::{autocrop, detect_edges, AutocropParams, CropBounds};
pub use self::denoise::{denoise, DenoiseParams, ThresholdMode};
pub use self::error::Error;
pub use self::fits_io::{fits_to_image, read_fits, to_dynamic_image, write_fits};
pub use self::gradient::{remove_gradient, GradientParams};
pub use self::image::Image;
pub use self::pipeline::{process, PipelineConfig};
pub use self::stretch::{remove_green, saturate, stretch, StretchAlgorithm, StretchOptions};
pub use self::synthetic::{make_test_image, Star, SyntheticImage, SyntheticParams};
