//! Custom error type for the application.

use std::fmt;

/// All errors that can occur during depth estimation.
#[derive(Debug)]
pub enum Error {
    /// Failed to load or initialize the RKNN model.
    Rknn(String),
    /// File I/O error.
    Io(std::io::Error),
    /// Image error (decode, encode, etc.).
    Image(String),
    /// Invalid argument or configuration.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Rknn(msg) => write!(f, "RKNN error: {msg}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Image(msg) => write!(f, "Image error: {msg}"),
            Error::Invalid(msg) => write!(f, "Invalid: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e.to_string())
    }
}

/// Shorthand Result type.
pub type Result<T> = std::result::Result<T, Error>;
