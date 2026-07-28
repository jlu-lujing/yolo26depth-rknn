//! YOLO26-Depth RKNN — Monocular depth estimation on RK3588 NPU.
//!
//! # Modules
//!
//! - [`cli`] — Command-line argument definitions
//! - [`error`] — Custom error types
//! - [`rknn`] — RKNN C bindings and model wrapper
//! - [`preprocess`] — Image loading and preprocessing
//! - [`postprocess`] — Depth resize, colormap, statistics
//! - [`pointcloud`] — Point cloud PLY generation

pub mod cli;
pub mod error;
pub mod postprocess;
pub mod pointcloud;
pub mod preprocess;
pub mod rknn;
