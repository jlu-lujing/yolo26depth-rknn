//! Command-line interface definition.

use clap::{Parser, ValueEnum};

/// Colormap normalization mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ColorMode {
    /// 1/d with 2%-98% percentile clipping (best contrast)
    Disparity,
    /// Linear mapping over the valid depth range
    Metric,
}

/// YOLO26 Depth Estimation on RK3588 NPU
#[derive(Parser, Debug)]
#[command(name = "yolo26depth-rknn", version, about, long_about = None)]
pub struct Args {
    /// Path to the .rknn model file
    #[arg(short, long)]
    pub model: String,

    /// Path to the input image
    #[arg(short, long)]
    pub image: String,

    /// Path to save the output depth colormap (PNG)
    #[arg(short = 'o', long)]
    pub output: String,

    /// Colormap mode
    #[arg(long, value_enum, default_value_t = ColorMode::Disparity)]
    pub mode: ColorMode,

    /// Number of benchmark iterations (0 = single inference)
    #[arg(long, default_value = "0")]
    pub benchmark: u32,

    /// Save depth map as raw binary (float32, HxW)
    #[arg(long)]
    pub save_depth: Option<String>,

    /// Save colored point cloud as PLY file
    #[arg(long)]
    pub save_ply: Option<String>,

    /// Point cloud downsample factor (1=full, 2=half, 4=quarter)
    #[arg(long, default_value = "2")]
    pub pc_downsample: usize,

    /// NPU core selection (RK3588 has 3 cores)
    #[arg(long, value_enum, default_value_t = crate::rknn::NpuCore::Auto)]
    pub core: crate::rknn::NpuCore,
}
