//! Depth post-processing: resize to original image size and colormap generation.

use image::RgbImage;
use log::info;

use crate::cli::ColorMode;
use crate::error::Result;

// ============================================================================
// Bilinear resize for f32 depth maps (f32 math matches OpenCV cv::INTER_LINEAR)
// ============================================================================

/// Resize a depth map from model output size to the original image size.
pub fn resize_depth(
    depth_raw: &[f32],
    model_w: usize,
    model_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<f32> {
    if model_w == target_w && model_h == target_h {
        return depth_raw.to_vec();
    }

    info!(
        "Resize depth: {}x{} -> {}x{} (bilinear)",
        model_w, model_h, target_w, target_h
    );

    let mut dst = vec![0.0f32; target_w * target_h];
    let sx = model_w as f32 / target_w as f32;
    let sy = model_h as f32 / target_h as f32;

    for y in 0..target_h {
        let sy_f = (y as f32 + 0.5) * sy - 0.5;
        let y0 = sy_f.floor() as usize;
        let y1 = (y0 + 1).min(model_h - 1);
        let fy = sy_f - y0 as f32;

        for x in 0..target_w {
            let sx_f = (x as f32 + 0.5) * sx - 0.5;
            let x0 = sx_f.floor() as usize;
            let x1 = (x0 + 1).min(model_w - 1);
            let fx = sx_f - x0 as f32;

            let v = depth_raw[y0 * model_w + x0] as f32 * (1.0 - fx) * (1.0 - fy)
                + depth_raw[y0 * model_w + x1] as f32 * fx * (1.0 - fy)
                + depth_raw[y1 * model_w + x0] as f32 * (1.0 - fx) * fy
                + depth_raw[y1 * model_w + x1] as f32 * fx * fy;
            dst[y * target_w + x] = v;
        }
    }
    dst
}

// ============================================================================
// Statistics helpers
// ============================================================================

/// Linear interpolation percentile from a sorted slice.
pub fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Print depth statistics to stderr via log.
pub fn print_stats(depth: &[f32]) {
    let valid: Vec<f32> = depth.iter().copied().filter(|&d| d > 0.0).collect();
    if valid.is_empty() {
        log::warn!("No valid depth values!");
        return;
    }
    let mut sorted = valid;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let p5 = percentile(&sorted, 5.0);
    let p95 = percentile(&sorted, 95.0);
    info!("Depth median: {:.2} m", median);
    info!("Depth 5%-95%: {:.1} ~ {:.1} m", p5, p95);
}

// ============================================================================
// JET colormap — exact OpenCV COLORMAP_JET LUT (RGB)
// ============================================================================

/// OpenCV COLORMAP_JET RGB lookup table (256 entries), extracted from
/// `cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)`.
/// BGR→RGB transposed for image::RgbImage.
/// This gives pixel-identical output to Python and C++ which use OpenCV directly.
#[rustfmt::skip]
static JET_LUT_RGB: [[u8; 3]; 256] = [
    [0,0,128],[0,0,132],[0,0,136],[0,0,140],[0,0,144],[0,0,148],[0,0,152],[0,0,156],
    [0,0,160],[0,0,164],[0,0,168],[0,0,172],[0,0,176],[0,0,180],[0,0,184],[0,0,188],
    [0,0,192],[0,0,196],[0,0,200],[0,0,204],[0,0,208],[0,0,212],[0,0,216],[0,0,220],
    [0,0,224],[0,0,228],[0,0,232],[0,0,236],[0,0,240],[0,0,244],[0,0,248],[0,0,252],
    [0,0,255],[0,4,255],[0,8,255],[0,12,255],[0,16,255],[0,20,255],[0,24,255],[0,28,255],
    [0,32,255],[0,36,255],[0,40,255],[0,44,255],[0,48,255],[0,52,255],[0,56,255],[0,60,255],
    [0,64,255],[0,68,255],[0,72,255],[0,76,255],[0,80,255],[0,84,255],[0,88,255],[0,92,255],
    [0,96,255],[0,100,255],[0,104,255],[0,108,255],[0,112,255],[0,116,255],[0,120,255],[0,124,255],
    [0,128,255],[0,132,255],[0,136,255],[0,140,255],[0,144,255],[0,148,255],[0,152,255],[0,156,255],
    [0,160,255],[0,164,255],[0,168,255],[0,172,255],[0,176,255],[0,180,255],[0,184,255],[0,188,255],
    [0,192,255],[0,196,255],[0,200,255],[0,204,255],[0,208,255],[0,212,255],[0,216,255],[0,220,255],
    [0,224,255],[0,228,255],[0,232,255],[0,236,255],[0,240,255],[0,244,255],[0,248,255],[0,252,255],
    [2,255,254],[6,255,250],[10,255,246],[14,255,242],[18,255,238],[22,255,234],[26,255,230],[30,255,226],
    [34,255,222],[38,255,218],[42,255,214],[46,255,210],[50,255,206],[54,255,202],[58,255,198],[62,255,194],
    [66,255,190],[70,255,186],[74,255,182],[78,255,178],[82,255,174],[86,255,170],[90,255,166],[94,255,162],
    [98,255,158],[102,255,154],[106,255,150],[110,255,146],[114,255,142],[118,255,138],[122,255,134],[126,255,130],
    [130,255,126],[134,255,122],[138,255,118],[142,255,114],[146,255,110],[150,255,106],[154,255,102],[158,255,98],
    [162,255,94],[166,255,90],[170,255,86],[174,255,82],[178,255,78],[182,255,74],[186,255,70],[190,255,66],
    [194,255,62],[198,255,58],[202,255,54],[206,255,50],[210,255,46],[214,255,42],[218,255,38],[222,255,34],
    [226,255,30],[230,255,26],[234,255,22],[238,255,18],[242,255,14],[246,255,10],[250,255,6],[254,255,1],
    [255,252,0],[255,248,0],[255,244,0],[255,240,0],[255,236,0],[255,232,0],[255,228,0],[255,224,0],
    [255,220,0],[255,216,0],[255,212,0],[255,208,0],[255,204,0],[255,200,0],[255,196,0],[255,192,0],
    [255,188,0],[255,184,0],[255,180,0],[255,176,0],[255,172,0],[255,168,0],[255,164,0],[255,160,0],
    [255,156,0],[255,152,0],[255,148,0],[255,144,0],[255,140,0],[255,136,0],[255,132,0],[255,128,0],
    [255,124,0],[255,120,0],[255,116,0],[255,112,0],[255,108,0],[255,104,0],[255,100,0],[255,96,0],
    [255,92,0],[255,88,0],[255,84,0],[255,80,0],[255,76,0],[255,72,0],[255,68,0],[255,64,0],
    [255,60,0],[255,56,0],[255,52,0],[255,48,0],[255,44,0],[255,40,0],[255,36,0],[255,32,0],
    [255,28,0],[255,24,0],[255,20,0],[255,16,0],[255,12,0],[255,8,0],[255,4,0],[255,0,0],
    [252,0,0],[248,0,0],[244,0,0],[240,0,0],[236,0,0],[232,0,0],[228,0,0],[224,0,0],
    [220,0,0],[216,0,0],[212,0,0],[208,0,0],[204,0,0],[200,0,0],[196,0,0],[192,0,0],
    [188,0,0],[184,0,0],[180,0,0],[176,0,0],[172,0,0],[168,0,0],[164,0,0],[160,0,0],
    [156,0,0],[152,0,0],[148,0,0],[144,0,0],[140,0,0],[136,0,0],[132,0,0],[128,0,0],
];

/// Look up the OpenCV COLORMAP_JET color for a normalized value t in [0, 1].
/// Returns [R, G, B] (for image::RgbImage).
/// Uses truncation (not rounding) to match Python's `astype(np.uint8)`.
fn jet_color(t: f32) -> [u8; 3] {
    let idx = (t * 255.0) as usize;
    let idx = idx.min(255);
    JET_LUT_RGB[idx]
}

// ============================================================================
// Colormap generation
// ============================================================================

/// Convert a depth map to a JET colormap image and save it.
pub fn save_colormap(
    depth: &[f32],
    width: usize,
    height: usize,
    output_path: &str,
    mode: ColorMode,
) -> Result<()> {
    let pixels: Vec<u8> = match mode {
        ColorMode::Disparity => disparity_to_pixels(depth),
        ColorMode::Metric => metric_to_pixels(depth),
    };

    let img = RgbImage::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| crate::error::Error::Invalid("failed to create image buffer".into()))?;
    img.save(output_path)?;
    info!("Colormap saved: {}", output_path);
    Ok(())
}

/// Disparity mode: 1/d with 2%-98% percentile clipping.
fn disparity_to_pixels(depth: &[f32]) -> Vec<u8> {
    let n = depth.len();
    let mut valid_disparity = Vec::new();
    let mut disparity = vec![0.0f32; n];

    for (i, &d) in depth.iter().enumerate() {
        if d > 0.0 {
            let disp = 1.0 / d;
            disparity[i] = disp;
            valid_disparity.push(disp);
        }
    }

    valid_disparity.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile(&valid_disparity, 2.0);
    let hi = percentile(&valid_disparity, 98.0);
    let range = (hi - lo).max(1e-6);

    depth
        .iter()
        .zip(disparity.iter())
        .flat_map(|(&d, &v)| {
            if d <= 0.0 {
                [0u8, 0u8, 0u8]
            } else {
                let t = ((v - lo) / range).clamp(0.0, 1.0);
                jet_color(t)
            }
        })
        .collect()
}

/// Metric mode: linear mapping over the valid (positive, finite) depth range.
fn metric_to_pixels(depth: &[f32]) -> Vec<u8> {
    let valid: Vec<f32> = depth.iter().filter(|&&d| d > 0.0 && d.is_finite()).copied().collect();
    if valid.is_empty() {
        return vec![0u8; depth.len() * 3];
    }
    let min_d = valid.iter().copied().fold(f32::INFINITY, f32::min);
    let max_d = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_d - min_d).max(1e-8);

    depth
        .iter()
        .flat_map(|&v| {
            if v > 0.0 && v.is_finite() {
                let t = ((v - min_d) / range).clamp(0.0, 1.0);
                jet_color(t)
            } else {
                [0u8, 0u8, 0u8]
            }
        })
        .collect()
}
