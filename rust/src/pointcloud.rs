//! Point cloud generation from depth map and color image.

use std::io::Write;

use log::info;

use crate::error::Result;

/// Default camera intrinsics (approximate 50-degree FOV).
fn default_intrinsics(w: usize, h: usize) -> (f32, f32, f32, f32) {
    let fx = h as f32;
    let fy = h as f32;
    let cx = (w as f32) / 2.0;
    let cy = (h as f32) / 2.0;
    (fx, fy, cx, cy)
}

/// Save a colored point cloud as binary PLY.
///
/// The depth map is back-projected using default camera intrinsics
/// (fx=fy=height, cx=w/2, cy=h/2) and filtered by `min_depth`.
pub fn save_pointcloud(
    depth: &[f32],
    img: &image::DynamicImage,
    w: usize,
    h: usize,
    path: &str,
    downsample: usize,
) -> Result<()> {
    let rgb = img.to_rgb8();
    let (fx, fy, cx, cy) = default_intrinsics(w, h);

    // Collect vertices as flat buffer (15 bytes each: 3xf32 + 3xu8)
    let mut buf = Vec::new();

    for y in (0..h).step_by(downsample) {
        for x in (0..w).step_by(downsample) {
            let d = depth[y * w + x];
            if d > 0.1 {
                let px = x as f32;
                let py = y as f32;
                let x3 = (px - cx) * d / fx;
                let y3 = (py - cy) * d / fy;
                let pixel = rgb[(px as u32, py as u32)];
                buf.extend_from_slice(&x3.to_le_bytes());
                buf.extend_from_slice(&y3.to_le_bytes());
                buf.extend_from_slice(&d.to_le_bytes());
                buf.push(pixel[0]); // R
                buf.push(pixel[1]); // G
                buf.push(pixel[2]); // B
            }
        }
    }

    let n = buf.len() / 15;
    let mut f = std::fs::File::create(path)?;
    let hdr = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        n
    );
    f.write_all(hdr.as_bytes())?;
    f.write_all(&buf)?;

    info!("Point cloud: {} ({} points)", path, n);
    Ok(())
}

/// Save raw depth map as binary float32.
pub fn save_depth_binary(depth: &[f32], path: &str) -> Result<()> {
    let buf: Vec<u8> = depth.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, buf)?;
    info!("Depth saved: {}", path);
    Ok(())
}
