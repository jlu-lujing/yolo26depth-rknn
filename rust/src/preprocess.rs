//! Image preprocessing: load, resize to model input size with rect (aspect-ratio-preserving) mode.
//!
//! Matches ultralytics PT predict: the image is scaled so the longer side equals
//! the model's max dimension while keeping the aspect ratio, then rounded to the
//! nearest stride-32 multiple.  If the computed rect size matches the model input,
//! the resize is direct.  Otherwise a warning is logged and the image is stretched.
//!
//! Uses plain bilinear sampling (matching OpenCV `INTER_LINEAR`) rather than
//! `image::imageops::resize`, whose Triangle filter applies anti-aliasing on
//! downscale and produces different pixels than the Python pipeline.

use image::GenericImageView;

use crate::error::Result;

const STRIDE: f64 = 32.0;

/// Compute rect input size matching ultralytics behavior.
///
/// `src_w`/`src_h` are the source image dimensions; `model_max_dim` is the
/// larger of the model's input H/W (for square models this is just imgsz).
/// Returns `(rect_h, rect_w)` stride-32 aligned.
fn compute_rect_input(src_h: u32, src_w: u32, model_max_dim: u32) -> (u32, u32) {
    let scale = model_max_dim as f64 / src_h.max(src_w) as f64;
    let new_h = (src_h as f64 * scale / STRIDE).round() as u32 * 32;
    let new_w = (src_w as f64 * scale / STRIDE).round() as u32 * 32;
    (new_h.max(32), new_w.max(32))
}

/// Plain bilinear resize of an RGB8 buffer — bit-compatible with
/// `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
fn resize_bilinear_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    for y in 0..dst_h {
        let fy = ((y as f64 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f64;

        for x in 0..dst_w {
            let fx = ((x as f64 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f64;

            let p00 = (y0 * src_w + x0) * 3;
            let p01 = (y0 * src_w + x1) * 3;
            let p10 = (y1 * src_w + x0) * 3;
            let p11 = (y1 * src_w + x1) * 3;
            let d = (y * dst_w + x) * 3;

            for c in 0..3 {
                let v = src[p00 + c] as f64 * (1.0 - wx) * (1.0 - wy)
                    + src[p01 + c] as f64 * wx * (1.0 - wy)
                    + src[p10 + c] as f64 * (1.0 - wx) * wy
                    + src[p11 + c] as f64 * wx * wy;
                dst[d + c] = (v + 0.5) as u8;
            }
        }
    }
    dst
}

/// Load an image and resize it to the model's input dimensions with rect preprocessing.
///
/// Returns a flat RGB buffer (NHWC layout) suitable for RKNN input.
pub fn preprocess(img_path: &str, target_w: usize, target_h: usize) -> Result<Vec<u8>> {
    let img = image::open(img_path)?;
    let (orig_w, orig_h) = img.dimensions();

    // Compute rect input size from image aspect ratio
    let model_max_dim = target_h.max(target_w) as u32;
    let (rect_h, rect_w) = compute_rect_input(orig_h, orig_w, model_max_dim);

    let (rw, rh): (usize, usize) = if rect_h == target_h as u32 && rect_w == target_w as u32 {
        (target_w, target_h)
    } else {
        log::warn!(
            "Image aspect ratio ({}:{}) does not match model aspect ratio ({}:{}). \
             Rect input would be {}x{} but model expects {}x{}. \
             Results will differ from ultralytics PT predict. \
             Export a rect model with --imgsz {}x{} for best accuracy.",
            orig_w, orig_h, target_w, target_h, rect_w, rect_h, target_w, target_h, rect_w, rect_h
        );
        // Fallback: stretch to model size
        (target_w, target_h)
    };

    log::info!(
        "Preprocess: {}x{} -> {}x{} (bilinear)",
        orig_w, orig_h, rw, rh
    );

    let rgb = img.to_rgb8();
    Ok(resize_bilinear_rgb(
        rgb.as_raw(),
        orig_w as usize,
        orig_h as usize,
        rw,
        rh,
    ))
}

/// Load an image and return it as a DynamicImage (for point cloud color lookup).
pub fn load_image(img_path: &str) -> Result<image::DynamicImage> {
    Ok(image::open(img_path)?)
}
