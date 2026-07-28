#!/usr/bin/env python3
"""Shared utilities for YOLO26-Depth RKNN inference."""
from __future__ import annotations

import re
import warnings

import cv2
import numpy as np


STRIDE = 32  # YOLO network stride — all input dims must be multiples of 32


def _align32(x: int) -> int:
    """Round an integer to the nearest multiple of STRIDE, clamped to >= 32."""
    return max(int(round(x / STRIDE) * STRIDE), STRIDE)


def compute_rect_input(src_h: int, src_w: int, model_max_dim: int) -> tuple[int, int]:
    """Compute rect (aspect-ratio-preserving) input size, matching ultralytics PT predict.

    Ultralytics' rect inference scales the image so that the longer side equals
    *model_max_dim* while keeping the aspect ratio, then rounds each dimension
    to the nearest stride-32 multiple.

    Args:
        src_h: Source image height.
        src_w: Source image width.
        model_max_dim: The reference dimension (usually the larger of model's H/W
                       for square models this is just imgsz).

    Returns:
        (input_h, input_w) stride-32 aligned.
    """
    scale = model_max_dim / max(src_h, src_w)
    new_h = _align32(src_h * scale)
    new_w = _align32(src_w * scale)
    return new_h, new_w


def derive_model_max_dim(imgsz: tuple[int, int]) -> int:
    """Derive the effective 'max_dim' from a model's fixed input size (H, W).

    For square models (640,640) this is 640.
    For rect models (640,480) this is 640 (the larger dimension).
    """
    return max(imgsz)


def parse_imgsz(value: str | int) -> tuple[int, int]:
    """Parse an imgsz spec into (H, W).

    Examples:
        640       -> (640, 640)
        "640"     -> (640, 640)
        "640x480" -> (640, 480)   # H x W, rect model
    """
    s = str(value)
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    return int(s), int(s)


def detect_imgsz_from_path(model_path: str) -> tuple[int, int]:
    """Auto-detect input size (H, W) from model filename.

    Examples:
        yolo26n-depth-float.rknn           -> (640, 640)
        yolo26n-depth_768-float.rknn       -> (768, 768)
        yolo26x-depth_1280.onnx            -> (1280, 1280)
        yolo26n-depth_640x480-float.rknn   -> (640, 480)
    """
    m = re.search(r'_(\d+x\d+|\d+)[-.]', model_path)
    return parse_imgsz(m.group(1)) if m else (640, 640)


def detect_imgsz_from_onnx(onnx_path: str) -> tuple[int, int]:
    """Read input size (H, W) from ONNX model metadata."""
    import onnx
    m = onnx.load(onnx_path)
    shape = m.graph.input[0].type.tensor_type.shape.dim
    return int(shape[2].dim_value), int(shape[3].dim_value)  # H, W in N,C,H,W


def prepare_input_rect(image: np.ndarray, imgsz: tuple[int, int],
                       normalize: bool = True) -> np.ndarray:
    """Preprocess image with rect (aspect-ratio-preserving) inference, matching ultralytics PT predict.

    For each input image, the rect input size is computed from the image's aspect
    ratio. If it matches the model's fixed input size (H, W), the image is resized
    directly. Otherwise a warning is emitted and the image is stretched (fallback).

    Args:
        image: BGR source image (H, W, 3) uint8.
        imgsz: Model input size (H, W) — fixed for ONNX/RKNN models.
        normalize: If True, convert to range [0, 1] float32 NCHW (for ONNX).
                   If False, return uint8 NHWC (for RKNN which handles /255 internally).

    Returns:
        Preprocessed input tensor.
    """
    src_h, src_w = image.shape[:2]
    model_max_dim = derive_model_max_dim(imgsz)
    rect_h, rect_w = compute_rect_input(src_h, src_w, model_max_dim)
    mh, mw = imgsz

    if rect_h != mh or rect_w != mw:
        warnings.warn(
            f"Image aspect ratio ({src_w}:{src_h}) does not match model aspect ratio "
            f"({mw}:{mh}). Rect input would be {rect_h}x{rect_w} but model expects "
            f"{mh}x{mw}. Results will differ from ultralytics PT predict. "
            f"Export a rect model with --imgsz {rect_h}x{rect_w} for best accuracy.",
            UserWarning, stacklevel=2
        )
        # Fallback: stretch to model size
        rh, rw = mh, mw
    else:
        rh, rw = mh, mw

    inp = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    if normalize:
        # ONNX: float32 NCHW, /255
        inp_np = rgb.astype(np.float32) / 255.0
        return inp_np.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    else:
        # RKNN: uint8 NHWC (internal /255)
        return rgb[np.newaxis]


def colorize_depth(depth: np.ndarray, mode: str = "disparity") -> np.ndarray:
    """Convert metric depth map to a BGR heatmap image.

    Args:
        depth: (H, W) float32 depth map in meters.
        mode: "disparity" (1/d with percentile clipping) or "metric" (linear).

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if mode == "disparity":
        v = np.zeros_like(depth)
        v[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
        lo, hi = np.percentile(v[valid], (2, 98))
    else:
        v = depth
        lo = float(depth[valid].min())
        hi = float(depth[valid].max())

    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(idx, cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def save_outputs(depth: np.ndarray, image: np.ndarray, save_path: str | None = None,
                 save_depth: str | None = None, save_heat: str | None = None,
                 mode: str = "disparity", overlay_alpha: float = 0.5):
    """Save inference results (overlay, heatmap, raw depth).

    Args:
        depth: (H, W) float32 depth map.
        image: Original BGR image (H, W, 3) uint8.
        save_path: If set, save overlay (image + heatmap blended).
        save_depth: If set, save raw depth as .npy.
        save_heat:  If set, save standalone heatmap image.
        mode:       "disparity" or "metric" for colorization.
        overlay_alpha: Blend weight for heatmap (0-1).
    """
    if save_path or save_heat:
        heat = colorize_depth(depth, mode=mode)

    if save_heat:
        cv2.imwrite(save_heat, heat)
        print(f"  Heatmap saved: {save_heat}")

    if save_path:
        overlay = cv2.addWeighted(image, 1 - overlay_alpha, heat, overlay_alpha, 0)
        cv2.imwrite(save_path, overlay)
        print(f"  Overlay saved: {save_path}")

    if save_depth:
        np.save(save_depth, depth.astype(np.float32))
        print(f"  Depth saved: {save_depth}")
