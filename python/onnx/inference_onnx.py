#!/usr/bin/env python3
"""YOLO26 Depth inference using ONNX Runtime (CPU).

Usage:
    # Single inference
    python inference_onnx.py --model yolo26n-depth.onnx --image bus.jpg --save result.png

    # Benchmark
    python inference_onnx.py --model yolo26n-depth.onnx --image bus.jpg --benchmark 100

    # Auto-detect input size from ONNX metadata
    python inference_onnx.py --model yolo26n-depth_768.onnx --image bus.jpg

Requires: pip install onnxruntime opencv-python numpy onnx
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

# Import shared utils (works both as repo module and standalone script)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from yolo26depth_rknn.utils import (
    detect_imgsz_from_onnx,
    parse_imgsz,
    prepare_input_rect,
    save_outputs,
)


class YOLO26DepthONNX:
    """ONNX depth estimation model wrapper."""

    def __init__(self, model_path: str, imgsz: str | int | None = None):
        self.model_path = model_path
        # (H, W): CLI arg > ONNX metadata
        if imgsz is not None:
            self.imgsz = parse_imgsz(imgsz)
        else:
            self.imgsz = detect_imgsz_from_onnx(model_path)
        self.sess = ort.InferenceSession(str(model_path), providers=[
            "CPUExecutionProvider"
        ])
        self.input_name = self.sess.get_inputs()[0].name
        out_shape = self.sess.get_outputs()[0].shape
        print(f"ONNX model: {model_path} (imgsz={self.imgsz}, output={out_shape})")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference, return (H, W) float32 depth at original image size.

        Preprocessing matches ultralytics PT predict: the image is resized with
        aspect-ratio-preserving rect scaling.  If the image aspect ratio differs
        from the model's, a warning is emitted.
        """
        src_h, src_w = image.shape[:2]

        # Rect-aware preprocessing (float32 NCHW, /255 for ONNX)
        inp = prepare_input_rect(image, self.imgsz, normalize=True)
        out = self.sess.run(None, {self.input_name: inp})[0]
        depth = np.squeeze(out)

        # Resize depth back to original image size
        depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        return depth


def print_depth_stats(depth: np.ndarray):
    """Print depth statistics to stdout."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        print("  No valid depth values!")
        return
    print(f"  Median: {np.median(valid):.2f} m")
    print(f"  5%-95%: {np.percentile(valid, 5):.1f} ~ {np.percentile(valid, 95):.1f} m")


def main():
    parser = argparse.ArgumentParser(description="YOLO26-Depth ONNX Inference")
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--save", default=None, help="Save overlay (image + heatmap blend)")
    parser.add_argument("--save-heat", default=None, help="Save standalone heatmap")
    parser.add_argument("--save-depth", default=None, help="Save raw depth as .npy")
    parser.add_argument("--mode", choices=["disparity", "metric"], default="disparity",
                        help="Colorize mode (default: disparity)")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Benchmark N iterations (0=off)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--imgsz", type=str, default=None,
                        help="Input size, e.g. 640 or 640x480 (auto-detected from ONNX metadata)")
    args = parser.parse_args()

    # Load model and image
    model = YOLO26DepthONNX(args.model, imgsz=args.imgsz)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    print(f"Input image: {image.shape[1]}x{image.shape[0]}")

    # Warmup
    for _ in range(args.warmup):
        model.predict(image)

    if args.benchmark > 0:
        # Benchmark mode
        times = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            model.predict(image)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        print(f"\nBenchmark ({args.benchmark} iterations):")
        print(f"  Latency: {avg:.1f} ms ({1000 / avg:.2f} FPS)")
        print(f"  Min: {min(times):.1f} ms  Max: {max(times):.1f} ms")
    else:
        # Single inference
        t0 = time.perf_counter()
        depth = model.predict(image)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nInference: {elapsed:.1f} ms")
        print_depth_stats(depth)

        # Save outputs
        save_outputs(depth, image,
                     save_path=args.save,
                     save_depth=args.save_depth,
                     save_heat=args.save_heat,
                     mode=args.mode)


if __name__ == "__main__":
    main()
