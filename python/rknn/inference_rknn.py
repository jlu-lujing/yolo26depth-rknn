#!/usr/bin/env python3
"""YOLO26 Depth inference using RKNN on RK3588 NPU.

Usage:
    # Single inference with overlay
    python inference_rknn.py --model yolo26n-depth-float.rknn --image bus.jpg --save result.png

    # Benchmark (100 iterations)
    python inference_rknn.py --model yolo26n-depth-float.rknn --image bus.jpg --benchmark

    # Save heatmap + raw depth
    python inference_rknn.py --model yolo26n-depth_768-float.rknn --image bus.jpg \
        --save-heat heatmap.png --save-depth depth.npy

Requires: rknn-toolkit-lite2 on RK3588 (or rknn-toolkit2 on x86)
"""
from __future__ import annotations
import argparse
import importlib.util
import time

import cv2
import numpy as np

# Try on-device runtime first, fall back to simulator
if importlib.util.find_spec("rknnlite"):
    from rknnlite.api import RKNNLite  # noqa
    _RKNN = RKNNLite
else:
    from rknn.api import RKNN  # noqa
    _RKNN = RKNN

# Import shared utils (works both as repo module and standalone script)
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from yolo26depth_rknn.utils import (
    detect_imgsz_from_path,
    parse_imgsz,
    prepare_input_rect,
    save_outputs,
)


class YOLO26DepthRKNN:
    """RKNN depth estimation model wrapper."""

    # Maps --core choice to RKNNLite core mask (RK3588 has 3 NPU cores)
    CORE_MASKS = {
        "auto": 0,  # NPU_CORE_AUTO
        "0": 1,     # NPU_CORE_0
        "1": 2,     # NPU_CORE_1
        "2": 4,     # NPU_CORE_2
        "012": 7,   # NPU_CORE_0_1_2
    }

    def __init__(self, model_path: str, imgsz: str | int | None = None, core: str = "auto"):
        # (H, W): from --imgsz (e.g. "640" or "640x480") or the model filename
        self.imgsz = parse_imgsz(imgsz) if imgsz else detect_imgsz_from_path(model_path)
        self.rknn = _RKNN()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        try:
            ret = self.rknn.init_runtime(core_mask=self.CORE_MASKS[core])
        except TypeError:
            # rknn-toolkit2 simulator has no core_mask parameter
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")
        print(f"RKNN model: {model_path} (imgsz={self.imgsz}, core={core})")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image into model input tensor (uint8 NHWC).

        Can be called once and reused across benchmark iterations.
        """
        return prepare_input_rect(image, self.imgsz, normalize=False)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run NPU inference on a preprocessed input tensor.

        Returns (H, W) float32 depth at model output resolution (not original size).
        """
        outputs = self.rknn.inference(inputs=[input_tensor])
        # Output is already float32 — no redundant astype()
        return np.squeeze(outputs[0])

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference, return (H, W) float32 depth at original image size.

        Preprocessing matches ultralytics PT predict: the image is resized with
        aspect-ratio-preserving rect scaling.  If the image aspect ratio differs
        from the model's, a warning is emitted.
        """
        src_h, src_w = image.shape[:2]
        tensor = self.preprocess(image)
        depth = self.infer(tensor)
        # Resize depth back to original image size
        depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def __del__(self):
        try:
            self.rknn.release()
        except Exception:
            pass


def print_depth_stats(depth: np.ndarray):
    """Print depth statistics to stdout."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        print("  No valid depth values!")
        return
    print(f"  Median: {np.median(valid):.2f} m")
    print(f"  5%-95%: {np.percentile(valid, 5):.1f} ~ {np.percentile(valid, 95):.1f} m")


def main():
    parser = argparse.ArgumentParser(description="YOLO26-Depth RKNN Inference")
    parser.add_argument("--model", required=True, help="Path to .rknn model")
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
                        help="Input size, e.g. 640 or 640x480 (auto-detected from model name)")
    parser.add_argument("--core", choices=["auto", "0", "1", "2", "012"], default="auto",
                        help="NPU core selection (default: auto)")
    args = parser.parse_args()

    # Load model and image
    model = YOLO26DepthRKNN(args.model, imgsz=args.imgsz, core=args.core)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    print(f"Input image: {image.shape[1]}x{image.shape[0]}")

    # Preprocess once
    tensor = model.preprocess(image)

    # Warmup
    for _ in range(args.warmup):
        model.infer(tensor)

    if args.benchmark > 0:
        # Benchmark: time NPU infer + resize back to original size
        src_h, src_w = image.shape[:2]
        times = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            depth = model.infer(tensor)
            depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        print(f"\nBenchmark ({args.benchmark} iterations):")
        print(f"  Latency: {avg:.1f} ms ({1000 / avg:.2f} FPS)")
        print(f"  Min: {min(times):.1f} ms  Max: {max(times):.1f} ms")
    else:
        # Single inference
        t0 = time.perf_counter()
        depth = model.infer(tensor)
        depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
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
