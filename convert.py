#!/usr/bin/env python3
"""Convert YOLO26-Depth ONNX models to RKNN for RK3588.

Usage:
    # Convert a single model
    python convert.py --model yolo26n-depth.onnx

    # Convert multiple models
    python convert.py --model yolo26n-depth.onnx yolo26s-depth.onnx yolo26x-depth.onnx

    # With INT8 quantization
    python convert.py --model yolo26n-depth.onnx --quantize --dataset datasets.txt

Requirements:
    pip install rknn-toolkit2 opencv-python numpy onnx
"""
from __future__ import annotations
import argparse
import os
import time

import cv2
import numpy as np
import onnx
import os
import sys
from rknn.api import RKNN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo26depth_rknn.utils import prepare_input_rect, detect_imgsz_from_onnx


TARGET_PLATFORM = "rk3588"
DISABLE_RULES = ["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"]
DATASET = "./datasets.txt"


def detect_input_size(onnx_path: str) -> tuple[int, int]:
    """Read input (H, W) from ONNX model."""
    m = onnx.load(onnx_path)
    shape = m.graph.input[0].type.tensor_type.shape.dim
    return int(shape[2].dim_value), int(shape[3].dim_value)


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    quantize: bool = False,
    dataset: str | None = None,
    test_image: str | None = None,
) -> bool:
    """Convert a single ONNX model to RKNN.

    Returns True on success.
    """
    rknn = RKNN(verbose=False)

    # Config
    config = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": TARGET_PLATFORM,
        "disable_rules": DISABLE_RULES,
    }
    if quantize:
        config.update({
            "quantized_algorithm": "normal",
            "quantized_method": "channel",
        })
    ret = rknn.config(**config)
    if ret != 0:
        print(f"  Config failed: {ret}")
        return False

    # Load ONNX
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  Load ONNX failed: {ret}")
        return False

    # Build
    t0 = time.time()
    build_kwargs = {"do_quantization": quantize, "rknn_batch_size": 1}
    if quantize:
        ds = dataset or DATASET
        if not os.path.exists(ds):
            print(f"  Dataset file not found: {ds}")
            return False
        build_kwargs["dataset"] = ds
    ret = rknn.build(**build_kwargs)
    if ret != 0:
        print(f"  Build failed: {ret}")
        return False
    build_time = time.time() - t0

    # Export
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"  Export failed: {ret}")
        return False

    # Optional: verify with test image
    if test_image and os.path.exists(test_image):
        rknn.init_runtime()
        img = cv2.imread(test_image)
        imgsz = detect_imgsz_from_onnx(onnx_path)
        inp_np = prepare_input_rect(img, imgsz, normalize=False)
        outputs = rknn.inference(inputs=[inp_np])
        depth = np.squeeze(outputs[0]).astype(np.float32)
        valid = depth[depth > 0]
        print(f"  Verify: shape={depth.shape}, median={np.median(valid):.2f} m")
        try:
            rknn.release()
        except Exception:
            pass

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO26-Depth ONNX to RKNN")
    parser.add_argument("--model", type=str, nargs="+", required=True,
                        help="Path(s) to .onnx model file(s)")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 quantization (see README: accuracy loss is large for this model)")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset file for quantization (default: datasets.txt)")
    parser.add_argument("--test-image", type=str, default=None,
                        help="Test image path for verification")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (default: same as input)")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if RKNN already exists")
    args = parser.parse_args()

    success_count = 0
    skip_count = 0

    for onnx_path in args.model:
        if not os.path.exists(onnx_path):
            print(f"Model not found: {onnx_path}")
            continue

        base = os.path.splitext(os.path.basename(onnx_path))[0]
        suffix = "-float" if not args.quantize else "-int8"
        rknn_name = f"{base}{suffix}.rknn"
        rknn_path = os.path.join(args.output_dir, rknn_name)

        # Skip if exists
        if os.path.exists(rknn_path) and not args.force:
            print(f"\n  Skip (exists): {rknn_name}  (--force to re-convert)")
            skip_count += 1
            continue

        print(f"\n{'='*60}")
        print(f"  {onnx_path} → {rknn_name}")
        print(f"  Quantize: {args.quantize}")
        print(f"{'='*60}")

        if convert_onnx_to_rknn(
            onnx_path, rknn_path,
            quantize=args.quantize,
            dataset=args.dataset,
            test_image=args.test_image,
        ):
            success_count += 1
        else:
            print(f"  FAILED: {onnx_path}")

    total = len(args.model)
    print(f"\nDone. {success_count} converted, {skip_count} skipped, {total - success_count - skip_count} failed out of {total}.")


if __name__ == "__main__":
    main()
