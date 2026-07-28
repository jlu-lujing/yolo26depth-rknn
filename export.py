#!/usr/bin/env python3
"""Export YOLO26-Depth PyTorch models to ONNX.

Supports all model sizes (n/s/m/l/x) and input resolutions (640/768/960/1280).

Usage:
    # Export a single model
    python export.py --model yolo26n-depth.pt --imgsz 640

    # Export multiple models
    python export.py --model yolo26n-depth.pt yolo26s-depth.pt --imgsz 640

    # Export all models at 640
    python export.py --all --imgsz 640

    # Export x at multiple resolutions
    python export.py --model yolo26x-depth.pt --imgsz 640 768 960 1280

    # Rect (non-square, HxW) export — matches ultralytics' aspect-preserving
    # rect inference exactly when W/H equals your camera aspect ratio
    python export.py --model yolo26n-depth.pt --imgsz 640x480

Requirements:
    pip install ultralytics onnxruntime onnx onnx-simplifier
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import onnx
import onnxruntime as ort
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo26depth_rknn.utils import parse_imgsz


SUPPORTED_MODELS = {
    "yolo26n-depth.pt": "n",
    "yolo26s-depth.pt": "s",
    "yolo26m-depth.pt": "m",
    "yolo26l-depth.pt": "l",
    "yolo26x-depth.pt": "x",
}

SUPPORTED_IMGSZ = [640, 768, 960, 1280]


def export_model(pt_path: str, imgsz: tuple[int, int], output_dir: str = ".", force: bool = False) -> str | None:
    """Export a single .pt model to ONNX at the given (H, W) resolution.

    Returns the path to the exported ONNX file, or None if skipped.
    """
    pt_name = os.path.basename(pt_path)
    pt_dir = os.path.dirname(os.path.abspath(pt_path))
    if pt_name not in SUPPORTED_MODELS:
        print(f"  Warning: {pt_name} not in supported models list, exporting anyway.")

    h, w = imgsz
    if (h, w) == (640, 640):
        onnx_name = pt_name.replace(".pt", ".onnx")
    elif h == w:
        onnx_name = pt_name.replace(".pt", f"_{h}.onnx")
    else:
        onnx_name = pt_name.replace(".pt", f"_{h}x{w}.onnx")

    output_path = os.path.join(output_dir, onnx_name)

    # Skip if already exists
    if os.path.exists(output_path) and not force:
        print(f"\n  Skip (exists): {onnx_name}  (--force to re-export)")
        return None

    # Remove stale output to avoid confusion
    if os.path.exists(output_path):
        os.remove(output_path)

    model = YOLO(pt_path)

    print(f"\n{'='*60}")
    print(f"  {pt_name} → {onnx_name}  (imgsz={h}x{w})")
    print(f"{'='*60}")

    t0 = time.time()
    model.export(
        format="onnx",
        imgsz=[h, w],
        simplify=True,
        opset=19,  # rknn-toolkit2 2.3.2 requires opset <= 19
    )
    elapsed = time.time() - t0

    # Ultralytics saves ONNX next to the .pt file with the base name;
    # move it to output_dir with the correct name.
    base_onnx = pt_name.replace(".pt", ".onnx")
    candidates = [
        os.path.join(pt_dir, base_onnx),       # next to .pt file
        os.path.join(output_dir, base_onnx),   # current dir
    ]
    for src in candidates:
        if src != output_path and os.path.exists(src):
            os.replace(src, output_path)
            break

    # Verify
    if not os.path.exists(output_path):
        print(f"  ERROR: Output not found at {output_path}")
        return None

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out_shape = sess.get_outputs()[0].shape
    m = onnx.load(output_path)
    opset = m.opset_import[0].version

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Output shape: {out_shape}")
    print(f"  Opset: {opset}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26-Depth to ONNX")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Path(s) to .pt model file(s)")
    parser.add_argument("--all", action="store_true",
                        help="Export all supported models")
    parser.add_argument("--imgsz", type=str, nargs="+", default=["640"],
                        help="Input resolution(s): 640, 768... or HxW rect like 640x480. Default: 640")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory. Default: current directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if ONNX already exists")
    args = parser.parse_args()

    # Determine which models to export
    if args.all:
        model_paths = list(SUPPORTED_MODELS)
    elif args.model:
        model_paths = args.model  # already a list from nargs="+"
    else:
        parser.error("Specify --model or --all")
        return

    # Missing local files: bare official names (e.g. "yolo26n-depth.pt") are
    # auto-downloaded by ultralytics; other paths are dropped with a warning.
    resolved = []
    for p in model_paths:
        if os.path.exists(p):
            resolved.append(p)
        elif os.path.basename(p) in SUPPORTED_MODELS:
            print(f"Not found locally, will auto-download: {os.path.basename(p)}")
            resolved.append(os.path.basename(p))
        else:
            print(f"Warning: {p} not found, skipping.")
    model_paths = resolved
    if not model_paths:
        print("No model files found.")
        sys.exit(1)

    # Export
    count = 0
    for pt_path in model_paths:
        for spec in args.imgsz:
            h, w = parse_imgsz(spec)
            if h == w and h not in SUPPORTED_IMGSZ:
                print(f"  Warning: imgsz={h} not in recommended sizes {SUPPORTED_IMGSZ}")
            if h % 32 or w % 32:
                print(f"  Warning: imgsz={h}x{w} is not a multiple of 32 (network stride)")
            if export_model(pt_path, (h, w), args.output, args.force):
                count += 1

    print(f"\nDone. Exported {count} model(s).")


if __name__ == "__main__":
    main()
