#!/usr/bin/env python3
"""Multi-core NPU throughput benchmark using a thread pool.

Spawns one worker thread per NPU core (RK3588 has 3). Each worker owns its
own RKNN instance pinned to a dedicated core, and frames are dispatched to
the pool asynchronously — simulating a video pipeline.

Usage (on RK3588):
    python examples/benchmark_pool.py --model yolo26n-depth-float.rknn \
        --image bus.jpg --frames 90

    # Single core baseline for comparison
    python examples/benchmark_pool.py --model yolo26n-depth-float.rknn \
        --image bus.jpg --frames 30 --cores 0
"""
from __future__ import annotations
import argparse
import itertools
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_rknn import YOLO26DepthRKNN  # noqa: E402

_tls = threading.local()
_core_iter = itertools.count()


def _init_worker(model_path: str, imgsz: int | None, cores: list[str]):
    """Create a per-thread model instance pinned to the next core."""
    core = cores[next(_core_iter) % len(cores)]
    _tls.model = YOLO26DepthRKNN(model_path, imgsz=imgsz, core=core)
    _tls.core = core
    _tls.count = 0


def _infer(image: np.ndarray) -> tuple[str, float]:
    """Run one inference, return (core, latency_ms)."""
    t0 = time.perf_counter()
    _tls.model.predict(image)
    _tls.count += 1
    return _tls.core, (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser(description="Thread-pool NPU throughput benchmark")
    parser.add_argument("--model", required=True, help="Path to .rknn model")
    parser.add_argument("--image", required=True, help="Input image (reused as every frame)")
    parser.add_argument("--frames", type=int, default=90, help="Total frames to process")
    parser.add_argument("--cores", nargs="+", default=["0", "1", "2"],
                        choices=["auto", "0", "1", "2", "012"],
                        help="NPU cores; one worker thread per entry (default: 0 1 2)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup frames per worker")
    parser.add_argument("--imgsz", type=int, default=None, help="Input size (auto-detected)")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    n_workers = len(args.cores)
    pool = ThreadPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(args.model, args.imgsz, args.cores),
    )

    # Warmup (also forces lazy worker/model creation)
    list(pool.map(_infer, [image] * (args.warmup * n_workers)))

    # Async dispatch: submit all frames, then collect
    t0 = time.perf_counter()
    futures = [pool.submit(_infer, image) for _ in range(args.frames)]
    results = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    pool.shutdown()

    latencies = sorted(ms for _, ms in results)
    per_core = {}
    for core, _ in results:
        per_core[core] = per_core.get(core, 0) + 1

    print(f"\nThread-pool benchmark: {args.frames} frames, "
          f"{n_workers} workers on cores {args.cores}")
    print(f"  Wall time:  {wall:.2f} s")
    print(f"  Throughput: {args.frames / wall:.2f} FPS (aggregate)")
    print(f"  Latency:    median {latencies[len(latencies) // 2]:.1f} ms, "
          f"p95 {latencies[int(len(latencies) * 0.95)]:.1f} ms")
    print(f"  Frames per core: {per_core}")


if __name__ == "__main__":
    main()
