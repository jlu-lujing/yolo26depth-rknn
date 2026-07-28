#!/usr/bin/env python3
"""Convert depth map + image to colored point cloud (.ply).

Supports reading depth from .npy files or raw binary float32.

Usage:
    python depth_to_pointcloud.py --depth depth.npy --image bus.jpg --output pointcloud.ply
    python depth_to_pointcloud.py --depth depth.npy --image bus.jpg --output pointcloud.ply --downsample 4
"""
from __future__ import annotations
import argparse
import os

import cv2
import numpy as np


def default_intrinsics(w: int, h: int) -> np.ndarray:
    """Return default camera intrinsics (approximate 50-degree FOV)."""
    fx = float(h)
    fy = float(h)
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def load_depth(path: str, image_shape: tuple | None = None) -> np.ndarray:
    """Load depth map from .npy or raw binary float32 file.

    For raw binary: infers shape from image_shape or file size.
    """
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)

    # Raw binary float32
    size = os.path.getsize(path)
    n = size // 4
    if image_shape is not None:
        h, w = image_shape[:2]
        if n != h * w:
            raise ValueError(f"Depth file size ({n}) doesn't match image ({w}*{h}={w*h})")
        data = np.fromfile(path, dtype=np.float32).reshape(h, w)
    else:
        # Try common square sizes
        import math
        side = int(math.sqrt(n))
        if side * side == n:
            data = np.fromfile(path, dtype=np.float32).reshape(side, side)
        else:
            raise ValueError(f"Cannot infer depth shape from {n} floats. Provide --h --w.")
    return data


def generate_pointcloud(
    image: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray | None = None,
    downsample: int = 1,
    max_depth: float | None = None,
    min_depth: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate colored point cloud from image and depth map.

    Args:
        image:      BGR image (H, W, 3) uint8.
        depth:      Depth map (H, W) float32 in meters.
        K:          Camera intrinsics (3, 3), auto-generated if None.
        downsample: Skip every N pixels in each direction.
        max_depth:  Ignore points beyond this distance (meters).
        min_depth:  Ignore points closer than this distance (meters).

    Returns:
        points: (N, 3) float32 array (XYZ).
        colors: (N, 3) uint8 array (RGB).
    """
    h, w = depth.shape[:2]
    if K is None:
        K = default_intrinsics(w, h)

    # Create valid mask with downsampling
    valid = np.isfinite(depth) & (depth >= min_depth)
    if max_depth is not None:
        valid = valid & (depth <= max_depth)

    # Downsample
    ys, xs = np.where(valid[::downsample, ::downsample])

    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    # Back-project to 3D (vectorized)
    u = (xs * downsample).astype(np.float32)
    v = (ys * downsample).astype(np.float32)
    d = depth[ys, xs]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    points = np.stack([x, y, z], axis=1)

    # Colors: BGR -> RGB (manual swap — cv2.cvtColor on (N,3) array
    # produces (N,3,3) due to treating each column as a channel)
    bgr = image[ys, xs]
    rgb = bgr[:, [2, 1, 0]]

    return points, rgb


def write_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Write colored point cloud to binary PLY file (vectorized, fast).

    Args:
        filepath: Output .ply file path.
        points:   (N, 3) float32 array (XYZ).
        colors:   (N, 3) uint8 array (RGB).
    """
    n = len(points)
    with open(filepath, "wb") as f:
        # Header
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"end_header\n"
        )
        f.write(header.encode("ascii"))

        # Build per-vertex buffer: 3xfloat32 + 3xuint8 = 15 bytes each
        vertex_struct = np.empty(n, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', '<u1'), ('g', '<u1'), ('b', '<u1'),
        ])
        vertex_struct['x'] = points[:, 0]
        vertex_struct['y'] = points[:, 1]
        vertex_struct['z'] = points[:, 2]
        # .copy() needed — structured array assignment rejects non-contiguous views
        vertex_struct['r'] = colors[:, 0].copy()
        vertex_struct['g'] = colors[:, 1].copy()
        vertex_struct['b'] = colors[:, 2].copy()
        f.write(vertex_struct.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Depth Map to Point Cloud")
    parser.add_argument("--depth", required=True, help="Path to depth (.npy or raw binary float32)")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--output", required=True, help="Output .ply file path")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor (1=full, 2=half, 4=quarter)")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="Maximum depth to include (meters)")
    parser.add_argument("--min-depth", type=float, default=0.1,
                        help="Minimum depth to include (meters)")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y")
    parser.add_argument("--w", type=int, default=None, help="Depth width (for raw binary)")
    parser.add_argument("--h", type=int, default=None, help="Depth height (for raw binary)")
    args = parser.parse_args()

    # Load data
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    img_shape = (image.shape[0], image.shape[1])
    depth_shape = (args.h, args.w) if (args.h and args.w) else img_shape
    depth = load_depth(args.depth, depth_shape)

    print(f"Image: {image.shape[1]}x{image.shape[0]}")
    print(f"Depth: {depth.shape[1]}x{depth.shape[0]}")

    # Camera intrinsics
    if args.fx is not None:
        K = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0, 0, 1]], dtype=np.float32)
    else:
        K = default_intrinsics(image.shape[1], image.shape[0])
    print(f"Intrinsics: fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}, cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f}")

    # Generate point cloud
    points, colors = generate_pointcloud(
        image, depth, K=K,
        downsample=args.downsample,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )

    print(f"Points: {len(points):,}")
    if len(points) == 0:
        print("No valid points!")
        return

    # Write binary PLY
    write_ply(args.output, points, colors)
    print(f"Saved: {args.output} ({len(points):,} points, {os.path.getsize(args.output) / 1024:.0f} KB)")

    # Stats
    print(f"X range: {points[:, 0].min():.1f} ~ {points[:, 0].max():.1f}")
    print(f"Y range: {points[:, 1].min():.1f} ~ {points[:, 1].max():.1f}")
    print(f"Z range: {points[:, 2].min():.1f} ~ {points[:, 2].max():.1f}")


if __name__ == "__main__":
    main()
