import argparse
import os
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute volume from segmented depth .npy (pixel area = 1)")
    parser.add_argument("npy", help="Path to segmented depth .npy (outside = 0, inside = depth in mm)")
    parser.add_argument("--invalid-zero", action="store_true", help="Treat zeros as invalid/outside")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="[Deprecated] Area per pixel (use --dx)")
    parser.add_argument("--dx", type=float, default=None, help="Area per pixel (dA). Volume = sum(height) * dx")
    parser.add_argument("--print-stats", action="store_true", help="Print min/max/mean heights and count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.npy):
        raise FileNotFoundError(f"File not found: {args.npy}")

    depth = np.load(args.npy).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {depth.shape}")

    # Valid interior: non-zero (or non-nan)
    valid_mask = depth > 0 if args.invalid_zero else ~np.isnan(depth)
    if not np.any(valid_mask):
        print("No valid interior pixels. Volume = 0")
        return

    interior_depth = depth[valid_mask]
    min_depth = float(np.min(interior_depth))

    # Height = min_depth - depth (if depth is distance from camera, smaller = higher). If depth unit increases with distance,
    # the relative height above the bottom (min depth) is (depth - min_depth). The user asked '가장 낮은 데이터를 기준으로 다른 부분의 높이'.
    # '낮은' = 작은 값 → 바닥. 다른 부분의 높이 = depth - min_depth.
    heights = interior_depth - min_depth
    heights[heights < 0] = 0.0

    # Riemann-sum style: Volume = sum(height) * dA, where dA is pixel area
    pixel_area = float(args.dx) if args.dx is not None else float(args.unit_scale)
    volume = float(np.sum(heights) * pixel_area)

    if args.print_stats:
        print(f"min_depth: {min_depth:.3f}")
        print(f"height_min: {float(np.min(heights)):.3f}")
        print(f"height_max: {float(np.max(heights)):.3f}")
        print(f"height_mean: {float(np.mean(heights)):.3f}")
        print(f"num_pixels: {interior_depth.size}")

    print(f"Volume (Riemann sum: sum(height) * dx, dx={pixel_area}): {volume:.3f}")


if __name__ == "__main__":
    main()


