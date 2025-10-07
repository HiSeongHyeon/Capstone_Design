import argparse
import os
from typing import Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment cup interior from depth .npy and save .npy (rim detection via Hough circle or min-depth based)")
    parser.add_argument("npy", help="Path to depth .npy file (units: mm)")
    parser.add_argument("--out", default=None, help="Output .npy path (default: Videos/<name>_cup.npy)")
    parser.add_argument("--invalid-zero", action="store_true", help="Treat zeros as invalid")
    parser.add_argument("--method", choices=["rim", "min", "ransac"], default="rim", help="Segmentation method: rim (Hough), ransac (edge points), or min (min-depth CC)")
    parser.add_argument(
        "--percentile-low", type=float, default=20.0, help="Low percentile for auto threshold when --delta-mm omitted (keep depths <= threshold)"
    )
    parser.add_argument(
        "--delta-mm", type=float, default=20.0, help="Keep pixels with depth <= (min_depth + delta)"
    )
    parser.add_argument("--min-size", type=int, default=200, help="Remove tiny components smaller than this (pixels)")
    # Edge/Hough parameters for rim method
    parser.add_argument("--canny1", type=float, default=150.0, help="Canny lower threshold (on 8-bit image)")
    parser.add_argument("--canny2", type=float, default=200.0, help="Canny upper threshold (on 8-bit image)")
    parser.add_argument("--hough-dp", type=float, default=1.2, help="HoughCircles dp parameter")
    parser.add_argument("--hough-min-dist", type=float, default=20.0, help="HoughCircles minDist parameter (pixels)")
    parser.add_argument("--hough-param1", type=float, default=100.0, help="HoughCircles param1 (Canny high threshold inside Hough)")
    parser.add_argument("--hough-param2", type=float, default=30.0, help="HoughCircles param2 (accumulator threshold)")
    parser.add_argument("--min-radius", type=int, default=10, help="Minimum circle radius (pixels)")
    parser.add_argument("--max-radius", type=int, default=0, help="Maximum circle radius (0 lets OpenCV decide)")
    # RANSAC parameters
    parser.add_argument("--ransac-iters", type=int, default=1500, help="RANSAC max iterations")
    parser.add_argument("--ransac-thresh", type=float, default=2.5, help="Inlier threshold in pixels")
    parser.add_argument("--ransac-min-inliers", type=int, default=120, help="Minimal inlier count to accept a circle")
    return parser.parse_args()


def compute_threshold_from_min(depth: np.ndarray, use_zero_invalid: bool, percentile_low: float, delta_mm: Optional[float]) -> float:
    valid_mask = depth > 0 if use_zero_invalid else ~np.isnan(depth)
    if not np.any(valid_mask):
        raise ValueError("No valid depth values found.")
    valid_vals = depth[valid_mask]
    min_val = float(valid_vals.min())
    if delta_mm is not None and delta_mm > 0:
        return min_val + float(delta_mm)
    p = float(np.clip(percentile_low, 0.0, 100.0))
    th = float(np.percentile(valid_vals, p))
    # Ensure threshold is at least min_val to avoid empty set
    return max(th, min_val)


def segment_interior(depth: np.ndarray, threshold: float, use_zero_invalid: bool, min_size: int) -> np.ndarray:
    valid_mask = depth > 0 if use_zero_invalid else ~np.isnan(depth)
    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=bool)

    # Binary: depths less or equal to threshold (nearer to the sensor or lower height)
    bin_mask = (depth <= threshold) & valid_mask

    if not np.any(bin_mask):
        return np.zeros_like(depth, dtype=bool)

    # Seed at the minimum depth location among valid
    depth_valid = depth.copy()
    depth_valid[~valid_mask] = np.inf
    min_idx = int(np.argmin(depth_valid))
    min_y, min_x = np.unravel_index(min_idx, depth.shape)

    # Connected components on binary mask
    bin_u8 = (bin_mask.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(depth, dtype=bool)

    label_at_min = labels[min_y, min_x]
    if label_at_min == 0:
        # If min lies outside due to numeric issues, choose the largest non-background component
        if num_labels <= 1:
            return np.zeros_like(depth, dtype=bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        label_at_min = int(np.argmax(areas)) + 1

    component_mask = labels == label_at_min

    if component_mask.sum() < max(1, min_size):
        return np.zeros_like(depth, dtype=bool)

    return component_mask


def normalize_to_uint8(depth: np.ndarray, use_zero_invalid: bool) -> np.ndarray:
    valid_mask = depth > 0 if use_zero_invalid else ~np.isnan(depth)
    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=np.uint8)
    vals = depth[valid_mask]
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (np.clip(depth, vmin, vmax) - vmin) / (vmax - vmin)
    scaled = np.nan_to_num(scaled, nan=0.0)
    img8 = (scaled * 255.0).astype(np.uint8)
    return img8


def segment_by_rim(depth: np.ndarray, use_zero_invalid: bool, min_size: int, canny1: float, canny2: float, dp: float, min_dist: float, param1: float, param2: float, min_radius: int, max_radius: int) -> np.ndarray:
    # Prepare 8-bit image for edge/circle detection
    img8 = normalize_to_uint8(depth, use_zero_invalid)
    # Optional smoothing to stabilize edges
    img_blur = cv2.GaussianBlur(img8, (5, 5), 0)
    edges = cv2.Canny(img_blur, threshold1=canny1, threshold2=canny2)

    # HoughCircles expects a grayscale input; we can pass the blurred gray image
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None or len(circles) == 0:
        return np.zeros_like(depth, dtype=bool)

    circles = np.squeeze(circles, axis=0)
    h, w = depth.shape

    # Choose circle closest to global minimum location (likely cup bottom center)
    valid_mask = depth > 0 if use_zero_invalid else ~np.isnan(depth)
    depth_valid = depth.copy()
    depth_valid[~valid_mask] = np.inf
    min_idx = int(np.argmin(depth_valid))
    min_y, min_x = np.unravel_index(min_idx, depth.shape)

    # Primary criterion: largest radius; secondary: closest to global minimum position
    # Convert to array for vectorized selection
    circles_arr = np.asarray(circles, dtype=np.float32)
    radii = circles_arr[:, 2]
    max_r = np.max(radii)
    largest_idx = np.where(np.isclose(radii, max_r))[0]
    if largest_idx.size > 1:
        # Tie-breaker by distance to min point
        subset = circles_arr[largest_idx]
        dx = subset[:, 0] - float(min_x)
        dy = subset[:, 1] - float(min_y)
        d2 = dx * dx + dy * dy
        best_local = int(largest_idx[np.argmin(d2)])
    else:
        best_local = int(largest_idx[0])

    cx, cy, r = circles_arr[best_local]
    cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))

    # Create filled circle mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx_i, cy_i), r_i, 255, thickness=-1)

    # Keep only interior (and valid pixels)
    interior = mask.astype(bool)
    interior &= valid_mask

    # Remove if too small
    if interior.sum() < max(1, min_size):
        return np.zeros_like(depth, dtype=bool)
    return interior


def fit_circle_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    denom = 2.0 * ((x1*(y2 - y3)) + (x2*(y3 - y1)) + (x3*(y1 - y2)))
    if abs(denom) < 1e-6:
        return None
    ux = ((x1*x1 + y1*y1)*(y2 - y3) + (x2*x2 + y2*y2)*(y3 - y1) + (x3*x3 + y3*y3)*(y1 - y2)) / denom
    uy = ((x1*x1 + y1*y1)*(x3 - x2) + (x2*x2 + y2*y2)*(x1 - x3) + (x3*x3 + y3*y3)*(x2 - x1)) / denom
    r = np.hypot(ux - x1, uy - y1)
    return ux, uy, r


def segment_by_ransac(
    depth: np.ndarray,
    use_zero_invalid: bool,
    min_size: int,
    canny1: float,
    canny2: float,
    ransac_iters: int,
    ransac_thresh: float,
    ransac_min_inliers: int,
) -> np.ndarray:
    img8 = normalize_to_uint8(depth, use_zero_invalid)
    img_blur = cv2.GaussianBlur(img8, (5, 5), 0)
    edges = cv2.Canny(img_blur, threshold1=canny1, threshold2=canny2)

    ys, xs = np.where(edges > 0)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if points.shape[0] < 3:
        return np.zeros_like(depth, dtype=bool)

    rng = np.random.default_rng()
    best_inliers = None
    best_model = None

    for _ in range(max(1, ransac_iters)):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        c = fit_circle_3pts(points[idx[0]], points[idx[1]], points[idx[2]])
        if c is None:
            continue
        cx, cy, r = c
        # Compute distances to circle
        d = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        residual = np.abs(d - r)
        inliers = residual <= ransac_thresh
        num_inliers = int(np.count_nonzero(inliers))
        if best_inliers is None or num_inliers > int(np.count_nonzero(best_inliers)):
            best_inliers = inliers
            best_model = (cx, cy, r)

    if best_inliers is None or int(np.count_nonzero(best_inliers)) < max(1, ransac_min_inliers):
        return np.zeros_like(depth, dtype=bool)

    cx, cy, r = best_model
    h, w = depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, thickness=-1)

    valid_mask = depth > 0 if use_zero_invalid else ~np.isnan(depth)
    interior = (mask > 0) & valid_mask
    if interior.sum() < max(1, min_size):
        return np.zeros_like(depth, dtype=bool)
    return interior


def main() -> None:
    args = parse_args()
    inp = args.npy
    if not os.path.isfile(inp):
        raise FileNotFoundError(f"File not found: {inp}")

    depth = np.load(inp).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

    # Choose method: rim (Hough circle) preferred, fallback to min-depth method when needed
    if args.method == "rim":
        mask = segment_by_rim(
            depth,
            use_zero_invalid=args.invalid_zero,
            min_size=args.min_size,
            canny1=args.canny1,
            canny2=args.canny2,
            dp=args.hough_dp,
            min_dist=args.hough_min_dist,
            param1=args.hough_param1,
            param2=args.hough_param2,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )
        if not np.any(mask):
            thr = compute_threshold_from_min(depth, args.invalid_zero, args.percentile_low, args.delta_mm)
            mask = segment_interior(depth, thr, args.invalid_zero, args.min_size)
    elif args.method == "ransac":
        mask = segment_by_ransac(
            depth,
            use_zero_invalid=args.invalid_zero,
            min_size=args.min_size,
            canny1=args.canny1,
            canny2=args.canny2,
            ransac_iters=args.ransac_iters,
            ransac_thresh=args.ransac_thresh,
            ransac_min_inliers=args.ransac_min_inliers,
        )
        if not np.any(mask):
            # fallback to rim, then min
            mask = segment_by_rim(
                depth,
                use_zero_invalid=args.invalid_zero,
                min_size=args.min_size,
                canny1=args.canny1,
                canny2=args.canny2,
                dp=args.hough_dp,
                min_dist=args.hough_min_dist,
                param1=args.hough_param1,
                param2=args.hough_param2,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
            )
            if not np.any(mask):
                thr = compute_threshold_from_min(depth, args.invalid_zero, args.percentile_low, args.delta_mm)
                mask = segment_interior(depth, thr, args.invalid_zero, args.min_size)
    else:
        thr = compute_threshold_from_min(depth, args.invalid_zero, args.percentile_low, args.delta_mm)
        mask = segment_interior(depth, thr, args.invalid_zero, args.min_size)

    # Apply mask: keep interior, set outside to 0
    result = np.zeros_like(depth, dtype=np.float32)
    result[mask] = depth[mask]

    # Determine output path
    if args.out is not None:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(inp))[0]
        input_dir = os.path.dirname(inp)
        if not input_dir:
            # When relative filename is given from current dir (e.g., "file.npy"), dirname is "".
            # Use absolute path's directory instead.
            input_dir = os.path.dirname(os.path.abspath(inp))
        os.makedirs(input_dir, exist_ok=True)
        out_path = os.path.join(input_dir, f"{base}_cup.npy")

    np.save(out_path, result)
    print(f"Saved segmented depth to: {out_path}")


if __name__ == "__main__":
    main()


