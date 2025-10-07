import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np


def choose_colormap(name: str) -> int:
    name_upper = name.upper()
    maps = {
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "BONE": cv2.COLORMAP_BONE,
        "JET": cv2.COLORMAP_JET,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "SPRING": cv2.COLORMAP_SPRING,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "HOT": cv2.COLORMAP_HOT,
        "PARULA": getattr(cv2, "COLORMAP_PARULA", cv2.COLORMAP_JET),
        # Prefer TURBO if available in the installed OpenCV
        "TURBO": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "VIRIDIS": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "PLASMA": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
        "MAGMA": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "INFERNO": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "CIVIDIS": getattr(cv2, "COLORMAP_CIVIDIS", cv2.COLORMAP_JET),
    }
    if name_upper not in maps:
        raise ValueError(f"Unknown colormap: {name}. Try one of: {', '.join(sorted(maps.keys()))}")
    return maps[name_upper]


def compute_norm_range(
    depth_mm: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    use_positive_only: bool,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[float, float]:
    data = depth_mm
    if use_positive_only:
        data = data[data > 0]
    if data.size == 0:
        return 0.0, 1.0

    if vmin is None:
        vmin = float(np.percentile(data, percentile_low))
    if vmax is None:
        vmax = float(np.percentile(data, percentile_high))

    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def depth_to_heatmap(
    depth_mm: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: int,
    clip: bool = True,
) -> np.ndarray:
    arr = depth_mm.astype(np.float32)
    if clip:
        arr = np.clip(arr, vmin, vmax)

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    norm = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap)
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize depth .npy as heatmap (OpenCV imshow)")
    parser.add_argument("npy", help="Path to depth .npy file (units assumed: mm)")
    parser.add_argument("--min", dest="vmin", type=float, default=None, help="Min depth (mm) for normalization")
    parser.add_argument("--max", dest="vmax", type=float, default=None, help="Max depth (mm) for normalization")
    parser.add_argument("--colormap", default="TURBO", help="OpenCV colormap name (e.g., TURBO, JET, RAINBOW)")
    parser.add_argument("--invalid-zero", action="store_true", help="Treat zeros as invalid (render as black)")
    parser.add_argument("--no-gui", action="store_true", help="Do not open window; use with --save")
    parser.add_argument("--save", default=None, help="Optional path to save the heatmap image (e.g., output.png)")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), default=None, help="Resize output to WxH")
    parser.add_argument("--percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(1.0, 99.0), help="Percentiles for auto range when --min/--max omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.npy):
        raise FileNotFoundError(f"File not found: {args.npy}")

    depth = np.load(args.npy)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

    depth = depth.astype(np.float32)

    if args.invalid_zero:
        mask_invalid = depth <= 0
    else:
        mask_invalid = np.isnan(depth)

    vmin, vmax = compute_norm_range(
        depth_mm=depth,
        vmin=args.vmin,
        vmax=args.vmax,
        use_positive_only=True,
        percentile_low=args.percentiles[0],
        percentile_high=args.percentiles[1],
    )

    cmap = choose_colormap(args.colormap)
    heatmap = depth_to_heatmap(depth, vmin, vmax, cmap, clip=True)

    # Render invalid pixels as black
    if mask_invalid.any():
        heatmap[mask_invalid] = (0, 0, 0)

    if args.resize is not None:
        w, h = args.resize
        if w > 0 and h > 0:
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)

    if args.save:
        ok = cv2.imwrite(args.save, heatmap)
        if not ok:
            raise RuntimeError(f"Failed to save image to: {args.save}")

    if not args.no_gui:
        window = "depth_heatmap"
        cv2.imshow(window, heatmap)
        cv2.waitKey(0)
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()


