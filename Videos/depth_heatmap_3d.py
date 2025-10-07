import argparse
import os
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize depth .npy as 3D surface (matplotlib)")
    parser.add_argument("npy", help="Path to depth .npy file (units assumed: mm)")
    parser.add_argument("--min", dest="vmin", type=float, default=None, help="Min depth (mm) for normalization and color scale")
    parser.add_argument("--max", dest="vmax", type=float, default=None, help="Max depth (mm) for normalization and color scale")
    parser.add_argument("--colormap", default="turbo", help="Matplotlib colormap name (e.g., turbo, viridis, plasma, magma)")
    parser.add_argument("--invalid-zero", action="store_true", help="Treat zeros as invalid (masked)")
    parser.add_argument("--percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(1.0, 99.0), help="Percentiles for auto range when --min/--max omitted")
    parser.add_argument("--downsample", nargs=2, type=int, metavar=("SX", "SY"), default=(1, 1), help="Stride sampling along X and Y (e.g., 2 2)")
    parser.add_argument("--no-gui", action="store_true", help="Do not show window; useful with --save")
    parser.add_argument("--save", default=None, help="Optional path to save figure (e.g., output.png)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    return parser.parse_args()


def compute_norm_range(depth_mm: np.ndarray, vmin: Optional[float], vmax: Optional[float], use_positive_only: bool, percentile_low: float, percentile_high: float) -> Tuple[float, float]:
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


def main() -> None:
    # Use non-interactive backend if --no-gui is passed
    args = parse_args()
    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

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

    sy = max(1, int(args.downsample[1]))
    sx = max(1, int(args.downsample[0]))
    depth_ds = depth[::sy, ::sx]
    mask_ds = mask_invalid[::sy, ::sx]

    vmin, vmax = compute_norm_range(
        depth_mm=depth_ds,
        vmin=args.vmin,
        vmax=args.vmax,
        use_positive_only=True,
        percentile_low=args.percentiles[0],
        percentile_high=args.percentiles[1],
    )

    h, w = depth_ds.shape
    y = np.arange(h)
    x = np.arange(w)
    X, Y = np.meshgrid(x, y)

    Z = np.ma.array(depth_ds, mask=mask_ds)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=args.colormap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Depth (mm)')
    fig.colorbar(surf, shrink=0.6, aspect=12, label='Depth (mm)')
    ax.set_title('Depth Surface')

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches='tight')

    if not args.no_gui:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()


