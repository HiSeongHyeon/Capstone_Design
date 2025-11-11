
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def _fit_plane_svd(points: np.ndarray):
    centroid = points.mean(axis=0)
    Q = points - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / np.linalg.norm(n)
    d = -np.dot(n, centroid)
    return n, d

def _ransac_plane(points: np.ndarray, n_iter: int = 400, inlier_thresh: float = 1.0):
    N = points.shape[0]
    if N < 3:
        raise ValueError("평면 피팅에 필요한 점이 부족합니다.")
    best_inliers = None
    best_count = -1
    rng = np.random.default_rng(42)
    idx = np.arange(N)
    for _ in range(n_iter):
        sample = rng.choice(idx, size=3, replace=False)
        p0, p1, p2 = points[sample]
        v1 = p1 - p0
        v2 = p2 - p0
        cp = np.cross(v1, v2)
        n_norm = np.linalg.norm(cp)
        if n_norm < 1e-6:
            continue
        n = cp / n_norm
        d = -np.dot(n, p0)
        dist = np.abs(points @ n + d)
        inliers = dist < inlier_thresh
        count = np.count_nonzero(inliers)
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 10:
        n, d = _fit_plane_svd(points)
        inliers = np.ones(N, dtype=bool)
    else:
        n, d = _fit_plane_svd(points[best_inliers])
        inliers = best_inliers
    s = points @ n + d
    if np.median(s) > 0:
        n = -n
        d = -d
    return n, d, inliers

def _fit_circle_kasa(uv: np.ndarray):
    x = uv[:, 0]; y = uv[:, 1]
    A = np.stack([2*x, 2*y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol
    center = np.array([a, b_])
    r = np.sqrt(max(1e-9, c + a*a + b_*b_))
    return center, r

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray

class CupVolumeEstimator:
    def __init__(self, depth_mm: np.ndarray, intr: Intrinsics):
        self.depth = depth_mm.astype(np.float64)
        self.H, self.W = self.depth.shape
        self.valid = self.depth > 0
        self.intr = intr
        # 픽셀 중심 undistort
        u, v = np.meshgrid(np.arange(self.W, dtype=np.float32),
                           np.arange(self.H, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
        und = cv.undistortPoints(
            pts,
            cameraMatrix=np.array([[intr.fx, 0, intr.cx],
                                   [0, intr.fy, intr.cy],
                                   [0,       0,       1]], dtype=np.float64),
            distCoeffs=intr.dist.reshape(-1, 1),
            P=None
        ).reshape(-1, 2)
        self.xn = und[:, 0].reshape(self.H, self.W)
        self.yn = und[:, 1].reshape(self.H, self.W)
        # 픽셀 코너 undistort
        u_edges = np.arange(self.W + 1, dtype=np.float32) - 0.5
        v_edges = np.arange(self.H + 1, dtype=np.float32) - 0.5
        uu, vv = np.meshgrid(u_edges, v_edges, indexing='xy')
        epts = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
        eund = cv.undistortPoints(
            epts,
            cameraMatrix=np.array([[intr.fx, 0, intr.cx],
                                   [0, intr.fy, intr.cy],
                                   [0,       0,       1]], dtype=np.float64),
            distCoeffs=intr.dist.reshape(-1, 1),
            P=None
        ).reshape(-1, 2)
        self.xn_edges = eund[:, 0].reshape(self.H + 1, self.W + 1)
        self.yn_edges = eund[:, 1].reshape(self.H + 1, self.W + 1)

    def _points3d(self):
        Z = self.depth[self.valid]
        X = self.xn[self.valid] * Z
        Y = self.yn[self.valid] * Z
        return np.stack([X, Y, Z], axis=1)

    def estimate_rim_plane(self):
        # 경계 기반 림 후보
        mask = self.valid.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv.dilate(mask, kernel, iterations=1)
        ero = cv.erode(mask,  kernel, iterations=1)
        boundary = (dil > 0) & (ero == 0) & (mask > 0)
        idx = np.flatnonzero(boundary)
        if idx.size < 200:
            Z = self.depth[self.valid]
            q = np.percentile(Z, 3.0)
            close_mask = np.zeros_like(self.valid, dtype=bool)
            close_mask[self.valid] = Z <= q
            idx = np.flatnonzero(close_mask)
        P_all = self._points3d()
        flat_valid = np.flatnonzero(self.valid.ravel())
        order = np.argsort(flat_valid)
        pos = order[np.searchsorted(flat_valid, idx, sorter=order)]
        rim_points = P_all[pos]
        n, d, inliers_local = _ransac_plane(rim_points, n_iter=400, inlier_thresh=1.0)
        global_inliers = np.zeros(P_all.shape[0], dtype=bool)
        global_inliers[pos[inliers_local]] = True
        return n, d, global_inliers

    def _per_pixel_plane_area(self, n, d):
        H, W = self.H, self.W
        r00 = np.stack([self.xn_edges[:-1, :-1], self.yn_edges[:-1, :-1], np.ones((H, W))], axis=2)
        r10 = np.stack([self.xn_edges[:-1, 1:],  self.yn_edges[:-1, 1:],  np.ones((H, W))], axis=2)
        r01 = np.stack([self.xn_edges[1:,  :-1], self.yn_edges[1:,  :-1], np.ones((H, W))], axis=2)
        r11 = np.stack([self.xn_edges[1:,  1:],  self.yn_edges[1:,  1:],  np.ones((H, W))], axis=2)
        eps = 1e-12
        t00 = -d / (r00 @ n + eps)
        t10 = -d / (r10 @ n + eps)
        t01 = -d / (r01 @ n + eps)
        t11 = -d / (r11 @ n + eps)
        valid = (t00 > 0) & (t10 > 0) & (t01 > 0) & (t11 > 0)
        q00 = r00 * t00[..., None]
        q10 = r10 * t10[..., None]
        q01 = r01 * t01[..., None]
        q11 = r11 * t11[..., None]
        tri1 = np.cross(q10 - q00, q01 - q00)
        tri2 = np.cross(q11 - q01, q10 - q01)
        A = 0.5 * (np.linalg.norm(tri1, axis=2) + np.linalg.norm(tri2, axis=2))
        A[~valid] = 0.0
        return A

    def _plane_basis(self, n, d):
        p0 = -d * n
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(tmp @ n) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ n) * n; e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        return p0, e1, e2

    def estimate_pixel_integration(self, known_diameter_mm: Optional[float] = None):
        n, d, _ = self.estimate_rim_plane()
        Xc = self.xn * self.depth
        Yc = self.yn * self.depth
        Zc = self.depth
        s = Xc * n[0] + Yc * n[1] + Zc * n[2] + d
        h_center = np.zeros_like(Zc, dtype=np.float64)
        h_center[self.valid] = np.maximum(0.0, -s[self.valid])
        A_plane = self._per_pixel_plane_area(n, d)
        volume_mm3 = float(np.sum(A_plane * h_center))
        footprint_area = float(np.sum(A_plane * self.valid))
        diameter_from_area = 2.0 * np.sqrt(footprint_area / np.pi) if footprint_area > 0 else 0.0
        # circle-fit-based diameter (보조)
        p0, e1, e2 = self._plane_basis(n, d)
        P = np.stack([Xc, Yc, Zc], axis=2)
        hmap = np.zeros_like(Zc, dtype=np.float64); hmap[:] = np.nan
        hmap[self.valid] = h_center[self.valid]
        near = np.isfinite(hmap) & (hmap < 0.8)
        if np.count_nonzero(near) < 50:
            near = np.isfinite(hmap) & (hmap < 1.5)
        near_pts = P[near].reshape(-1, 3)
        if near_pts.shape[0] >= 10:
            uv = np.stack([(near_pts - p0) @ e1, (near_pts - p0) @ e2], axis=1)
            c_uv, r_uv = _fit_circle_kasa(uv)
            diameter_circle = 2.0 * float(r_uv)
            center_uv = c_uv
        else:
            diameter_circle = diameter_from_area
            # 중심은 발자국 면적 중심 근사 (유효 픽셀 평균)
            valid_3d = P[self.valid].reshape(-1,3)
            uv_all = np.stack([(valid_3d - p0) @ e1, (valid_3d - p0) @ e2], axis=1)
            center_uv = uv_all.mean(axis=0) if uv_all.shape[0]>0 else np.array([0.0,0.0])
        result = {
            "n": n, "d": d, "h_center": h_center, "A_plane": A_plane,
            "volume_mm3": volume_mm3,
            "diam_from_area_mm": diameter_from_area,
            "diam_from_circle_mm": float(diameter_circle),
            "plane_basis": (p0, e1, e2),
            "center_uv": center_uv
        }
        if known_diameter_mm is not None and diameter_from_area > 1e-6:
            sratio = known_diameter_mm / diameter_from_area
            result["volume_scaled_mm3"] = volume_mm3 * (sratio ** 3)
            result["scale_ratio"] = sratio
        return result

    def estimate_radial(self, base: Dict, nbins: int = 80, smooth: bool = False, window: int = 11, poly: int = 2):
        # 기반 결과(평면, 면적, 높이, 중심 등)
        n = base["n"]; d = base["d"]
        h_center = base["h_center"]; A_plane = base["A_plane"]
        p0, e1, e2 = base["plane_basis"]
        center_uv = base["center_uv"]
        # plane coords of pixel centers
        Xc = self.xn * self.depth
        Yc = self.yn * self.depth
        Zc = self.depth
        P = np.stack([Xc, Yc, Zc], axis=2)
        U = (P - p0) @ e1
        V = (P - p0) @ e2
        R = np.sqrt((U - center_uv[0])**2 + (V - center_uv[1])**2)
        # rim radius from footprint area
        footprint_area = float(np.sum(A_plane * self.valid))
        R_area = np.sqrt(footprint_area / np.pi) if footprint_area>0 else 0.0
        # bins
        r_edges = np.linspace(0.0, R_area, nbins+1)
        r_mid = 0.5*(r_edges[:-1] + r_edges[1:])
        ring_area = np.zeros(nbins, dtype=np.float64)
        ring_havg = np.zeros(nbins, dtype=np.float64)
        ring_vol = np.zeros(nbins, dtype=np.float64)
        for k in range(nbins):
            mask = self.valid & (R >= r_edges[k]) & (R < r_edges[k+1])
            A_k = np.sum(A_plane[mask])
            ring_area[k] = A_k
            if A_k > 0:
                h_mean = np.sum(h_center[mask] * A_plane[mask]) / A_k
                ring_havg[k] = h_mean
                ring_vol[k] = h_mean * A_k
            else:
                ring_havg[k] = 0.0
                ring_vol[k] = 0.0
        V_exact = float(np.sum(ring_vol))

        # optional smoothing (단순 이동평균)
        if smooth and nbins >= 5 and np.count_nonzero(ring_area) > 0:
            w = max(3, min(window, (nbins//2)*2+1))  # 홀수
            pad = w//2
            # 면적 가중 이동평균
            pad_h = np.pad(ring_havg*ring_area, (pad,pad), mode='edge')
            pad_a = np.pad(ring_area, (pad,pad), mode='edge')
            num = np.convolve(pad_h, np.ones(w), mode='valid')
            den = np.convolve(pad_a, np.ones(w), mode='valid')
            h_smooth = np.divide(num, den, out=np.zeros_like(num), where=den>0)
            V_smooth = float(np.sum(h_smooth * ring_area))
        else:
            h_smooth = ring_havg.copy()
            V_smooth = V_exact

        return {
            "r_edges": r_edges, "r_mid": r_mid,
            "ring_area": ring_area,
            "ring_havg": ring_havg,
            "ring_havg_smooth": h_smooth,
            "ring_volume": ring_vol,
            "volume_mm3_radial": V_exact,
            "volume_mm3_radial_smooth": V_smooth,
            "R_area": R_area
        }

    # ---- 플롯: 각 Figure 하나의 차트만 생성 (색상 지정 없음) ----
    def plot_report(self, base: Dict, radial: Dict, prefix: str = "cup"):
        import matplotlib.pyplot as plt
        n = base["n"]; d = base["d"]
        h_center = base["h_center"]; A_plane = base["A_plane"]
        p0, e1, e2 = base["plane_basis"]
        center_uv = base["center_uv"]
        # plane coords
        Xc = self.xn * self.depth
        Yc = self.yn * self.depth
        Zc = self.depth
        P = np.stack([Xc, Yc, Zc], axis=2)
        U = (P - p0) @ e1
        V = (P - p0) @ e2
        R = np.sqrt((U - center_uv[0])**2 + (V - center_uv[1])**2)

        # 1) 3D scatter (다운샘플)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        step = max(1, int(np.sqrt((self.H*self.W)/6000)))
        sel = np.zeros_like(self.valid)
        sel[::step, ::step] = True
        m = self.valid & sel
        ax.scatter((Xc[m]), (Yc[m]), (Zc[m]), s=1)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        fig.savefig(f"/mnt/data/{prefix}_3D_scatter.png", dpi=150); plt.close(fig)

        # 2) Top view (U,V) with rim circle
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(U[self.valid], V[self.valid], s=1)
        from matplotlib.patches import Circle
        R_area = radial["R_area"]
        circ = Circle((center_uv[0], center_uv[1]), R_area, fill=False)
        ax.add_patch(circ)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('U (mm)'); ax.set_ylabel('V (mm)')
        fig.savefig(f"/mnt/data/{prefix}_top_plane.png", dpi=150); plt.close(fig)

        # 3) r–h scatter
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(R[self.valid], h_center[self.valid], s=1)
        ax.set_xlabel('r (mm)'); ax.set_ylabel('height h (mm)')
        fig.savefig(f"/mnt/data/{prefix}_r_vs_h.png", dpi=150); plt.close(fig)

        # 4) height map (픽셀 격자상)
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(h_center, origin='upper')
        ax.set_title('Height map (mm)')
        fig.colorbar(im, ax=ax)
        fig.savefig(f"/mnt/data/{prefix}_height_map.png", dpi=150); plt.close(fig)

        # 5) height histogram
        fig = plt.figure()
        ax = fig.add_subplot()
        vals = h_center[self.valid].ravel()
        ax.hist(vals, bins=50)
        ax.set_xlabel('h (mm)'); ax.set_ylabel('count')
        fig.savefig(f"/mnt/data/{prefix}_height_hist.png", dpi=150); plt.close(fig)

        # 6) radial mean profile
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(radial["r_mid"], radial["ring_havg"], label="mean h(r)")
        ax.plot(radial["r_mid"], radial["ring_havg_smooth"], label="smoothed h(r)")
        ax.set_xlabel('r (mm)'); ax.set_ylabel('mean height (mm)')
        ax.legend()
        fig.savefig(f"/mnt/data/{prefix}_radial_profile.png", dpi=150); plt.close(fig)

        # 7) ring area
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(radial["r_mid"], radial["ring_area"])
        ax.set_xlabel('r (mm)'); ax.set_ylabel('ring area on plane (mm^2)')
        fig.savefig(f"/mnt/data/{prefix}_ring_area.png", dpi=150); plt.close(fig)

        # 8) ring volume
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(radial["r_mid"], radial["ring_volume"])
        ax.set_xlabel('r (mm)'); ax.set_ylabel('ring volume (mm^3)')
        fig.savefig(f"/mnt/data/{prefix}_ring_volume.png", dpi=150); plt.close(fig)

    def save_radial_profile_csv(self, radial: Dict, path: str):
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['r_left_mm','r_right_mm','r_mid_mm','ring_area_mm2','ring_mean_h_mm','ring_mean_h_smooth_mm','ring_volume_mm3'])
            for k in range(len(radial["r_mid"])):
                w.writerow([radial["r_edges"][k], radial["r_edges"][k+1], radial["r_mid"][k],
                            radial["ring_area"][k], radial["ring_havg"][k], radial["ring_havg_smooth"][k],
                            radial["ring_volume"][k]])

    def estimate_all(self, known_diameter_mm: Optional[float] = None, radial_bins: int = 80, smooth_radial: bool = True):
        base = self.estimate_pixel_integration(known_diameter_mm=known_diameter_mm)
        radial = self.estimate_radial(base, nbins=radial_bins, smooth=smooth_radial, window=11, poly=2)
        out = {
            "pixel_volume_mm3": base["volume_mm3"],
            "pixel_volume_mL": base["volume_mm3"]/1000.0,
            "pixel_volume_scaled_mm3": base.get("volume_scaled_mm3", None),
            "pixel_volume_scaled_mL": (base.get("volume_scaled_mm3")/1000.0) if base.get("volume_scaled_mm3") is not None else None,
            "diameter_from_area_mm": base["diam_from_area_mm"],
            "diameter_from_circle_mm": base["diam_from_circle_mm"],
            "radial_volume_mm3": radial["volume_mm3_radial"],
            "radial_volume_mL": radial["volume_mm3_radial"]/1000.0,
            "radial_volume_smooth_mm3": radial["volume_mm3_radial_smooth"],
            "radial_volume_smooth_mL": radial["volume_mm3_radial_smooth"]/1000.0,
            "R_area_mm": radial["R_area"],
            "plane_n": base["n"],
            "plane_d": base["d"],
            "h_center": base["h_center"],
            "A_plane": base["A_plane"],
            "plane_basis": base["plane_basis"],
            "center_uv": base["center_uv"],
            "radial": radial
        }
        return out

# ---------------------------- CLI Example ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cup volume estimator from depth map (mm)")
    parser.add_argument("--npy", type=str, required=True, help="Depth .npy path (mm), masked to cup interior")
    parser.add_argument("--fx", type=float, default=190.46873334)
    parser.add_argument("--fy", type=float, default=191.201416)
    parser.add_argument("--cx", type=float, default=120.00074471)
    parser.add_argument("--cy", type=float, default=90.00851171)
    parser.add_argument("--dist", type=float, nargs="+",
                        default=[4.93083651e-01, -1.25632226e+00, 2.28374174e-03, -1.58899167e-05, 5.29625368e-01],
                        help="k1 k2 p1 p2 k3")
    parser.add_argument("--known_diam", type=float, default=None, help="Known rim diameter in mm (e.g., 90)")
    args = parser.parse_args()

    depth = np.load(args.npy)
    intr = Intrinsics(args.fx, args.fy, args.cx, args.cy, np.array(args.dist, dtype=np.float64))
    est = CupVolumeEstimator(depth, intr)
    out = est.estimate(known_diameter_mm=args.known_diam)

    # Pretty print
    def fmt(k, v): 
        return f"{k:>32s}: {v:.6f}" if isinstance(v, float) else f"{k:>32s}: {v}"
    print("\n--- Cup Volume Estimation ---")
    for k in ["volume_mm3", "volume_mL", "volume_scaled_mm3", "volume_scaled_mL",
              "measured_diameter_from_area_mm", "measured_diameter_from_circle_mm",
              "max_height_mm", "plane_nx", "plane_ny", "plane_nz", "plane_d"]:
        if k in out:
            print(fmt(k, out[k]))
