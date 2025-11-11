
"""
Cup Volume Estimator (Depth-only, robust & tilt-invariant)
----------------------------------------------------------
핵심 아이디어
1) 픽셀 좌표를 왜곡 보정(undistort) 후 정규화 좌표계로 변환
2) 깊이 Z(mm)와 정규화 좌표(xn, yn)를 이용해 3D 포인트 (X, Y, Z) 생성
3) 컵 림 평면을 RANSAC으로 추정 (마스크 경계 → 림 후보)
4) 각 픽셀의 4개 코너 광선과 림 평면의 교점을 구해, 픽셀의 '림 평면 위 면적'을 계산
5) 픽셀 중심점의 '림 평면으로부터의 높이'를 구해, 부피 = Σ(면적 × 높이)

장점
- 카메라가 기울어져 있어도 정확 (림 평면 기준 적분)
- 중복 투영/이중 적분 문제 방지(픽셀→평면 1:1 매핑)
- 왜곡 보정 포함
- 지름(90 mm 등)으로 전체 스케일을 교정 가능
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import cv2 as cv

# ---------------------------- Linear Algebra Helpers ----------------------------

def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """평면 n^T x + d = 0 (||n||=1)을 SVD로 피팅"""
    centroid = points.mean(axis=0)
    Q = points - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / np.linalg.norm(n)
    d = -np.dot(n, centroid)
    return n, d

def _ransac_plane(points: np.ndarray, n_iter: int = 400, inlier_thresh: float = 1.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """RANSAC으로 강건한 평면 피팅"""
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

    # 컵 내부 방향으로 높이가 양수가 되도록 법선 방향 통일
    s = points @ n + d
    if np.median(s) > 0:
        n = -n
        d = -d
    return n, d, inliers

def _fit_circle_kasa(uv: np.ndarray) -> Tuple[np.ndarray, float]:
    """2D 원의 Kasa 피팅 (x^2 + y^2 + ax + by + c = 0)"""
    x = uv[:, 0]
    y = uv[:, 1]
    A = np.stack([2*x, 2*y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol
    center = np.array([a, b_])
    r = np.sqrt(max(1e-9, c + a*a + b_*b_))
    return center, r

# ---------------------------- Core Estimator ----------------------------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # (k1, k2, p1, p2, k3)

class CupVolumeEstimator:
    def __init__(self, depth_mm: np.ndarray, intr: Intrinsics):
        """
        depth_mm: (H, W) 실수 배열, 단위 mm. 유효 픽셀은 > 0
        intr: 카메라 내부 파라미터 및 왜곡계수
        """
        self.depth = depth_mm.astype(np.float64)
        self.H, self.W = self.depth.shape
        self.valid = self.depth > 0
        self.intr = intr

        # 픽셀 중심의 undistorted 정규화 좌표 (xn, yn)
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

        # 픽셀 코너의 undistorted 정규화 좌표 (면적 계산용)
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

    def _points3d(self) -> np.ndarray:
        """유효 픽셀들의 (X,Y,Z) Nx3"""
        Z = self.depth[self.valid]
        X = self.xn[self.valid] * Z
        Y = self.yn[self.valid] * Z
        P = np.stack([X, Y, Z], axis=1)
        return P

    def _rim_candidates(self) -> np.ndarray:
        """림 후보: 마스크 경계(1차) + 얕은 깊이 상위 퍼센타일(폴백)"""
        mask = self.valid.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv.dilate(mask, kernel, iterations=1)
        ero = cv.erode(mask,  kernel, iterations=1)
        boundary = (dil > 0) & (ero == 0) & (mask > 0)
        idx = np.flatnonzero(boundary)
        if idx.size < 200:
            Z = self.depth[self.valid]
            q = np.percentile(Z, 3.0)  # 상위 3% (카메라에 가장 가까움)
            close_mask = np.zeros_like(self.valid, dtype=bool)
            close_mask[self.valid] = Z <= q
            idx = np.flatnonzero(close_mask)
        return idx

    def estimate_rim_plane(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """림 평면 추정 (n, d, 전역 유효점 기준 inlier 마스크)"""
        idx = self._rim_candidates()
        P_all = self._points3d()
        flat_valid = np.flatnonzero(self.valid.ravel())
        order = np.argsort(flat_valid)
        pos = order[np.searchsorted(flat_valid, idx, sorter=order)]
        rim_points = P_all[pos]
        n, d, inliers_local = _ransac_plane(rim_points, n_iter=400, inlier_thresh=1.0)
        global_inliers = np.zeros(P_all.shape[0], dtype=bool)
        global_inliers[pos[inliers_local]] = True
        return n, d, global_inliers

    def _per_pixel_plane_area(self, n: np.ndarray, d: float) -> np.ndarray:
        """
        각 픽셀 중심을 둘러싼 4개 코너 광선과 평면의 교점으로 만들어지는
        사각형을 두 개의 삼각형으로 분할해 면적을 계산.
        """
        H, W = self.H, self.W
        # Ray directions for corners
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

    def estimate(self, known_diameter_mm: Optional[float] = None) -> Dict[str, float]:
        """
        known_diameter_mm가 주어지면, 측정된 림 지름과 비교하여 부피에 균일 스케일 보정(s^3)을 적용.
        """
        n, d, rim_inliers = self.estimate_rim_plane()

        # 중심점 3D와 높이(림 평면 기준, 내부방향 양수)
        Xc = self.xn * self.depth
        Yc = self.yn * self.depth
        Zc = self.depth
        s = Xc * n[0] + Yc * n[1] + Zc * n[2] + d
        h_center = np.zeros_like(Zc, dtype=np.float64)
        h_center[self.valid] = np.maximum(0.0, -s[self.valid])

        # 픽셀당 평면 위 면적
        A_plane = self._per_pixel_plane_area(n, d)

        # 부피 적분
        volume_mm3 = float(np.sum(A_plane * h_center))

        # 평면 위 발자국 면적과 등가 원지름
        footprint_area = float(np.sum(A_plane * self.valid))
        diameter_from_area = 2.0 * np.sqrt(footprint_area / np.pi) if footprint_area > 0 else 0.0

        # 림에 매우 가까운 점들로 지름(보조 측정)
        P = np.stack([Xc, Yc, Zc], axis=2)
        hmap = np.zeros_like(Zc, dtype=np.float64); hmap[:] = np.nan
        hmap[self.valid] = h_center[self.valid]
        near = np.isfinite(hmap) & (hmap < 0.8)
        if np.count_nonzero(near) < 50:
            near = np.isfinite(hmap) & (hmap < 1.5)
        near_pts = P[near].reshape(-1, 3)
        # 평면 좌표계로 투영 (원 피팅용)
        p0 = -d * n
        # 기저 벡터 생성
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(tmp @ n) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ n) * n; e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        uv = np.stack([(near_pts - p0) @ e1, (near_pts - p0) @ e2], axis=1)
        if uv.shape[0] >= 10:
            c_uv, r_uv = _fit_circle_kasa(uv)
            diameter_circle = 2.0 * float(r_uv)
        else:
            diameter_circle = diameter_from_area  # fallback

        result = {
            "volume_mm3": volume_mm3,
            "volume_mL": volume_mm3 / 1000.0,
            "measured_diameter_from_area_mm": diameter_from_area,
            "measured_diameter_from_circle_mm": diameter_circle,
            "max_height_mm": float(np.nanmax(h_center)),
            "plane_nx": float(n[0]),
            "plane_ny": float(n[1]),
            "plane_nz": float(n[2]),
            "plane_d": float(d),
        }

        if known_diameter_mm is not None and diameter_from_area > 1e-6:
            sratio = known_diameter_mm / diameter_from_area
            vol_scaled = volume_mm3 * (sratio ** 3)
            result.update({
                "scale_ratio_by_diameter": sratio,
                "volume_scaled_mm3": vol_scaled,
                "volume_scaled_mL": vol_scaled / 1000.0,
            })

        return result

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
