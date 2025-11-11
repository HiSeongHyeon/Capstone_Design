"""
Cup Volume Estimator (Depth-only, robust & tilt-invariant)
----------------------------------------------------------
핵심 아이디어 (알려진 지름으로 항상 스케일 보정) + 3D 시각화

1) 픽셀 좌표를 왜곡 보정(undistort) 후 정규화 좌표계로 변환
2) 깊이 Z(mm)와 정규화 좌표(xn, yn)를 이용해 3D 포인트 (X, Y, Z) 생성
3) 컵 림 평면을 RANSAC으로 추정 (마스크 경계 → 림 후보)
4) 각 픽셀의 '림 평면 위 면적'을 계산 (A_plane)
5) 픽셀 중심점의 '림 평면으로부터의 높이'를 계산 (h_center)
6) 원본 부피 = Σ(A_plane × h_center)
7) 림 평면의 총면적(Σ A_plane)으로 '측정된 지름'을 계산
8) (실제 지름 / 측정된 지름)^3 비율을 원본 부피에 곱하여 스케일 보정
9) (옵션) --plot 플래그 시, 3D 포인트 클라우드 및 측정 결과 시각화
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import cv2 as cv
import argparse # Argparse import 추가

# ---------------------------- Matplotlib (Plotting) ---------------------------
# 시각화를 위한 라이브러리 추가
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 한글 폰트 설정
try:
    # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
except RuntimeError:
    try:
        # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    except RuntimeError:
        # Linux (나눔폰트 등 설치 필요)
        plt.rcParams['font.family'] = 'NanumGothic'
        
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
# -----------------------------------------------------------------------------

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

    def _plot_results(self, x_mm_orig: np.ndarray, y_mm_orig: np.ndarray, z_mm_orig: np.ndarray, 
                      z_values_hist: np.ndarray, height_mm: float, valid_mask: np.ndarray, 
                      result_dict: Dict[str, float], method_name: str):
        """
        [추가된 함수]
        요청한 3D 시각화 로직을 수행합니다.
        데이터는 CupVolumeEstimator의 계산 결과를 사용합니다.
        """
        print("\n시각화 생성 중...")
        
        diameter_mm = result_dict["measured_diameter_from_area_mm"]
        
        # 플롯을 위해 2D X-Y 평면 기준으로 컵 중심을 찾습니다.
        # (이는 시각화용이며, 부피 계산에는 영향을 주지 않습니다)
        points_2d = np.column_stack((x_mm_orig, y_mm_orig)).astype(np.float32)
        center_3d = (0.0, 0.0)
        if points_2d.shape[0] >= 3:
            (cx, cy), _ = cv.minEnclosingCircle(points_2d)
            center_3d = (cx, cy)
        
        # 플롯 데이터를 2D 중심으로 이동
        x_mm = x_mm_orig - center_3d[0]
        y_mm = y_mm_orig - center_3d[1]
        z_mm = z_mm_orig
        
        # 서브플롯 생성
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(f"컵 부피 측정 결과: {method_name}", fontsize=16, y=1.02)
        
        # 1. 3D 플롯
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        # 점이 많으면 느려지므로 20000개로 다운샘플링
        sample_idx = np.random.permutation(len(x_mm))[:20000]
        scatter = ax1.scatter(x_mm[sample_idx], y_mm[sample_idx], z_mm[sample_idx], 
                              c=z_mm[sample_idx], cmap='viridis', s=1)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Depth Z (mm)')
        ax1.set_title(f'3D Point Cloud (카메라 좌표계 기준)')
        plt.colorbar(scatter, ax=ax1, label='Depth (mm)', shrink=0.7)
        ax1.view_init(elev=30, azim=45)
        
        # 축 스케일 동일하게 설정
        try:
            max_range = np.array([x_mm.max()-x_mm.min(), 
                                y_mm.max()-y_mm.min(),
                                z_mm.max()-z_mm.min()]).max() / 2.0
            mid_x = (x_mm.max()+x_mm.min()) * 0.5
            mid_y = (y_mm.max()+y_mm.min()) * 0.5
            mid_z = (z_mm.max()+z_mm.min()) * 0.5
            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        except ValueError:
            pass # 데이터가 없는 경우 스킵
        
        # 2. 상단 뷰 (X-Y 평면)
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(x_mm[sample_idx], y_mm[sample_idx], 
                               c=z_mm[sample_idx], cmap='viridis', s=2)
        
        # 원 그리기 (직경 표시) - RANSAC의 면적 기반 직경 사용
        circle = plt.Circle((0, 0), diameter_mm/2, fill=False, color='red', linewidth=2)
        ax2.add_patch(circle)
        
        # 직경 선 그리기
        ax2.plot([-diameter_mm/2, diameter_mm/2], [0, 0], 'r--', linewidth=2)
        ax2.text(0, -diameter_mm/2 * 1.2, f'측정된 지름: {diameter_mm:.1f} mm', 
                ha='center', fontsize=12, color='red', weight='bold')
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top View (X-Y Plane Projection)')
        ax2.set_aspect('equal')
        ax2.grid(True)
        plt.colorbar(scatter2, ax=ax2, label='Depth (mm)', shrink=0.7)
        
        # 3. 측면 뷰 (X-Z 평면)
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(x_mm[sample_idx], z_mm[sample_idx], 
                               c=z_mm[sample_idx], cmap='viridis', s=2)
        
        # '높이'는 RANSAC 평면 기준이므로 Z축 플롯과 직접 비교 어려움
        # 대신 Z값(깊이)의 범위를 표시
        min_z, max_z = np.min(z_mm), np.max(z_mm)
        ax3.axhline(y=min_z, color='g', linestyle='--', alpha=0.5)
        ax3.axhline(y=max_z, color='g', linestyle='--', alpha=0.5)
        ax3.text(0, (max_z + min_z)/2, f'Z(깊이) 범위: {max_z - min_z:.1f} mm', 
                ha='center', fontsize=12, color='green')
        # RANSAC으로 계산한 컵의 실제 '높이'를 별도 표시
        ax3.text(0, min_z, f'RANSAC 계산 높이: {height_mm:.1f} mm',
                 ha='center', fontsize=12, color='red', weight='bold', va='top')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Depth Z (mm)')
        ax3.set_title('Side View (X-Z Plane Projection)')
        ax3.grid(True)
        plt.colorbar(scatter3, ax=ax3, label='Depth (mm)', shrink=0.7)
        
        # 4. 깊이 히트맵
        ax4 = fig.add_subplot(2, 3, 4)
        depth_display = self.depth.copy()
        depth_display[~valid_mask] = np.nan
        im = ax4.imshow(depth_display, cmap='viridis')
        ax4.set_title('Depth Heatmap (Original)')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax4, label='Depth (mm)', shrink=0.7)
        
        # 5. 깊이 분포 히스토그램
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(z_values_hist, bins=50, color='steelblue', edgecolor='black')
        ax5.axvline(np.mean(z_values_hist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(z_values_hist):.1f} mm')
        ax5.axvline(np.max(z_values_hist), color='green', linestyle='--', linewidth=2, label=f'Max: {np.max(z_values_hist):.1f} mm')
        ax5.set_xlabel('Depth (mm)')
        ax5.set_ylabel('Pixel Count')
        ax5.set_title('Depth Distribution (Raw Z)')
        ax5.legend()
        ax5.grid(True)
        
        # 6. 측정값 요약
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        === {method_name} ===
        
        최종 부피 (보정됨): {result_dict['volume_scaled_mL']:.2f} mL
        측정된 지름 (면적 기준): {result_dict['measured_diameter_from_area_mm']:.2f} mm
        측정된 최대 높이 (평면 기준): {result_dict['max_height_mm']:.2f} mm
        
        스케일 보정 비율: {result_dict['scale_ratio_by_diameter']:.4f}
        
        RANSAC 평면 (n):
         nx: {result_dict['plane_nx']:.3f}
         ny: {result_dict['plane_ny']:.3f}
         nz: {result_dict['plane_nz']:.3f}
        RANSAC 평면 (d): {result_dict['plane_d']:.2f}
        
        ---
        유효 픽셀 수: {len(z_values_hist)}
        평균 깊이 (Raw Z): {np.mean(z_values_hist):.2f} mm
        """
        
        ax6.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
                verticalalignment='top', linespacing=1.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 슈퍼타이틀 공간 확보
        plt.savefig(f'volume_analysis_{method_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def estimate(self, known_diameter_mm: float, do_plot: bool = False) -> Dict[str, float]:
        """
        known_diameter_mm (필수): 컵 림의 실제 지름(mm).
        do_plot (선택): True일 경우 3D 시각화 수행.
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

        # 부피 적분 (보정 전 원본 값)
        volume_mm3_raw = float(np.sum(A_plane * h_center))

        # 평면 위 발자국 면적과 등가 원지름 (스케일 보정용)
        footprint_area = float(np.sum(A_plane * self.valid))
        diameter_from_area = 2.0 * np.sqrt(footprint_area / np.pi) if footprint_area > 0 else 0.0

        # 스케일 보정
        sratio = 0.0
        vol_scaled = volume_mm3_raw
        if diameter_from_area > 1e-6:
            sratio = known_diameter_mm / diameter_from_area
            vol_scaled = volume_mm3_raw * (sratio ** 3)
        
        max_h = 0.0
        if np.any(np.isfinite(h_center)):
             max_h = float(np.nanmax(h_center))

        result = {
            "volume_scaled_mm3": vol_scaled,
            "volume_scaled_mL": vol_scaled / 1000.0,
            "measured_diameter_from_area_mm": diameter_from_area,
            "scale_ratio_by_diameter": sratio,
            "max_height_mm": max_h,
            "plane_nx": float(n[0]),
            "plane_ny": float(n[1]),
            "plane_nz": float(n[2]),
            "plane_d": float(d),
        }

        # [수정] 플래그가 True일 경우 시각화 함수 호출
        if do_plot:
            P = self._points3d()
            z_values_hist = self.depth[self.valid]
            self._plot_results(
                x_mm_orig=P[:, 0],
                y_mm_orig=P[:, 1],
                z_mm_orig=P[:, 2],
                z_values_hist=z_values_hist,
                height_mm=result["max_height_mm"],
                valid_mask=self.valid,
                result_dict=result,
                method_name="Tilt-Invariant (Scaled)"
            )

        return result

# ---------------------------- CLI Example ----------------------------

if __name__ == "__main__":
    # argparse를 이쪽으로 이동
    parser = argparse.ArgumentParser(description="Cup volume estimator from depth map (mm)")
    parser.add_argument("--npy", type=str, required=True, help="Depth .npy path (mm), masked to cup interior")
    parser.add_argument("--fx", type=float, default=190.46873334)
    parser.add_argument("--fy", type=float, default=191.201416)
    parser.add_argument("--cx", type=float, default=120.00074471)
    parser.add_argument("--cy", type=float, default=90.00851171)
    parser.add_argument("--dist", type=float, nargs="+",
                        default=[4.93083651e-01, -1.25632226e+00, 2.28374174e-03, -1.58899167e-05, 5.29625368e-01],
                        help="k1 k2 p1 p2 k3")
    parser.add_argument("--known_diam", type=float, required=True, help="Known rim diameter in mm (e.g., 90)")
    
    # [추가] 플롯 옵션
    parser.add_argument("--plot", action="store_true", help="결과를 3D 플롯으로 시각화합니다.")
    
    args = parser.parse_args()

    try:
        depth = np.load(args.npy)
    except FileNotFoundError:
        print(f"오류: {args.npy} 파일을 찾을 수 없습니다.")
        exit(1)
    except Exception as e:
        print(f"오류: {args.npy} 파일 로드 중 문제 발생: {e}")
        exit(1)

    intr = Intrinsics(args.fx, args.fy, args.cx, args.cy, np.array(args.dist, dtype=np.float64))
    
    if np.count_nonzero(depth > 0) < 50:
        print("오류: NPY 파일에 유효한 깊이 데이터(>0)가 거의 없습니다.")
        exit(1)
        
    est = CupVolumeEstimator(depth, intr)
    
    print("부피 계산 중...")
    out = est.estimate(known_diameter_mm=args.known_diam, do_plot=args.plot)

    # Pretty print
    def fmt(k, v): 
        return f"{k:>32s}: {v:.6f}" if isinstance(v, float) else f"{k:>32s}: {v}"
    
    print("\n--- Cup Volume Estimation (Scaled) ---")
    for k in ["volume_scaled_mm3", "volume_scaled_mL",
              "measured_diameter_from_area_mm", "scale_ratio_by_diameter",
              "max_height_mm", "plane_nx", "plane_ny", "plane_nz", "plane_d"]:
        if k in out:
            print(fmt(k, out[k]))