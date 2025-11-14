import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import trimesh
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import Rbf
import warnings
import os # save_meshes용 import

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class computeVolume:
    def __init__(self, depth_file_path, calibration_file_path=None):
        """
        Parameters:
        - depth_file_path: [수정] masked_raw_depth.npy 파일 경로
        - calibration_file_path: 캘리브레이션 파일 경로
        """
        # [수정] self.depth_data는 이제 '마스킹된 원본' 데이터입니다.
        self.depth_data = np.load(depth_file_path)
        
        self.depth_scale = 1
        self.depth_data = self.depth_data * self.depth_scale
        
        self.image_height, self.image_width = self.depth_data.shape
        
        print(f"Depth 데이터 로드 완료 (마스킹된 원본):")
        print(f"  Shape: {self.depth_data.shape}")
        if np.any(self.depth_data > 0):
            print(f"  Min depth: {np.min(self.depth_data[self.depth_data > 0]):.2f} mm")
            print(f"  Max depth: {np.max(self.depth_data):.2f} mm")
        else:
            print("  Depth 데이터가 없습니다.")
        print(f"  Depth scale factor: {self.depth_scale}")

        # [신규] 1. Baseline 계산 및 상대 높이 데이터 생성
        self.compute_baseline_and_relative_depth()
        
        # [신규] 2. 픽셀 중심 찾기 (X, Y 변환의 기준점)
        # find_cup_circle은 self.depth_data(원본)를 기반으로 작동하므로 수정 불필요
        _ , self.center_pixel = self.find_cup_circle() # (cx, cy) 튜플 저장
        
        # [신규] 3. 원근 보정 포인트 클라우드 생성
        # 이 함수가 self.points (N, 3) 배열을 생성합니다.
        self.create_point_cloud_perspective_corrected()

    def get_scale_factor_at_depth(self, depth_mm):
        """[신규] 주어진 절대 깊이(mm)에 대한 mm/px 스케일 팩터를 계산합니다."""
        # mm/px = 0.003344 * depth + 0.171625
        return 0.003344 * depth_mm + 0.171625

    def compute_baseline_and_relative_depth(self):
        """
        [신규] 원본 depth 데이터에서 baseline을 찾아 상대 높이로 변환합니다.
        findMask.py에서 이동된 로직입니다.
        
        실행 결과:
        - self.relative_depth_data (상대 높이 데이터)
        - self.baseline_depth (계산된 기준점)
        """
        # self.depth_data는 마스킹된 '원본' 깊이 데이터입니다.
        valid_depths = self.depth_data[self.depth_data > 0]
        
        if len(valid_depths) < 20:
            print("경고: 유효한 데이터가 20개 미만입니다. Baseline을 0으로 설정합니다.")
            self.baseline_depth = 0
            # 원본 데이터를 복사 (이후 0 미만 값 제거 로직을 위해)
            self.relative_depth_data = self.depth_data.copy() 
        else:
            sorted_depths = np.sort(valid_depths)
            lowest_20_depths = sorted_depths[:20]
            self.baseline_depth = np.median(lowest_20_depths)
            print(f"계산된 Baseline Depth: {self.baseline_depth:.2f} mm")

            # 상대 높이 계산 (numpy 브로드캐스팅 활용)
            self.relative_depth_data = np.zeros_like(self.depth_data)
            valid_mask = self.depth_data > 0
            self.relative_depth_data[valid_mask] = self.depth_data[valid_mask] - self.baseline_depth
        
        # 0 미만 값 제거 (Baseline보다 낮은 노이즈 등)
        self.relative_depth_data[self.relative_depth_data < 0] = 0

    def find_cup_circle(self):
        """
        컵의 원형 경계를 찾아 중심 픽셀을 반환합니다. 
        (원본 depth 데이터 기준)
        """
        valid_mask = (self.depth_data > 0).astype(np.uint8)
        
        if not np.any(valid_mask):
            print("경고: 유효한 depth 데이터가 없어 컵 경계를 찾을 수 없습니다.")
            return self.image_width, (self.image_width/2, self.image_height/2) # 폴백

        kernel = np.ones((3,3), np.uint8)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_CLOSE, kernel)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_OPEN, kernel)
        
        contours, _ = cv.findContours(valid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            
            (x, y), radius = cv.minEnclosingCircle(largest_contour)
            
            if len(largest_contour) >= 5:
                ellipse = cv.fitEllipse(largest_contour)
                center, (width, height), angle = ellipse
                diameter_pixels = (width + height) / 2
                center_x, center_y = center
            else:
                diameter_pixels = radius * 2
                center_x, center_y = x, y
                
            return diameter_pixels, (center_x, center_y) # (지름, (cx, cy))
        
        y_indices, x_indices = np.where(valid_mask)
        if len(y_indices) == 0:
             return self.image_width, (self.image_width/2, self.image_height/2) # 폴백

        diameter_pixels = max(x_indices.max() - x_indices.min(), 
                              y_indices.max() - y_indices.min())
        center_x = (x_indices.max() + x_indices.min())/2
        center_y = (y_indices.max() + y_indices.min())/2
        
        return diameter_pixels, (center_x, center_y)

    def create_point_cloud_perspective_corrected(self):
        """
        [신규] 픽셀별 깊이를 고려한 원근 보정 포인트 클라우드 생성
        
        - X, Y (mm) = (x_px - cx_px) * scale_factor(z_abs)
        - Z (mm)    = z_abs - z_baseline
        
        결과를 self.points에 저장합니다.
        """
        
        # 1. 유효한 픽셀 인덱스 (y, x) 찾기
        # '상대 높이'가 0보다 큰 지점을 유효한 포인트로 간주
        valid_mask = self.relative_depth_data > 0
        y_indices, x_indices = np.where(valid_mask)
        
        if len(y_indices) == 0:
            print("포인트 클라우드 생성 실패: 유효한 데이터 없음")
            self.points = np.array([]) # 빈 배열 초기화
            return

        # 2. Z축 값 (상대 높이)
        z_values_relative = self.relative_depth_data[valid_mask]
        
        # 3. X, Y 스케일링을 위한 '절대 높이'
        z_values_absolute = self.depth_data[valid_mask]
        
        # 4. 픽셀별 스케일 팩터 계산 (벡터화 연산)
        # (N,) 형태의 배열이 생성됨
        scale_factors = self.get_scale_factor_at_depth(z_values_absolute)
        
        # 5. 픽셀 중심 (미리 계산됨)
        cx, cy = self.center_pixel
        
        # 6. X, Y 좌표 계산 (벡터화)
        # x_mm = (x_pixel - cx) * (mm/px)
        x_mm = (x_indices - cx) * scale_factors
        y_mm = (y_indices - cy) * scale_factors
        
        # (N, 3) 형태의 포인트 클라우드 생성
        self.points = np.column_stack([x_mm, y_mm, z_values_relative])
        
        print(f"원근 보정 포인트 클라우드 생성 완료. (포인트 {len(self.points)}개)")

    def plot_mesh_comparison(self, meshes, volumes, method_names):
        """
        메시(Trimesh) 또는 포인트 클라우드(np.ndarray)를 비교 시각화
        """
        n_methods = len(meshes)
        if n_methods == 0:
            print("시각화할 메시 또는 포인트 클라우드가 없습니다.")
            return
            
        fig = plt.figure(figsize=(max(3, 3 * n_methods), 6))
        
        for i, (data, volume, name) in enumerate(zip(meshes, volumes, method_names)):
            
            # 3D 메시 플롯
            ax = fig.add_subplot(3, n_methods, i+1, projection='3d')
            plot_type = None # 'mesh' 또는 'points'
            vertices = None
            faces = None

            if data is not None:
                if isinstance(data, trimesh.Trimesh):
                    plot_type = 'mesh'
                    vertices = data.vertices
                    faces = data.faces
                elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
                    plot_type = 'points'
                    vertices = data # 포인트 클라우드 자체
                else:
                    ax.text(0.5, 0.5, 0.5, 'Plot Error (Unknown Type)', ha='center', va='center')
                    ax.set_title(f'{name}\nPlot Error')
                    continue
            
            if plot_type == 'mesh':
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                triangles=faces, alpha=0.8, cmap='viridis')
            elif plot_type == 'points':
                # 포인트가 너무 많으면 느리므로 샘플링 (e.g., 20000개)
                sample_size = min(20000, len(vertices))
                if len(vertices) > sample_size:
                    idx = np.random.choice(len(vertices), sample_size, replace=False)
                    sampled_points = vertices[idx]
                else:
                    sampled_points = vertices
                    
                ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                           c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

            if plot_type:
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title(f'{name}\nVolume: {volume/1000:.1f} mL')
                ax.view_init(elev=30, azim=45)
            else:
                ax.text(0.5, 0.5, 0.5, 'Failed', ha='center', va='center')
                ax.set_title(f'{name}\nFailed')
            
            # Top view
            ax2 = fig.add_subplot(3, n_methods, n_methods+i+1)
            if plot_type == 'mesh':
                ax2.triplot(vertices[:, 0], vertices[:, 1], faces, 'b-', alpha=0.3, linewidth=0.5)
                ax2.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                            cmap='viridis', s=1)
            elif plot_type == 'points':
                sample_size = min(20000, len(vertices))
                if len(vertices) > sample_size:
                    idx = np.random.choice(len(vertices), sample_size, replace=False)
                    sampled_points = vertices[idx]
                else:
                    sampled_points = vertices
                ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], c=sampled_points[:, 2], 
                            cmap='viridis', s=0.1, alpha=0.5)
            
            if plot_type:
                ax2.set_xlabel('X (mm)')
                ax2.set_ylabel('Y (mm)')
                ax2.set_title(f'Top View')
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
            
            # Side view
            ax3 = fig.add_subplot(3, n_methods, 2*n_methods+i+1)
            if plot_type == 'mesh':
                ax3.scatter(vertices[:, 0], vertices[:, 2], c=vertices[:, 2], 
                            cmap='viridis', s=1)
            elif plot_type == 'points':
                sample_size = min(20000, len(vertices))
                if len(vertices) > sample_size:
                    idx = np.random.choice(len(vertices), sample_size, replace=False)
                    sampled_points = vertices[idx]
                else:
                    sampled_points = vertices
                ax3.scatter(sampled_points[:, 0], sampled_points[:, 2], c=sampled_points[:, 2], 
                            cmap='viridis', s=0.1, alpha=0.5)

            if plot_type:
                ax3.set_xlabel('X (mm)')
                ax3.set_ylabel('Z (mm)')
                ax3.set_title(f'Side View')
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mesh_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Method 1: 구분구적법
    def estimate_cup_volume_improved(self):
        """
        [수정] 개선된 구분구적 방법 (픽셀별 면적 고려)
        (시각화를 위해 self.points를 반환하지만, 계산은 원본 데이터를 사용)
        """
        valid_mask = self.relative_depth_data > 0
        if not np.any(valid_mask):
            print("구분구적법 실패: 유효한 데이터 없음")
            return 0, None

        # Z값 (상대 높이)
        z_values_relative = self.relative_depth_data[valid_mask]
        
        # 스케일 계산을 위한 절대 높이
        z_values_absolute = self.depth_data[valid_mask]
        
        # 픽셀별 스케일 팩터 및 면적 계산
        scale_factors = self.get_scale_factor_at_depth(z_values_absolute)
        
        # pixel_area (mm^2) = (mm/px) * (mm/px)
        pixel_areas_mm2 = scale_factors * scale_factors
        
        # 픽셀별 부피 = 높이(mm) * 면적(mm^2)
        pixel_volumes = z_values_relative * pixel_areas_mm2
        
        total_volume = np.sum(pixel_volumes)
        
        # 시각화를 위해 미리 계산된 self.points 반환
        return total_volume, self.points

    # Method 4: RBF Interpolation
    def volume_rbf_interpolation(self):
        """[수정] RBF 보간 (원근 보정된 self.points 사용)"""
        try:
            if len(self.points) < 4:
                print("RBF 실패: 유효한 데이터 포인트 부족 (4개 미만)")
                return 0, None

            # [수정] self.points에서 직접 다운샘플링
            sample_size = min(10000, len(self.points))
            indices = np.random.choice(len(self.points), sample_size, replace=False)
            sampled_points = self.points[indices]
            
            x_centered = sampled_points[:, 0]
            y_centered = sampled_points[:, 1]
            z_sampled = sampled_points[:, 2]

            if len(z_sampled) < 4:
                print("RBF 실패: 보간에 필요한 최소 데이터 포인트 부족")
                return 0, None

            # RBF 보간
            rbf = Rbf(x_centered, y_centered, z_sampled, function='multiquadric', smooth=0.5)
            
            # 규칙적인 그리드 생성
            grid_size = 100
            x_min, x_max = x_centered.min(), x_centered.max()
            y_min, y_max = y_centered.min(), y_centered.max()

            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # 높이값 예측
            Z = rbf(X.ravel(), Y.ravel()).reshape(X.shape)

            # 중심이 (0,0)이 아닐 수 있으므로 그리드 경계 기준으로 마스크
            mask_radius_x = (x_max - x_min) / 2
            mask_radius_y = (y_max - y_min) / 2
            mask_center_x = (x_max + x_min) / 2
            mask_center_y = (y_max + y_min) / 2
            
            # 타원형 마스크 (데이터 분포에 맞게)
            mask = (((X - mask_center_x) / mask_radius_x)**2 + 
                    ((Y - mask_center_y) / mask_radius_y)**2) <= 1
            
            # Delaunay triangulation으로 메시 생성
            points_mesh = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if mask[i, j] and Z[i, j] > 0:
                        points_mesh.append([X[i, j], Y[i, j], Z[i, j]])
            
            if len(points_mesh) > 3:
                points_mesh = np.array(points_mesh)
                
                # Trimesh 객체 생성
                tri = Delaunay(points_mesh[:, :2])
                mesh = trimesh.Trimesh(vertices=points_mesh, faces=tri.simplices)
                
                # [수정] 기울어진 뚜껑 추가
                mesh = self.add_tilted_lid_to_mesh(mesh)
                
                print(f"메쉬의 최소 높이 (RBF 표면): {np.min(points_mesh[:,2]):.2f} mm")
                print(f"메쉬의 최대 높이 (RBF 표면): {np.max(points_mesh[:,2]):.2f} mm")
                
                volume = abs(mesh.volume)
                return volume, mesh
            else:
                print("RBF 실패: 그리드 포인트 부족으로 메쉬 생성 불가")
                return 0, None
                
        except Exception as e:
            print(f"RBF Interpolation 실패: {e}")
            return 0, None

    def add_tilted_lid_to_mesh(self, mesh):
        """
        메시의 2D Convex Hull에 해당하는 3D 정점들을 찾아
        최소제곱법으로 평면을 피팅(fit)하여 기울어진 뚜껑을 추가합니다.
        
        이는 컵이 기울어져 있을 때, 컵의 rim(가장자리)에 맞는
        기울어진 뚜껑을 생성하기 위함입니다.
        """
        try:
            vertices = mesh.vertices
            if len(vertices) < 3:
                print("뚜껑 추가 실패: 정점 부족 (3개 미만)")
                return mesh

            xy_points = vertices[:, :2]
            
            # 고유한 XY 정점이 3개 미만이면 ConvexHull 계산 불가
            unique_xy_points = np.unique(xy_points, axis=0)
            if len(unique_xy_points) < 3:
                print("뚜껑 추가 실패: 고유한 XY 정점 부족 (3개 미만)")
                return mesh

            # 1. 2D Convex Hull을 찾아 Rim의 인덱스 식별
            hull_2d = ConvexHull(xy_points)
            
            # RBF 그리드 정점(vertices)에서 2D 헐(hull)의 경계에 해당하는 3D 정점(rim_points)을 추출
            rim_indices = hull_2d.vertices
            rim_points = vertices[rim_indices] # (N, 3)
# [RBF_mesh.py]의 add_tilted_lid_to_mesh 함수 내부

            # ... (이전 코드) ...
            
            # RBF 그리드 정점(vertices)에서 2D 헐(hull)의 경계에 해당하는 3D 정점(rim_points)을 추출
            rim_indices = hull_2d.vertices
            rim_points = vertices[rim_indices] # (N, 3)
            
            # ======================================================
            # [여기에 아래 코드 삽입]
            # 림 직경(mm)을 계산하고 출력합니다.
            try: 
                rim_xy = rim_points[:, :2] # 림의 X, Y 좌표 (mm)
                x_min, y_min = np.min(rim_xy, axis=0)
                x_max, y_max = np.max(rim_xy, axis=0)
                diameter_x = x_max - x_min
                diameter_y = y_max - y_min
                
                print("\n" + "=" * 30)
                print("--- 림(Rim) 직경 측정 (mm) ---")
                print(f"  측정된 림 X-직경: {diameter_x:.2f} mm")
                print(f"  측정된 림 Y-직경: {diameter_y:.2f} mm")
                print(f"  측정된 림 평균 직경: {((diameter_x + diameter_y) / 2):.2f} mm")
                print("=" * 30 + "\n")

            except Exception as e:
                print(f"[림 직경 측정 중 오류]: {e}")
            # [삽입 코드 끝]
            # ======================================================


            # ... (이후 코드 계속) ...
            
            if len(rim_points) < 3:
                 print("뚜껑 추가 실패: Rim 포인트 부족 (3개 미만)")
                 return mesh

            # 2. Rim 포인트들로 평면 피팅 (z = ax + by + c)
            # A * [a, b, c]' = z
            A = np.c_[rim_points[:, 0], rim_points[:, 1], np.ones(len(rim_points))]
            b = rim_points[:, 2]
            
            # 최소제곱법으로 평면의 계수(a, b, c) 계산
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            a, b, c = coeffs
            
            print(f"평면 피팅 완료: z = {a:.2f}x + {b:.2f}y + {c:.2f}")

            # 3. 기울어진 뚜껑 정점 생성
            # 2D 헐(hull)의 (x, y) 좌표 사용
            lid_base_xy = xy_points[rim_indices]
            
            # 피팅된 평면 방정식을 사용해 z_lid 계산
            z_lid = a * lid_base_xy[:, 0] + b * lid_base_xy[:, 1] + c
            
            # 최종 뚜껑 정점 (x, y, z_lid)
            lid_vertices = np.c_[lid_base_xy, z_lid]
            
            # 4. 뚜껑 메시화 (기존 로직과 유사)
            center = np.mean(lid_vertices, axis=0)
            
            # 새 메시 생성
            all_vertices = np.vstack([vertices, lid_vertices, [center]])
            
            faces = mesh.faces.tolist()
            start_idx = len(vertices)
            center_idx = len(all_vertices) - 1
            
            for i in range(len(lid_vertices)):
                next_i = (i + 1) % len(lid_vertices)
                # 뚜껑 면의 법선 벡터가 올바른 방향(보통 위쪽)을 향하도록 정점 순서 조정
                faces.append([start_idx + i, center_idx, start_idx + next_i])
            
            new_mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces, process=True)
            
            if not new_mesh.is_watertight:
                print("경고: 뚜껑을 추가했지만 메쉬가 watertight하지 않습니다.")
                # new_mesh.fill_holes() # 구멍 메우기 시도
            
            return new_mesh
            
        except Exception as e:
            print(f"기울어진 뚜껑 추가 중 오류: {e}")
            return mesh # 오류 발생 시 원본 메쉬 반환

    def compare_all_methods(self): # [수정] cup_diameter_mm 인자 제거
        """모든 방법 비교"""
        print("=" * 80)
        print("Arducam Depth Camera - 부피 계산 방법 비교 (원근 보정 적용됨)")
        print("=" * 80)
        print(f"실제 부피: 385 mL")
        # print(f"컵 직경 설정: {cup_diameter_mm} mm") # [삭제]
        print("=" * 80)
        
        methods = []
        volumes = []
        meshes = [] # 메시 또는 포인트 클라우드 저장
        
        # Method 1: 구분구적법
        print("\n1. 구분구적법 (Riemann Sum)...")
        vol1, data1 = self.estimate_cup_volume_improved() # [수정] 인자 제거
        methods.append("Riemann Sum")
        volumes.append(vol1)
        meshes.append(data1) # data1은 np.ndarray (포인트 클라우드)
        if vol1 > 0:
            print(f"   부피: {vol1/1000:.2f} mL, 오차: {abs(vol1/1000 - 385)/385*100:.1f}%")
        
        # Method 2: RBF
        print("\n2. RBF Interpolation...")
        vol4, data4 = self.volume_rbf_interpolation() # [수정] 인자 제거
        methods.append("RBF")
        volumes.append(vol4)
        meshes.append(data4) # data4는 trimesh.Trimesh
        if vol4 > 0:
            print(f"   부피: {vol4/1000:.2f} mL, 오차: {abs(vol4/1000 - 385)/385*100:.1f}%")
        
        # 결과 요약
        print("\n" + "=" * 80)
        print("결과 요약:")
        print("-" * 80)
        print(f"{'Method':<20} {'Volume (mL)':<15} {'Error (%)':<15} {'Status'}")
        print("-" * 80)
        
        for method, volume, data in zip(methods, volumes, meshes):
            if volume > 0:
                error = abs(volume/1000 - 385) / 385 * 100
                status = "✓ (Mesh)" if isinstance(data, trimesh.Trimesh) else "✓ (Points)"
                print(f"{method:<20} {volume/1000:<15.2f} {error:<15.1f} {status}")
            else:
                print(f"{method:<20} {'Failed':<15} {'-':<15} ✗")
        
        # 메시 비교 시각화
        print("\n메시/포인트 클라우드 형상 비교 시각화 생성 중...")
        valid_data = [d for d in meshes if d is not None]
        valid_volumes = [v for i, v in enumerate(volumes) if meshes[i] is not None]
        valid_methods = [methods[i] for i in range(len(methods)) if meshes[i] is not None]
        
        if valid_data:
            self.plot_mesh_comparison(valid_data, valid_volumes, valid_methods)
        
        # 부피 비교 막대 그래프
        self.plot_volume_comparison(methods, volumes)
        
        return methods, volumes, meshes

    def plot_volume_comparison(self, methods, volumes):
        """부피 비교 막대 그래프"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 유효한 결과만 필터링
        valid_methods = []
        valid_volumes = []
        for method, volume in zip(methods, volumes):
            if volume > 0:
                valid_methods.append(method)
                valid_volumes.append(volume/1000)
        
        if valid_volumes:
            colors = ['blue' if abs(v - 385) / 385 < 0.1 else 'orange' for v in valid_volumes]
            bars = ax1.bar(range(len(valid_methods)), valid_volumes, color=colors)
            ax1.axhline(y=385, color='red', linestyle='--', label='실제 부피 (385 mL)')
            ax1.set_xticks(range(len(valid_methods)))
            ax1.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax1.set_ylabel('부피 (mL)')
            ax1.set_title('부피 계산 방법별 결과')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            for i, (bar, v) in enumerate(zip(bars, valid_volumes)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         f'{v:.1f}', ha='center', va='bottom')
            
            errors = [abs(v - 385) / 385 * 100 for v in valid_volumes]
            colors2 = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in errors]
            bars2 = ax2.bar(range(len(valid_methods)), errors, color=colors2)
            ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% 오차')
            ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% 오차')
            ax2.set_xticks(range(len(valid_methods)))
            ax2.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax2.set_ylabel('오차율 (%)')
            ax2.set_title('방법별 오차율')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            for bar, e in zip(bars2, errors):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{e:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('volume_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def save_meshes(self, meshes, methods):
        """메시(Trimesh) 또는 포인트 클라우드(np.ndarray)를 파일로 저장"""
        os.makedirs('meshes', exist_ok=True)
        
        for data, method in zip(meshes, methods):
            if data is not None:
                filename = f"meshes/{method.replace(' ', '_').lower()}.ply"
                try:
                    if isinstance(data, trimesh.Trimesh):
                        data.export(filename)
                        print(f"메시 저장: {filename}")
                    elif isinstance(data, np.ndarray):
                        # Trimesh를 사용해 포인트 클라우드 저장
                        pc = trimesh.PointCloud(data)
                        pc.export(filename)
                        print(f"포인트 클라우드 저장: {filename}")
                    else:
                        print(f"저장 스킵 (지원되지 않는 타입: {type(data)}): {method}")
                except Exception as e:
                    print(f"파일 저장 실패 ({method}): {e}")

if __name__ == "__main__":
    try:
        # [수정] 'masked_raw_depth.npy' 파일을 로드하도록 변경
        np.load('masked_depth.npy')
    except FileNotFoundError:
        raise FileNotFoundError("masked_depth.npy 파일이 없습니다. 먼저 findMask.py를 실행해주세요.")
    
    # [수정] 입력 파일 이름 변경
    example = computeVolume('masked_depth.npy')
    
    # [수정] compare_all_methods에서 인자 제거
    methods, volumes, meshes = example.compare_all_methods()
    
    # 메시 파일로 저장
    print("\n메시/포인트 클라우드 파일 저장 중...")
    example.save_meshes(meshes, methods)
    
    print("\n완료!")