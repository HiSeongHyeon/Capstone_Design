import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import trimesh
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import Rbf
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class computeVolume:
    def __init__(self, depth_file_path, calibration_file_path=None):
        """
        Parameters:
        - depth_file_path: depth 데이터 파일 경로
        - calibration_file_path: 캘리브레이션 파일 경로
        """
        self.depth_data = np.load(depth_file_path)
        
        # depth 스케일링 수정
        self.depth_scale = 3/4
        self.depth_data = self.depth_data * self.depth_scale
        
        self.image_height, self.image_width = self.depth_data.shape
        
        print(f"Depth 데이터 로드 완료:")
        print(f"  Shape: {self.depth_data.shape}")
        if np.any(self.depth_data > 0):
            print(f"  Min depth: {np.min(self.depth_data[self.depth_data > 0]):.2f} mm")
            print(f"  Max depth: {np.max(self.depth_data):.2f} mm")
        else:
            print("  Depth 데이터가 없습니다.")
        print(f"  Depth scale factor: {self.depth_scale}")

    def find_cup_circle(self):
        """컵의 원형 경계를 찾아 정확한 지름을 계산"""
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
            else:
                diameter_pixels = radius * 2
                
            return diameter_pixels, (x, y)
        
        y_indices, x_indices = np.where(valid_mask)
        if len(y_indices) == 0:
             return self.image_width, (self.image_width/2, self.image_height/2) # 폴백

        diameter_pixels = max(x_indices.max() - x_indices.min(), 
                              y_indices.max() - y_indices.min())
        center = ((x_indices.max() + x_indices.min())/2, 
                  (y_indices.max() + y_indices.min())/2)
        
        return diameter_pixels, center

    def create_point_cloud(self, scale_factor=None, cup_diameter_mm=90):
        """포인트 클라우드 생성"""
        valid_mask = self.depth_data > 0
        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data[valid_mask]
        
        if scale_factor is None:
            pixel_diameter, center = self.find_cup_circle()
            if pixel_diameter == 0: pixel_diameter = 1 # 0으로 나누기 방지
            scale_factor = cup_diameter_mm / pixel_diameter
        else:
            _, center = self.find_cup_circle()

        
        points = np.column_stack([
            (x_indices - center[0]) * scale_factor,
            (y_indices - center[1]) * scale_factor,
            z_values
        ])
        
        return points, scale_factor

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
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
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
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
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
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
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

    # Method 1: 구분구적법 (포인트 클라우드 반환)
    def estimate_cup_volume_improved(self, cup_diameter_mm=90):
        """
        개선된 구분구적 방법 (시각화를 위해 포인트 클라우드 반환)
        """
        valid_mask = self.depth_data > 0
        if not np.any(valid_mask):
            print("구분구적법 실패: 유효한 데이터 없음")
            return 0, None

        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data[valid_mask]
        
        pixel_diameter, center = self.find_cup_circle()
        if pixel_diameter == 0: pixel_diameter = 1 # 0으로 나누기 방지
        
        scale_factor = cup_diameter_mm / pixel_diameter
        print(f"Scale factor (Riemann): {scale_factor}")
        
        pixel_area = scale_factor * scale_factor
        total_volume = np.sum(z_values) * pixel_area
        
        # 포인트 클라우드 생성 (시각화용)
        points = np.column_stack([
            (x_indices - center[0]) * scale_factor,
            (y_indices - center[1]) * scale_factor,
            z_values
        ])
        
        return total_volume, points  # (부피, 포인트 클라우드)

    # Method 4: RBF Interpolation (번호 수정됨)
    def volume_rbf_interpolation(self, cup_diameter_mm=90):
        """RBF 보간을 이용한 표면 재구성"""
        try:
            valid_mask = self.depth_data > 0
            if not np.any(valid_mask):
                print("RBF 실패: 유효한 데이터 없음")
                return 0, None

            y_indices, x_indices = np.where(valid_mask)
            z_values = self.depth_data[valid_mask]
            
            pixel_diameter, center = self.find_cup_circle()
            if pixel_diameter == 0: pixel_diameter = 1
            
            scale_factor = cup_diameter_mm / pixel_diameter
            
            # 데이터 다운샘플링 (RBF는 많은 점에서 느림)
            sample_size = min(10000, len(z_values))
            indices = np.random.choice(len(z_values), sample_size, replace=False)
            
            x_centered = (x_indices[indices] - center[0]) * scale_factor
            y_centered = (y_indices[indices] - center[1]) * scale_factor
            z_sampled = z_values[indices]
            
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
                
                # 바닥면 추가
                mesh = self.add_bottom_to_mesh_trimesh(mesh)
                print(f"메쉬의 최대 높이 (RBF): {np.max(mesh.vertices[:,2]):.2f} mm")
                
                volume = abs(mesh.volume)
                return volume, mesh
            else:
                print("RBF 실패: 그리드 포인트 부족으로 메쉬 생성 불가")
                return 0, None
                
        except Exception as e:
            print(f"RBF Interpolation 실패: {e}")
            return 0, None

    def add_bottom_to_mesh_trimesh(self, mesh):
        """Trimesh 메시에 바닥면 추가"""
        try:
            if len(mesh.vertices) < 3:
                return mesh # 정점이 부족하면 바닥 추가 불가

            vertices = mesh.vertices
            min_z = np.min(vertices[:, 2])
            
            # XY 평면에서 경계 찾기
            xy_points = vertices[:, :2]
            
            # ConvexHull은 3개 이상의 점 필요
            if len(np.unique(xy_points, axis=0)) < 3:
                print("바닥면 추가 실패: 고유한 XY 정점 부족")
                return mesh

            hull_2d = ConvexHull(xy_points)
            
            # 바닥 점들 생성
            bottom_vertices = []
            for idx in hull_2d.vertices:
                bottom_vertices.append([xy_points[idx, 0], xy_points[idx, 1], min_z])
            
            # 바닥 중심점
            center = np.mean(bottom_vertices, axis=0)
            
            # 새 메시 생성
            all_vertices = np.vstack([vertices, bottom_vertices, [center]])
            
            # 바닥 면 생성
            faces = mesh.faces.tolist()
            start_idx = len(vertices)
            center_idx = len(all_vertices) - 1
            
            for i in range(len(bottom_vertices)):
                next_i = (i + 1) % len(bottom_vertices)
                faces.append([start_idx + i, start_idx + next_i, center_idx])
            
            return trimesh.Trimesh(vertices=all_vertices, faces=faces, process=False)
        except Exception as e:
            print(f"바닥면 추가 중 오류: {e}")
            return mesh # 오류 발생 시 원본 메쉬 반환

    def compare_all_methods(self, cup_diameter_mm=90):
        """모든 방법 비교"""
        print("=" * 80)
        print("Arducam Depth Camera - 부피 계산 방법 비교")
        print("=" * 80)
        print(f"실제 부피: 385 mL")
        print(f"컵 직경 설정: {cup_diameter_mm} mm")
        print("=" * 80)
        
        methods = []
        volumes = []
        meshes = [] # 메시 또는 포인트 클라우드 저장
        
        # Method 1: 구분구적법
        print("\n1. 구분구적법 (Riemann Sum)...")
        vol1, data1 = self.estimate_cup_volume_improved(cup_diameter_mm)
        methods.append("Riemann Sum")
        volumes.append(vol1)
        meshes.append(data1) # data1은 np.ndarray (포인트 클라우드)
        if vol1 > 0:
            print(f"   부피: {vol1/1000:.2f} mL, 오차: {abs(vol1/1000 - 385)/385*100:.1f}%")
        
        # Method 2: RBF
        print("\n2. RBF Interpolation...")
        vol4, data4 = self.volume_rbf_interpolation(cup_diameter_mm)
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
        import os
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
        # 기존 파일이 있는지 확인
        np.load('masked_depth.npy')
    except FileNotFoundError:
        # 오류 출력
        raise FileNotFoundError("masked_depth.npy 파일이 없습니다. 해당 파일을 준비해주세요.")
    
    # 컴퓨터 객체 생성
    example = computeVolume('masked_depth.npy')
    
    # 모든 방법 비교 실행
    methods, volumes, meshes = example.compare_all_methods(cup_diameter_mm=90)
    
    # 메시 파일로 저장
    print("\n메시/포인트 클라우드 파일 저장 중...")
    example.save_meshes(meshes, methods)
    
    print("\n완료!")