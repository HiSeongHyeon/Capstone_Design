import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import Rbf
import warnings
import os 

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class computeVolume:
    def __init__(self, depth_file_path, calibration_file_path=None):
        self.depth_data = np.load(depth_file_path)
        
        self.depth_scale = 1
        self.depth_data = self.depth_data * self.depth_scale
        
        self.image_height, self.image_width = self.depth_data.shape
        
        print(f"Depth 데이터 로드 완료:")
        print(f"  Shape: {self.depth_data.shape}")
        if np.any(self.depth_data > 0):
            print(f"  Min depth: {np.min(self.depth_data[self.depth_data > 0]):.2f} mm")
            print(f"  Max depth: {np.max(self.depth_data):.2f} mm")
        else:
            print("  Depth 데이터가 없습니다.")

    def calculate_average_scale_factor(self):
        """
        컵 영역의 평균 깊이를 이용하여 단일 변환 계수(mm/px)를 계산
        회귀식: mm/px = 0.003690 * depth + 0.123885
        """
        valid_z = self.depth_data[self.depth_data > 0]
        
        if len(valid_z) == 0:
            print("경고: 유효한 Depth 데이터가 없어 기본 스케일(1.0)을 사용합니다.")
            return 1.0, 0
            
        mean_depth = np.mean(valid_z)
        scale_factor = 0.003690 * mean_depth +0.123885 #need fix
        
        print(f"  [Scale Calculation]")
        print(f"  평균 깊이: {mean_depth:.2f} mm")
        print(f"  적용된 Scale Factor: {scale_factor:.5f} mm/px")
        
        return scale_factor, mean_depth

    def find_cup_center(self):
        """컵의 중심 좌표(픽셀)만 찾음"""
        valid_mask = (self.depth_data > 0).astype(np.uint8)
        
        if not np.any(valid_mask):
            return (self.image_width/2, self.image_height/2)

        kernel = np.ones((3,3), np.uint8)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_CLOSE, kernel)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_OPEN, kernel)
        
        contours, _ = cv.findContours(valid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            (x, y), _ = cv.minEnclosingCircle(largest_contour)
            return (x, y)
        
        # 컨투어 실패 시 무게 중심 사용
        y_indices, x_indices = np.where(valid_mask)
        if len(y_indices) == 0:
             return (self.image_width/2, self.image_height/2)

        center_x = (x_indices.max() + x_indices.min()) / 2
        center_y = (y_indices.max() + y_indices.min()) / 2
        return (center_x, center_y)

    def plot_mesh_comparison(self, meshes, volumes, method_names):
        """메시(Trimesh) 또는 포인트 클라우드(np.ndarray)를 비교 시각화"""
        n_methods = len(meshes)
        if n_methods == 0: return
            
        fig = plt.figure(figsize=(max(3, 3 * n_methods), 6))
        
        for i, (data, volume, name) in enumerate(zip(meshes, volumes, method_names)):
            ax = fig.add_subplot(3, n_methods, i+1, projection='3d')
            plot_type = None 
            vertices = None
            faces = None

            if data is not None:
                if isinstance(data, trimesh.Trimesh):
                    plot_type = 'mesh'
                    vertices = data.vertices
                    faces = data.faces
                elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
                    plot_type = 'points'
                    vertices = data 
                
            if plot_type == 'mesh':
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                triangles=faces, alpha=0.8, cmap='viridis')
            elif plot_type == 'points':
                sample_size = min(20000, len(vertices))
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
                ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                           c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

            if plot_type:
                ax.set_title(f'{name}\nVol: {volume/1000:.1f} mL')
                ax.view_init(elev=30, azim=45)
            
            # Top view
            ax2 = fig.add_subplot(3, n_methods, n_methods+i+1)
            if plot_type == 'mesh':
                ax2.triplot(vertices[:, 0], vertices[:, 1], faces, 'b-', alpha=0.3, linewidth=0.5)
            elif plot_type == 'points':
                sample_size = min(20000, len(vertices))
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
                ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], c=sampled_points[:, 2], 
                            cmap='viridis', s=0.1, alpha=0.5)
            ax2.set_title('Top View')
            ax2.set_aspect('equal')
            
            # Side view
            ax3 = fig.add_subplot(3, n_methods, 2*n_methods+i+1)
            if plot_type == 'mesh':
                ax3.scatter(vertices[:, 0], vertices[:, 2], c=vertices[:, 2], cmap='viridis', s=1)
            elif plot_type == 'points':
                sample_size = min(20000, len(vertices))
                idx = np.random.choice(len(vertices), sample_size, replace=False)
                sampled_points = vertices[idx]
                ax3.scatter(sampled_points[:, 0], sampled_points[:, 2], c=sampled_points[:, 2], 
                            cmap='viridis', s=0.1, alpha=0.5)
            ax3.set_title('Side View')

        plt.tight_layout()
        plt.savefig('mesh_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Method 1: 구분구적법 (단일 평균 스케일 적용)
    def estimate_cup_volume_improved(self):
        """
        개선된 구분구적 방법 (회귀식 기반 단일 스케일 적용)
        """
        valid_mask = self.depth_data > 0
        if not np.any(valid_mask):
            return 0, None

        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data[valid_mask]
        
        center = self.find_cup_center()
        
        # [수정] 단일 평균 스케일 팩터 계산
        scale_factor, _ = self.calculate_average_scale_factor()
        
        # 모든 픽셀에 동일한 면적 적용
        pixel_area = scale_factor * scale_factor
        total_volume = np.sum(z_values) * pixel_area
        
        # 포인트 클라우드 생성 (시각화용) - 단일 스케일 적용
        points = np.column_stack([
            (x_indices - center[0]) * scale_factor,
            (y_indices - center[1]) * scale_factor,
            z_values
        ])
        
        return total_volume, points

    # Method 2: RBF Interpolation (단일 평균 스케일 적용)
    def volume_rbf_interpolation(self):
        """RBF 보간을 이용한 표면 재구성"""
        try:
            valid_mask = self.depth_data > 0
            if not np.any(valid_mask):
                return 0, None

            y_indices, x_indices = np.where(valid_mask)
            z_values = self.depth_data[valid_mask]
            
            center = self.find_cup_center()
            
            # [수정] 단일 평균 스케일 팩터 계산
            scale_factor, _ = self.calculate_average_scale_factor()
            
            # 데이터 다운샘플링
            sample_size = min(10000, len(z_values))
            indices = np.random.choice(len(z_values), sample_size, replace=False)
            
            # [수정] 모든 점에 동일한 스케일 적용 (포인트 클라우드 보정 없음)
            x_centered = (x_indices[indices] - center[0]) * scale_factor
            y_centered = (y_indices[indices] - center[1]) * scale_factor
            z_sampled = z_values[indices]
            
            if len(z_sampled) < 4:
                return 0, None

            # RBF 보간
            rbf = Rbf(x_centered, y_centered, z_sampled, function='multiquadric', smooth=0.5)
            
            grid_size = 100
            x_min, x_max = x_centered.min(), x_centered.max()
            y_min, y_max = y_centered.min(), y_centered.max()

            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            Z = rbf(X.ravel(), Y.ravel()).reshape(X.shape)

            mask_radius_x = (x_max - x_min) / 2
            mask_radius_y = (y_max - y_min) / 2
            mask_center_x = (x_max + x_min) / 2
            mask_center_y = (y_max + y_min) / 2
            
            mask = (((X - mask_center_x) / mask_radius_x)**2 + 
                    ((Y - mask_center_y) / mask_radius_y)**2) <= 1
            
            points_mesh = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if mask[i, j] and Z[i, j] > 0:
                        points_mesh.append([X[i, j], Y[i, j], Z[i, j]])
            
            if len(points_mesh) > 3:
                points_mesh = np.array(points_mesh)
                tri = Delaunay(points_mesh[:, :2])
                mesh = trimesh.Trimesh(vertices=points_mesh, faces=tri.simplices)
                
                mesh = self.add_tilted_lid_to_mesh(mesh)
                
                volume = abs(mesh.volume)
                return volume, mesh
            else:
                return 0, None
                
        except Exception as e:
            print(f"RBF Interpolation 실패: {e}")
            return 0, None

    def add_tilted_lid_to_mesh(self, mesh):
        """메시의 2D Convex Hull과 평면 피팅을 이용해 뚜껑 추가"""
        try:
            vertices = mesh.vertices
            if len(vertices) < 3: return mesh

            xy_points = vertices[:, :2]
            unique_xy_points = np.unique(xy_points, axis=0)
            if len(unique_xy_points) < 3: return mesh

            hull_2d = ConvexHull(xy_points)
            rim_indices = hull_2d.vertices
            rim_points = vertices[rim_indices]
            
            if len(rim_points) < 3: return mesh

            A = np.c_[rim_points[:, 0], rim_points[:, 1], np.ones(len(rim_points))]
            b = rim_points[:, 2]
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            a, b, c = coeffs
            
            lid_base_xy = xy_points[rim_indices]
            z_lid = a * lid_base_xy[:, 0] + b * lid_base_xy[:, 1] + c
            lid_vertices = np.c_[lid_base_xy, z_lid]
            
            center = np.mean(lid_vertices, axis=0)
            
            all_vertices = np.vstack([vertices, lid_vertices, [center]])
            faces = mesh.faces.tolist()
            start_idx = len(vertices)
            center_idx = len(all_vertices) - 1
            
            for i in range(len(lid_vertices)):
                next_i = (i + 1) % len(lid_vertices)
                faces.append([start_idx + i, center_idx, start_idx + next_i])
            
            new_mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces, process=True)
            return new_mesh
            
        except Exception as e:
            print(f"뚜껑 추가 오류: {e}")
            return mesh

    def compare_all_methods(self):
        print("=" * 80)
        print("Depth Camera Volume Calculation (Mean Depth Scale Factor)")
        print(f"Formula: mm/px = 0.003344 * MEAN_DEPTH + 0.171625")
        print("=" * 80)
        
        methods = []
        volumes = []
        meshes = [] 
        
        # Method 1
        print("\n1. Riemann Sum (Fixed Average Scale)...")
        vol1, data1 = self.estimate_cup_volume_improved()
        methods.append("Riemann Sum")
        volumes.append(vol1)
        meshes.append(data1)
        if vol1 > 0:
            print(f"   부피: {vol1/1000:.2f} mL, 오차: {abs(vol1/1000 - 385)/385*100:.1f}%")
        
        # Method 2
        print("\n2. RBF Interpolation (Fixed Average Scale)...")
        vol4, data4 = self.volume_rbf_interpolation()
        methods.append("RBF")
        volumes.append(vol4)
        meshes.append(data4)
        if vol4 > 0:
            print(f"   부피: {vol4/1000:.2f} mL, 오차: {abs(vol4/1000 - 385)/385*100:.1f}%")
        
        print("\n" + "=" * 80)
        print(f"{'Method':<20} {'Volume (mL)':<15} {'Error (%)':<15}")
        print("-" * 80)
        
        for method, volume, data in zip(methods, volumes, meshes):
            if volume > 0:
                error = abs(volume/1000 - 385) / 385 * 100
                print(f"{method:<20} {volume/1000:<15.2f} {error:<15.1f}")
            else:
                print(f"{method:<20} {'Failed':<15} {'-':<15}")
        
        self.plot_volume_comparison(methods, volumes)
        
        # 시각화 (유효한 데이터만)
        valid_data = [d for d in meshes if d is not None]
        valid_volumes = [v for i, v in enumerate(volumes) if meshes[i] is not None]
        valid_methods = [methods[i] for i in range(len(methods)) if meshes[i] is not None]
        if valid_data:
            self.plot_mesh_comparison(valid_data, valid_volumes, valid_methods)

        return methods, volumes, meshes

    def plot_volume_comparison(self, methods, volumes):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        valid_methods = []
        valid_volumes = []
        for method, volume in zip(methods, volumes):
            if volume > 0:
                valid_methods.append(method)
                valid_volumes.append(volume/1000)
        
        if valid_volumes:
            ax1.bar(valid_methods, valid_volumes, color='blue', alpha=0.7)
            ax1.axhline(y=385, color='red', linestyle='--', label='Actual (385 mL)')
            ax1.set_title('Volume Comparison')
            ax1.legend()
            
            errors = [abs(v - 385) / 385 * 100 for v in valid_volumes]
            ax2.bar(valid_methods, errors, color='orange', alpha=0.7)
            ax2.set_title('Error Rate (%)')
            
            for i, v in enumerate(valid_volumes):
                ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom')
            for i, e in enumerate(errors):
                ax2.text(i, e, f'{e:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('volume_comparison.png')

    def save_meshes(self, meshes, methods):
        os.makedirs('meshes', exist_ok=True)
        for data, method in zip(meshes, methods):
            if data is not None:
                filename = f"meshes/{method.replace(' ', '_').lower()}.ply"
                try:
                    if isinstance(data, trimesh.Trimesh):
                        data.export(filename)
                    elif isinstance(data, np.ndarray):
                        pc = trimesh.PointCloud(data)
                        pc.export(filename)
                    print(f"Saved: {filename}")
                except Exception as e:
                    print(f"Save failed ({method}): {e}")

if __name__ == "__main__":
    try:
        np.load('masked_depth.npy')
    except FileNotFoundError:
        raise FileNotFoundError("masked_depth.npy 파일이 없습니다.")
    
    example = computeVolume('masked_depth.npy')
    methods, volumes, meshes = example.compare_all_methods()
    example.save_meshes(meshes, methods)
    print("\n완료!")