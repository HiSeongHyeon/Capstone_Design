import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import Rbf
from skimage import measure
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
        
        # depth 스케일링 수정 - 실험적으로 조정 가능
        # 원래 3/4 스케일링이 과도한 것으로 보임
        self.depth_scale = 3/4  # 조정 가능한 파라미터
        self.depth_data = self.depth_data * self.depth_scale
        
        self.image_height, self.image_width = self.depth_data.shape
        
        print(f"Depth 데이터 로드 완료:")
        print(f"  Shape: {self.depth_data.shape}")
        print(f"  Min depth: {np.min(self.depth_data[self.depth_data > 0]):.2f} mm")
        print(f"  Max depth: {np.max(self.depth_data):.2f} mm")
        print(f"  Depth scale factor: {self.depth_scale}")

    def find_cup_circle(self):
        """컵의 원형 경계를 찾아 정확한 지름을 계산"""
        valid_mask = (self.depth_data > 0).astype(np.uint8)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_CLOSE, kernel)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv.findContours(valid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어 선택 (컵)
            largest_contour = max(contours, key=cv.contourArea)
            
            # 최소 외접원 찾기
            (x, y), radius = cv.minEnclosingCircle(largest_contour)
            
            # 타원 피팅 (더 정확한 방법)
            if len(largest_contour) >= 5:
                ellipse = cv.fitEllipse(largest_contour)
                center, (width, height), angle = ellipse
                # 평균 지름 사용
                diameter_pixels = (width + height) / 2
            else:
                diameter_pixels = radius * 2
                
            return diameter_pixels, (x, y)
        
        # 폴백: 바운딩 박스 사용
        y_indices, x_indices = np.where(valid_mask)
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
        
        pixel_diameter, center = self.find_cup_circle()
        
        if scale_factor is None:
            scale_factor = cup_diameter_mm / pixel_diameter
        
        points = np.column_stack([
            (x_indices - center[0]) * scale_factor,
            (y_indices - center[1]) * scale_factor,
            z_values
        ])
        
        return points, scale_factor

    def plot_mesh_comparison(self, meshes, volumes, method_names):
        """여러 메시 재구성 결과를 비교 시각화"""
        n_methods = len(meshes)
        fig = plt.figure(figsize=(20, 12))
        
        for i, (mesh, volume, name) in enumerate(zip(meshes, volumes, method_names)):
            # 3D 메시 플롯
            ax = fig.add_subplot(3, n_methods, i+1, projection='3d')
            
            if mesh is not None:
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    # Open3D 메시를 numpy로 변환
                    vertices = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles)
                elif isinstance(mesh, trimesh.Trimesh):
                    vertices = mesh.vertices
                    faces = mesh.faces
                else:
                    vertices = mesh['vertices']
                    faces = mesh['faces']
                
                # 메시 그리기
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                               triangles=faces, alpha=0.8, cmap='viridis')
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
            if mesh is not None:
                ax2.triplot(vertices[:, 0], vertices[:, 1], faces, 'b-', alpha=0.3, linewidth=0.5)
                ax2.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                           cmap='viridis', s=1)
                ax2.set_xlabel('X (mm)')
                ax2.set_ylabel('Y (mm)')
                ax2.set_title(f'Top View')
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
            
            # Side view
            ax3 = fig.add_subplot(3, n_methods, 2*n_methods+i+1)
            if mesh is not None:
                ax3.scatter(vertices[:, 0], vertices[:, 2], c=vertices[:, 2], 
                           cmap='viridis', s=1)
                ax3.set_xlabel('X (mm)')
                ax3.set_ylabel('Z (mm)')
                ax3.set_title(f'Side View')
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mesh_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Method 1: 원래 구분구적법
    def estimate_cup_volume_improved(self, cup_diameter_mm=90):
        """개선된 구분구적 방법"""
        valid_mask = self.depth_data > 0
        pixel_diameter, center = self.find_cup_circle()
        scale_factor = cup_diameter_mm / pixel_diameter
        print(scale_factor)
        
        pixel_area = scale_factor * scale_factor
        valid_depths = self.depth_data[valid_mask]
        total_volume = np.sum(valid_depths) * pixel_area
        
        return total_volume, None  # mesh는 없음

    # Method 2: Alpha Shape
    def volume_alpha_shape(self, cup_diameter_mm=90, alpha=2.0):
        """Alpha Shape를 이용한 메시 생성 및 부피 계산"""
        try:
            points, scale_factor = self.create_point_cloud(cup_diameter_mm=cup_diameter_mm)
            
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 노멀 추정
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
            )
            
            # Alpha shape 메시 생성 - 더 큰 alpha 값 사용
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=5.0)
            
            # 메시 정리
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Trimesh로 변환하여 watertight 만들기
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                
                # 구멍 메우기
                tri_mesh.fill_holes()
                
                # 바닥면 추가
                tri_mesh = self.add_bottom_to_mesh_trimesh(tri_mesh)
                
                # Watertight 확인 및 수정
                if not tri_mesh.is_watertight:
                    tri_mesh = trimesh.convex.convex_hull(tri_mesh)
                
                volume = abs(tri_mesh.volume)
                
                # 다시 Open3D로 변환 (시각화용)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
                
                return volume, mesh
            else:
                return 0, None
                
        except Exception as e:
            print(f"Alpha Shape 실패: {e}")
            return 0, None

    # Method 3: Poisson Surface Reconstruction
    def volume_poisson_reconstruction(self, cup_diameter_mm=90, depth=9):
        """Poisson 재구성을 이용한 메시 생성"""
        try:
            points, scale_factor = self.create_point_cloud(cup_diameter_mm=cup_diameter_mm)
            
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 노멀 추정
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
            )
            
            # 노멀 방향 일관성 확보
            pcd.orient_normals_consistent_tangent_plane(100)
            
            # Poisson 표면 재구성
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=1.1, linear_fit=False
            )
            
            # 메시 트리밍 (불필요한 부분 제거)
            bbox = pcd.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)
            
            # Trimesh로 변환
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                
                # 구멍 메우기
                tri_mesh.fill_holes()
                
                # 바닥면 추가
                tri_mesh = self.add_bottom_to_mesh_trimesh(tri_mesh)
                
                # Watertight 확인
                if not tri_mesh.is_watertight:
                    # 컨벡스 헐로 폐곡면 만들기
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(tri_mesh.vertices)
                    tri_mesh = trimesh.Trimesh(vertices=tri_mesh.vertices, faces=hull.simplices)
                
                volume = abs(tri_mesh.volume)
                
                # 다시 Open3D로 변환 (시각화용)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
                
                return volume, mesh
            else:
                return 0, None
                
        except Exception as e:
            print(f"Poisson Reconstruction 실패: {e}")
            return 0, None

    # Method 4: RBF Interpolation
    def volume_rbf_interpolation(self, cup_diameter_mm=90):
        """RBF 보간을 이용한 표면 재구성"""
        try:
            valid_mask = self.depth_data > 0
            y_indices, x_indices = np.where(valid_mask)
            z_values = self.depth_data[valid_mask]
            
            pixel_diameter, center = self.find_cup_circle()
            scale_factor = cup_diameter_mm / pixel_diameter
            
            # 데이터 다운샘플링 (RBF는 많은 점에서 느림)
            sample_size = min(10000, len(z_values))
            indices = np.random.choice(len(z_values), sample_size, replace=False)
            
            x_centered = (x_indices[indices] - center[0]) * scale_factor
            y_centered = (y_indices[indices] - center[1]) * scale_factor
            z_sampled = z_values[indices]
            
            # RBF 보간
            rbf = Rbf(x_centered, y_centered, z_sampled, function='multiquadric', smooth=0.5)
            
            # 규칙적인 그리드 생성
            grid_size = 100
            x_grid = np.linspace(x_centered.min(), x_centered.max(), grid_size)
            y_grid = np.linspace(y_centered.min(), y_centered.max(), grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # 높이값 예측
            Z = rbf(X.ravel(), Y.ravel()).reshape(X.shape)
            
            # 원형 마스크 적용 (컵 경계 밖 제거)
            radius_mm = cup_diameter_mm / 2
            mask = np.sqrt(X**2 + Y**2) <= radius_mm
            # Z[~mask] = 0
            
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
                # 메쉬의 평균 높이 출력
                print(f"메쉬의 평균 높이: {np.mean(mesh.vertices[:,2]):.2f} mm")
                
                volume = abs(mesh.volume)
                return volume, mesh
            else:
                return 0, None
                
        except Exception as e:
            print(f"RBF Interpolation 실패: {e}")
            return 0, None

    # Method 5: Cylindrical Shell Integration
    def volume_cylindrical_integration(self, cup_diameter_mm=90):
        """원통 좌표계 기반 부피 계산"""
        try:
            valid_mask = self.depth_data > 0
            pixel_diameter, center = self.find_cup_circle()
            scale_factor = cup_diameter_mm / pixel_diameter
            
            radius_mm = cup_diameter_mm / 2
            n_rings = 30
            n_angles = 72
            
            volume = 0
            r_values = np.linspace(0, radius_mm, n_rings)
            
            # 메시 생성을 위한 점들 저장
            vertices = []
            faces = []
            vertex_count = 0
            
            for i in range(len(r_values)-1):
                r_inner = r_values[i]
                r_outer = r_values[i+1]
                r_mean = (r_inner + r_outer) / 2
                
                ring_points = []
                for j, angle in enumerate(np.linspace(0, 2*np.pi, n_angles, endpoint=False)):
                    x = center[0] + r_mean * np.cos(angle) / scale_factor
                    y = center[1] + r_mean * np.sin(angle) / scale_factor
                    
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < self.image_width and 0 <= yi < self.image_height:
                        if self.depth_data[yi, xi] > 0:
                            depth = self.depth_data[yi, xi]
                            vertices.append([
                                r_mean * np.cos(angle),
                                r_mean * np.sin(angle),
                                depth
                            ])
                            ring_points.append(vertex_count)
                            vertex_count += 1
                
                # 링 사이의 면 생성
                if i > 0 and len(ring_points) > 2:
                    for j in range(len(ring_points)-1):
                        if j < len(ring_points)-1:
                            faces.append([ring_points[j], ring_points[j+1], 
                                        ring_points[j] - len(ring_points)])
                            faces.append([ring_points[j+1], 
                                        ring_points[j+1] - len(ring_points),
                                        ring_points[j] - len(ring_points)])
                
                # 부피 계산
                if ring_points:
                    mean_depth = np.mean([vertices[idx][2] for idx in ring_points])
                    ring_volume = np.pi * (r_outer**2 - r_inner**2) * mean_depth
                    volume += ring_volume
            
            # 메시 생성
            if len(vertices) > 3 and len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=np.array(vertices), 
                                      faces=np.array(faces))
                mesh = self.add_bottom_to_mesh_trimesh(mesh)
            else:
                mesh = None
            
            return volume, mesh
            
        except Exception as e:
            print(f"Cylindrical Integration 실패: {e}")
            return 0, None

    # Method 6: Ball Pivoting Algorithm
    def volume_ball_pivoting(self, cup_diameter_mm=90):
        """Ball Pivoting Algorithm으로 메시 생성"""
        try:
            points, scale_factor = self.create_point_cloud(cup_diameter_mm=cup_diameter_mm)
            
            # Open3D 포인트 클라우드 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 노멀 추정
            pcd.estimate_normals()
            
            # Ball Pivoting 파라미터
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            
            # BPA 메시 생성
            radii = [avg_dist * 0.5, avg_dist * 1, avg_dist * 2]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            # 메시 정리
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            
            # Trimesh로 변환
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                
                # 구멍 메우기
                tri_mesh.fill_holes()
                
                # 바닥면 추가
                tri_mesh = self.add_bottom_to_mesh_trimesh(tri_mesh)
                
                # Watertight 확인
                if not tri_mesh.is_watertight:
                    # repair mesh
                    trimesh.repair.fix_normals(tri_mesh)
                    trimesh.repair.fix_inversion(tri_mesh)
                    tri_mesh.fill_holes()
                
                volume = abs(tri_mesh.volume)
                
                # 다시 Open3D로 변환 (시각화용)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
                
                return volume, mesh
            else:
                return 0, None
                
        except Exception as e:
            print(f"Ball Pivoting 실패: {e}")
            return 0, None

    # Method 7: Convex Hull (참고용)
    def volume_convex_hull(self, cup_diameter_mm=90):
        """Convex Hull을 이용한 부피 계산 (상한값 참고용)"""
        try:
            points, scale_factor = self.create_point_cloud(cup_diameter_mm=cup_diameter_mm)
            
            # Convex Hull 생성
            hull = ConvexHull(points)
            
            # Trimesh 객체 생성
            mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
            
            volume = abs(mesh.volume)
            
            # Open3D 메시로 변환 (시각화용)
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)
            
            return volume, o3d_mesh
            
        except Exception as e:
            print(f"Convex Hull 실패: {e}")
            return 0, None

    # Method 8: Improved Cup-specific Volume Calculation
    def volume_cup_specific(self, cup_diameter_mm=90):
        """컵 형상에 특화된 개선된 부피 계산"""
        try:
            valid_mask = self.depth_data > 0
            pixel_diameter, center = self.find_cup_circle()
            scale_factor = cup_diameter_mm / pixel_diameter
            
            # 컵 반경
            radius_mm = cup_diameter_mm / 2
            
            # 방사형 섹터로 나누기
            n_sectors = 36  # 10도씩
            n_rings = 30    # 동심원
            
            # 각 링에서의 평균 깊이 계산
            ring_depths = []
            ring_radii = np.linspace(0, radius_mm, n_rings)
            
            for i in range(1, n_rings):
                r = ring_radii[i] / scale_factor  # 픽셀 단위로 변환
                depths_at_ring = []
                
                for angle in np.linspace(0, 2*np.pi, n_sectors, endpoint=False):
                    x = center[0] + r * np.cos(angle)
                    y = center[1] + r * np.sin(angle)
                    
                    # 보간을 사용하여 깊이값 추정
                    if 0 <= x < self.image_width-1 and 0 <= y < self.image_height-1:
                        # Bilinear interpolation
                        x0, y0 = int(x), int(y)
                        x1, y1 = min(x0+1, self.image_width-1), min(y0+1, self.image_height-1)
                        
                        dx, dy = x - x0, y - y0
                        
                        # 4개 점의 깊이값
                        d00 = self.depth_data[y0, x0]
                        d01 = self.depth_data[y0, x1]
                        d10 = self.depth_data[y1, x0]
                        d11 = self.depth_data[y1, x1]
                        
                        if d00 > 0 and d01 > 0 and d10 > 0 and d11 > 0:
                            # Bilinear interpolation
                            depth = (1-dx)*(1-dy)*d00 + dx*(1-dy)*d01 + (1-dx)*dy*d10 + dx*dy*d11
                            depths_at_ring.append(depth)
                
                if depths_at_ring:
                    ring_depths.append(np.median(depths_at_ring))  # 중앙값 사용
                else:
                    ring_depths.append(0)
            
            # 사다리꼴 공식으로 부피 계산
            volume = 0
            for i in range(len(ring_depths)-1):
                r1 = ring_radii[i+1]
                r0 = ring_radii[i]
                h1 = ring_depths[i] if i < len(ring_depths) else 0
                h2 = ring_depths[i+1] if i+1 < len(ring_depths) else 0
                
                # 원환의 부피
                dV = np.pi * (r1**2 - r0**2) * (h1 + h2) / 2
                volume += dV
            
            # 메시 생성 (시각화용)
            vertices = []
            faces = []
            
            for i, r in enumerate(ring_radii[1:]):
                for j, angle in enumerate(np.linspace(0, 2*np.pi, n_sectors, endpoint=False)):
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    z = ring_depths[i] if i < len(ring_depths) else 0
                    vertices.append([x, y, z])
            
            vertices = np.array(vertices)
            if len(vertices) > 3:
                tri = Delaunay(vertices[:, :2])
                mesh = trimesh.Trimesh(vertices=vertices, faces=tri.simplices)
                mesh = self.add_bottom_to_mesh_trimesh(mesh)
            else:
                mesh = None
            
            return volume, mesh
            
        except Exception as e:
            print(f"Cup-specific 계산 실패: {e}")
            return 0, None

    # Method 9: Voxel-based Volume
    def volume_voxel_based(self, cup_diameter_mm=90, voxel_size=1.0):
        """복셀 기반 부피 계산"""
        try:
            points, scale_factor = self.create_point_cloud(cup_diameter_mm=cup_diameter_mm)
            
            # 복셀 그리드 생성
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                voxel_size=voxel_size
            )
            
            # 복셀 개수 * 복셀 부피
            voxels = voxel_grid.get_voxels()
            n_voxels = len(voxels)
            volume = n_voxels * (voxel_size ** 3)
            
            # 메시 생성 (시각화용)
            # 각 복셀을 큐브로 표현
            mesh = o3d.geometry.TriangleMesh()
            for voxel in voxels:
                cube = o3d.geometry.TriangleMesh.create_box(
                    width=voxel_size, height=voxel_size, depth=voxel_size
                )
                cube.translate(
                    voxel.grid_index * voxel_size - [voxel_size/2, voxel_size/2, voxel_size/2]
                )
                mesh += cube
            
            return volume, mesh
            
        except Exception as e:
            print(f"Voxel-based 계산 실패: {e}")
            return 0, None

    def add_bottom_to_mesh_o3d(self, mesh, points):
        """Open3D 메시에 바닥면 추가"""
        try:
            vertices = np.asarray(mesh.vertices)
            
            # 바닥 높이 (최소 z값)
            min_z = np.min(points[:, 2])
            
            # 바닥 경계 점들 찾기
            boundary_points = []
            xy_points = points[:, :2]
            hull_2d = ConvexHull(xy_points)
            
            for idx in hull_2d.vertices:
                boundary_points.append([points[idx, 0], points[idx, 1], min_z])
            
            # 바닥 중심점
            center_x = np.mean([p[0] for p in boundary_points])
            center_y = np.mean([p[1] for p in boundary_points])
            center_point = [center_x, center_y, min_z]
            
            # 새 vertices 추가
            new_vertices = np.vstack([vertices, boundary_points, [center_point]])
            center_idx = len(new_vertices) - 1
            
            # 바닥 삼각형 생성
            triangles = np.asarray(mesh.triangles)
            new_triangles = []
            
            start_idx = len(vertices)
            for i in range(len(boundary_points)):
                next_i = (i + 1) % len(boundary_points)
                new_triangles.append([start_idx + i, start_idx + next_i, center_idx])
            
            # 메시 업데이트
            mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(
                np.vstack([triangles, new_triangles])
            )
            
            return mesh
        except:
            return mesh

    def add_bottom_to_mesh_trimesh(self, mesh):
        """Trimesh 메시에 바닥면 추가"""
        try:
            vertices = mesh.vertices
            min_z = np.min(vertices[:, 2])
            
            # XY 평면에서 경계 찾기
            xy_points = vertices[:, :2]
            hull_2d = ConvexHull(xy_points)
            
            # 바닥 점들 생성
            bottom_vertices = []
            for idx in hull_2d.vertices:
                bottom_vertices.append([vertices[idx, 0], vertices[idx, 1], min_z])
            
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
            
            return trimesh.Trimesh(vertices=all_vertices, faces=faces)
        except:
            return mesh

    def compare_all_methods(self, cup_diameter_mm=90):
        """모든 방법 비교"""
        print("=" * 80)
        print("Arducam Depth Camera - 모든 부피 계산 방법 비교")
        print("=" * 80)
        print(f"실제 부피: 385 mL")
        print(f"컵 직경 설정: {cup_diameter_mm} mm")
        print("=" * 80)
        
        methods = []
        volumes = []
        meshes = []
        
        # Method 1: 구분구적법
        print("\n1. 구분구적법 (Riemann Sum)...")
        vol1, mesh1 = self.estimate_cup_volume_improved(cup_diameter_mm)
        methods.append("Riemann Sum")
        volumes.append(vol1)
        meshes.append(mesh1)
        print(f"   부피: {vol1/1000:.2f} mL, 오차: {abs(vol1/1000 - 385)/385*100:.1f}%")
        
        # Method 2: Alpha Shape
        print("\n2. Alpha Shape...")
        vol2, mesh2 = self.volume_alpha_shape(cup_diameter_mm)
        methods.append("Alpha Shape")
        volumes.append(vol2)
        meshes.append(mesh2)
        if vol2 > 0:
            print(f"   부피: {vol2/1000:.2f} mL, 오차: {abs(vol2/1000 - 385)/385*100:.1f}%")
        
        # Method 3: Poisson
        print("\n3. Poisson Surface Reconstruction...")
        vol3, mesh3 = self.volume_poisson_reconstruction(cup_diameter_mm)
        methods.append("Poisson")
        volumes.append(vol3)
        meshes.append(mesh3)
        if vol3 > 0:
            print(f"   부피: {vol3/1000:.2f} mL, 오차: {abs(vol3/1000 - 385)/385*100:.1f}%")
        
        # Method 4: RBF
        print("\n4. RBF Interpolation...")
        vol4, mesh4 = self.volume_rbf_interpolation(cup_diameter_mm)
        methods.append("RBF")
        volumes.append(vol4)
        meshes.append(mesh4)
        if vol4 > 0:
            print(f"   부피: {vol4/1000:.2f} mL, 오차: {abs(vol4/1000 - 385)/385*100:.1f}%")
        
        # Method 5: Cylindrical
        print("\n5. Cylindrical Integration...")
        vol5, mesh5 = self.volume_cylindrical_integration(cup_diameter_mm)
        methods.append("Cylindrical")
        volumes.append(vol5)
        meshes.append(mesh5)
        if vol5 > 0:
            print(f"   부피: {vol5/1000:.2f} mL, 오차: {abs(vol5/1000 - 385)/385*100:.1f}%")
        
        # Method 6: Ball Pivoting
        print("\n6. Ball Pivoting Algorithm...")
        vol6, mesh6 = self.volume_ball_pivoting(cup_diameter_mm)
        methods.append("Ball Pivoting")
        volumes.append(vol6)
        meshes.append(mesh6)
        if vol6 > 0:
            print(f"   부피: {vol6/1000:.2f} mL, 오차: {abs(vol6/1000 - 385)/385*100:.1f}%")
        
        # Method 7: Convex Hull
        print("\n7. Convex Hull (상한값 참고)...")
        vol7, mesh7 = self.volume_convex_hull(cup_diameter_mm)
        methods.append("Convex Hull")
        volumes.append(vol7)
        meshes.append(mesh7)
        if vol7 > 0:
            print(f"   부피: {vol7/1000:.2f} mL, 오차: {abs(vol7/1000 - 385)/385*100:.1f}%")
        
        # Method 8: Cup-specific
        print("\n8. Cup-specific Method (개선된 원통형)...")
        vol8, mesh8 = self.volume_cup_specific(cup_diameter_mm)
        methods.append("Cup-specific")
        volumes.append(vol8)
        meshes.append(mesh8)
        if vol8 > 0:
            print(f"   부피: {vol8/1000:.2f} mL, 오차: {abs(vol8/1000 - 385)/385*100:.1f}%")
        
        # Method 9: Voxel-based
        print("\n9. Voxel-based Method...")
        vol9, mesh9 = self.volume_voxel_based(cup_diameter_mm, voxel_size=2.0)
        methods.append("Voxel-based")
        volumes.append(vol9)
        meshes.append(mesh9)
        if vol9 > 0:
            print(f"   부피: {vol9/1000:.2f} mL, 오차: {abs(vol9/1000 - 385)/385*100:.1f}%")
        
        # 결과 요약
        print("\n" + "=" * 80)
        print("결과 요약:")
        print("-" * 80)
        print(f"{'Method':<20} {'Volume (mL)':<15} {'Error (%)':<15} {'Status'}")
        print("-" * 80)
        
        for method, volume, mesh in zip(methods, volumes, meshes):
            if volume > 0:
                error = abs(volume/1000 - 385) / 385 * 100
                status = "✓" if mesh is not None else "Volume only"
                print(f"{method:<20} {volume/1000:<15.2f} {error:<15.1f} {status}")
            else:
                print(f"{method:<20} {'Failed':<15} {'-':<15} ✗")
        
        # 메시 비교 시각화
        print("\n메시 형상 비교 시각화 생성 중...")
        valid_meshes = [m for m in meshes if m is not None]
        valid_volumes = [v for i, v in enumerate(volumes) if meshes[i] is not None]
        valid_methods = [methods[i] for i in range(len(methods)) if meshes[i] is not None]
        
        if valid_meshes:
            self.plot_mesh_comparison(valid_meshes[:6], valid_volumes[:6], valid_methods[:6])  # 최대 6개만 표시
        
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
            # 부피 비교
            colors = ['blue' if abs(v - 385) / 385 < 0.1 else 'orange' for v in valid_volumes]
            bars = ax1.bar(range(len(valid_methods)), valid_volumes, color=colors)
            ax1.axhline(y=385, color='red', linestyle='--', label='실제 부피 (385 mL)')
            ax1.set_xticks(range(len(valid_methods)))
            ax1.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax1.set_ylabel('부피 (mL)')
            ax1.set_title('부피 계산 방법별 결과')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 값 표시
            for i, (bar, v) in enumerate(zip(bars, valid_volumes)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{v:.1f}', ha='center', va='bottom')
            
            # 오차율 비교
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
            
            # 값 표시
            for bar, e in zip(bars2, errors):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{e:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('volume_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def save_meshes(self, meshes, methods):
        """메시를 파일로 저장"""
        import os
        os.makedirs('meshes', exist_ok=True)
        
        for mesh, method in zip(meshes, methods):
            if mesh is not None:
                filename = f"meshes/{method.replace(' ', '_').lower()}.ply"
                try:
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        o3d.io.write_triangle_mesh(filename, mesh)
                    elif isinstance(mesh, trimesh.Trimesh):
                        mesh.export(filename)
                    print(f"메시 저장: {filename}")
                except Exception as e:
                    print(f"메시 저장 실패 ({method}): {e}")


if __name__ == "__main__":
    # 컴퓨터 객체 생성
    example = computeVolume('masked_depth.npy')
    
    # 모든 방법 비교 실행
    methods, volumes, meshes = example.compare_all_methods(cup_diameter_mm=90)
    
    # 메시 파일로 저장
    print("\n메시 파일 저장 중...")
    example.save_meshes(meshes, methods)
    
    print("\n완료!")