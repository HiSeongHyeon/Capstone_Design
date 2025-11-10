"""
컵 부피 측정 프로그램
1. 컵 지름을 90mm로 가정하고 스케일 보정 후 부피 계산 (방법 1)
2. 캘리브레이션 정보만 사용하여 부피 계산 (방법 2)
"""

import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class computeVolume:
    def __init__(self, depth_file_path):
        """
        Parameters:
        - depth_file_path: depth 데이터 파일 경로
        - calibration_file_path: 캘리브레이션 파일 경로
        """
        self.depth_data = np.load(depth_file_path)
        self.image_height, self.image_width = self.depth_data.shape
        
        # 실제 캘리브레이션 결과 사용
        self.mtx = np.array([[190.46873334,   0.,         120.00074471],
                            [  0.,         191.201416,    90.00851171],
                            [  0.,           0.,           1.        ]])
        self.dist = np.array([[ 4.93083651e-01, -1.25632226e+00,  2.28374174e-03, -1.58899167e-05,
   5.29625368e-01]])  # 왜곡 계수 0으로 설정
            
        # 왜곡 보정
        self.depth_data_undistorted = self.undistort_depth(self.depth_data)
        
        print(f"이미지 크기: {self.image_width} x {self.image_height}")
        print(f"초점거리 fx: {self.mtx[0,0]:.2f}, fy: {self.mtx[1,1]:.2f}")
        print(f"주점 cx: {self.mtx[0,2]:.2f}, cy: {self.mtx[1,2]:.2f}")

    def undistort_depth(self, depth_data):
        """depth 데이터의 왜곡을 보정합니다."""
        # Depth 데이터는 왜곡 보정하지 않음
        # (왜곡 보정이 3D 재구성을 오히려 왜곡시킬 수 있음)
        print("Depth 데이터는 왜곡 보정을 건너뜁니다.")
        self.mtx_undistorted = self.mtx.copy()
        return depth_data.copy()

    def find_cup_circle_in_3d(self, x_mm, y_mm):
        """3D 좌표에서 컵의 원형 경계를 찾아 정확한 지름을 계산"""
        
        # (x_mm, y_mm) 2D 점들로 변환
        points = np.column_stack((x_mm, y_mm))
        
        # 3D 포인트 클라우드의 Top-down 뷰(X-Y 평면)에서
        # 모든 점을 포함하는 최소 외접원을 찾습니다.
        # 이 원의 경계가 컵의 림(rim)에 해당합니다.
        points_2d = points.astype(np.float32)
        
        if points_2d.shape[0] < 3:
            print("Warning: 원을 피팅하기에 포인트가 부족합니다.")
            return 0, (0, 0)
            
        (cx, cy), radius = cv.minEnclosingCircle(points_2d)
        
        return radius * 2, (cx, cy)  # 지름과 중심 반환

    def plot_3d_with_measurements(self, scale_factor=None, method_name=""):
        """
        3D 플롯과 측정값을 표시
        
        Parameters:
        - scale_factor: mm/pixel 변환 계수 (방법 1용, None이면 방법 4)
        - method_name: 방법 이름
        """
        # 왜곡 보정된 데이터 사용
        valid_mask = self.depth_data_undistorted > 0
        
        # 3D 좌표 생성
        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data_undistorted[valid_mask]
        
        if scale_factor is not None:
            # 방법 1: 스케일 팩터 사용
            x_mm = x_indices * scale_factor
            y_mm = y_indices * scale_factor
            z_mm = z_values
            
            # 컵 중심 계산
            pixel_diameter, (cx, cy) = self.find_cup_circle()
            center_x_mm = cx * scale_factor
            center_y_mm = cy * scale_factor
            
            # 중심 기준으로 재배치
            x_mm = x_mm - center_x_mm
            y_mm = y_mm - center_y_mm
            
            # 직경 계산
            diameter_mm = pixel_diameter * scale_factor
            
            # 높이 계산
            height_mm = np.max(z_mm) - np.min(z_mm)
            
        else:
            # 방법 4: 캘리브레이션만 사용 (수정된 버전)
            fx = self.mtx[0, 0]
            fy = self.mtx[1, 1]
            cx = self.mtx[0, 2]
            cy = self.mtx[1, 2]
            
            # 각 점의 실제 깊이를 사용하여 3D 좌표 변환 (수정된 부분)
            x_mm = (x_indices - cx) * z_values / fx
            y_mm = (y_indices - cy) * z_values / fy
            z_mm = z_values
            
            # 3D 공간에서 원 피팅하여 직경 계산 (수정된 부분)
            diameter_mm, center_3d = self.find_cup_circle_in_3d(x_mm, y_mm)
            
            # 중심으로 이동
            x_mm = x_mm - center_3d[0]
            y_mm = y_mm - center_3d[1]
            
            # 높이 계산
            height_mm = np.max(z_mm) - np.min(z_mm)
        
        # 서브플롯 생성
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 3D 플롯
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(x_mm, y_mm, z_mm, c=z_mm, cmap='viridis', s=1)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Depth (mm)')
        ax1.set_title(f'3D Point Cloud - {method_name}')
        plt.colorbar(scatter, ax=ax1, label='Depth (mm)')
        ax1.view_init(elev=30, azim=45)
        
        # 축 스케일 동일하게 설정
        max_range = np.array([x_mm.max()-x_mm.min(), 
                             y_mm.max()-y_mm.min(),
                             z_mm.max()-z_mm.min()]).max() / 2.0
        mid_x = (x_mm.max()+x_mm.min()) * 0.5
        mid_y = (y_mm.max()+y_mm.min()) * 0.5
        mid_z = (z_mm.max()+z_mm.min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 2. 상단 뷰 (X-Y 평면)
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(x_mm, y_mm, c=z_mm, cmap='viridis', s=2)
        
        # 원 그리기 (직경 표시)
        circle = plt.Circle((0, 0), diameter_mm/2, fill=False, color='red', linewidth=2, label=f'Diameter: {diameter_mm:.1f} mm')
        ax2.add_patch(circle)
        
        # 직경 선 그리기
        ax2.plot([-diameter_mm/2, diameter_mm/2], [0, 0], 'r--', linewidth=2)
        ax2.text(0, -diameter_mm/2 - 10, f'Diameter: {diameter_mm:.1f} mm', 
                ha='center', fontsize=12, color='red', weight='bold')
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top View (X-Y Plane)')
        ax2.set_aspect('equal')
        ax2.grid(True)
        plt.colorbar(scatter2, ax=ax2, label='Depth (mm)')
        
        # 3. 측면 뷰 (X-Z 평면)
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(x_mm, z_mm, c=z_mm, cmap='viridis', s=2)
        
        # 높이 선 그리기
        ax3.axhline(y=np.min(z_mm), color='g', linestyle='--', alpha=0.5)
        ax3.axhline(y=np.max(z_mm), color='g', linestyle='--', alpha=0.5)
        ax3.text(0, (np.max(z_mm) + np.min(z_mm))/2, f'Height: {height_mm:.1f} mm', 
                fontsize=12, color='red', weight='bold')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Depth (mm)')
        ax3.set_title('Side View (X-Z Plane)')
        ax3.grid(True)
        plt.colorbar(scatter3, ax=ax3, label='Depth (mm)')
        
        # 4. 깊이 히트맵
        ax4 = fig.add_subplot(2, 3, 4)
        depth_display = self.depth_data_undistorted.copy()
        depth_display[~valid_mask] = np.nan
        im = ax4.imshow(depth_display, cmap='viridis')
        ax4.set_title('Depth Heatmap (Undistorted)')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax4, label='Depth (mm)')
        
        # 5. 깊이 분포 히스토그램
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(z_values, bins=50, color='steelblue', edgecolor='black')
        ax5.axvline(np.mean(z_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(z_values):.1f} mm')
        ax5.axvline(np.max(z_values), color='green', linestyle='--', linewidth=2, label=f'Max: {np.max(z_values):.1f} mm')
        ax5.set_xlabel('Depth (mm)')
        ax5.set_ylabel('Pixel Count')
        ax5.set_title('Depth Distribution')
        ax5.legend()
        ax5.grid(True)
        
        # 6. 측정값 요약
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        === {method_name} Result ===
        
        Diameter: {diameter_mm:.2f} mm
        Height: {height_mm:.2f} mm
        
        Mean Depth: {np.mean(z_values):.2f} mm
        Min Depth: {np.min(z_values):.2f} mm
        Max Depth: {np.max(z_values):.2f} mm
        
        Valid Pixels: {len(z_values)}
        Total Pixels: {self.image_width * self.image_height}
        Coverage: {len(z_values)/(self.image_width * self.image_height)*100:.1f}%
        """
        
        if scale_factor is not None:
            pixel_diameter, _ = self.find_cup_circle()
            summary_text += f"\nScale Factor: {scale_factor:.4f} mm/pixel"
            summary_text += f"\nPixel Diameter: {pixel_diameter:.1f} pixels"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'volume_analysis_{method_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return diameter_mm, height_mm

    def find_cup_circle(self):
        """컵의 원형 경계를 찾아 정확한 지름을 계산 (픽셀 좌표에서)"""
        valid_mask = (self.depth_data_undistorted > 0).astype(np.uint8)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_CLOSE, kernel)
        valid_mask = cv.morphologyEx(valid_mask, cv.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv.findContours(valid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어 선택 (컵)
            largest_contour = max(contours, key=cv.contourArea)
            
            # 타원 피팅 (더 정확한 방법)
            if len(largest_contour) >= 5:
                ellipse = cv.fitEllipse(largest_contour)
                center, (width, height), angle = ellipse
                # 평균 지름 사용
                diameter_pixels = (width + height) / 2
            else:
                (x, y), radius = cv.minEnclosingCircle(largest_contour)
                diameter_pixels = radius * 2
                center = (x, y)
                
            return diameter_pixels, center
        
        # 폴백: 바운딩 박스 사용
        y_indices, x_indices = np.where(valid_mask)
        diameter_pixels = max(x_indices.max() - x_indices.min(), 
                             y_indices.max() - y_indices.min())
        center = ((x_indices.max() + x_indices.min())/2, 
                 (y_indices.max() + y_indices.min())/2)
        
        return diameter_pixels, center

    def estimate_cup_volume_improved(self, cup_diameter_mm=90):
        """
        개선된 컵 부피 계산 방법 (Method 1: Scale Correction)
        """
        valid_mask = self.depth_data_undistorted > 0
        
        # 컵의 정확한 픽셀 지름 측정
        pixel_diameter, center = self.find_cup_circle()
        
        # 스케일 팩터 계산
        scale_factor = cup_diameter_mm / pixel_diameter
        
        print(f"검출된 컵 픽셀 지름: {pixel_diameter:.1f} pixels")
        print(f"스케일 팩터: {scale_factor:.4f} mm/pixel")
        print(f"컵 중심: ({center[0]:.1f}, {center[1]:.1f})")
        
        # 각 픽셀의 실제 면적
        pixel_area = scale_factor * scale_factor  # mm^2
        
# 깊이값 통계
        valid_depths = self.depth_data_undistorted[valid_mask]
        
        # [추가] 베이스라인 보정 (상대 깊이 계산)
        if len(valid_depths) > 10:
            sorted_depths = np.sort(valid_depths)
            # 가장 가까운 10번째 픽셀을 컵의 가장자리(baseline)로 설정
            baseline_depth = sorted_depths[10] 
        else:
            baseline_depth = 0
            
        internal_depths = valid_depths - baseline_depth
        internal_depths[internal_depths < 0] = 0 # 0 미만은 0 (컵 가장자리)
        
        print(f"유효 픽셀 수: {len(valid_depths)}")
        print(f"베이스라인 깊이 (컵 가장자리): {baseline_depth:.2f} mm")
        print(f"최대 내부 깊이 (컵 높이): {np.max(internal_depths):.2f} mm")
        
        # 부피 계산 (상대 깊이로 변경)
        total_volume = np.sum(internal_depths) * pixel_area # <--- internal_depths 사용
        
        # 3D 플롯 (시각화는 여전히 절대 좌표 사용)
        print("\n3D 시각화 생성 중...")
        # ... (plot_3d_with_measurements 호출은 동일) ...
        
        return total_volume
        
    def compute_volume_calibration_only(self):
        """
        캘리브레이션 정보만 사용하여 부피 계산 (Method 4: Calibration Only)
        왜곡 보정 후 구분구적법 수행
        """
        valid_mask = self.depth_data_undistorted > 0
        
        if not valid_mask.any():
            return 0
        
        fx = self.mtx[0, 0]
        fy = self.mtx[1, 1]
        cx = self.mtx[0, 2]
        cy = self.mtx[1, 2]
        
        print(f"카메라 내부 파라미터:")
        print(f"  fx: {fx:.2f}, fy: {fy:.2f}")
        print(f"  cx: {cx:.2f}, cy: {cy:.2f}")
        
        # 구분구적법: 각 픽셀의 3D 부피 계산
        total_volume = 0
        
        # 벡터화된 계산으로 속도 개선
        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data_undistorted[valid_mask]
        sorted_depths = np.sort(z_values[z_values > 0])
        if len(sorted_depths) > 10:
            lowest_10 = sorted_depths[10]
        baseline_depth = lowest_10 if len(sorted_depths) > 10 else 0
        z_values = z_values - baseline_depth
        z_values[z_values < 0] = 0  # 음수 제거
        
        # 각 픽셀의 실제 면적 계산 (해당 깊이에서)
        # 픽셀 (x,y)가 깊이 z에서 차지하는 실제 면적
        pixel_areas = (z_values / fx) * (z_values / fy)  # mm^2
        
        # 부피 = 면적 × 깊이
        pixel_volumes = pixel_areas * z_values  # mm^3
        
        # 총 부피
        total_volume = np.sum(pixel_volumes)
        
        valid_pixels = np.sum(valid_mask)
        avg_depth = np.mean(z_values)
        
        print(f"유효 픽셀 수: {valid_pixels}")
        print(f"평균 깊이: {avg_depth:.2f} mm")
        print(f"평균 픽셀 면적: {np.mean(pixel_areas):.4f} mm^2")
        
        # 3D 플롯 및 측정값 표시
        print("\n3D 시각화 생성 중...")
        diameter, height = self.plot_3d_with_measurements(scale_factor=None, 
                                                           method_name="Method 4 (Calibration)")
        
        return total_volume

    def compute_volume_simple_integration(self):
        """
        단순 적분 방법: 왜곡 보정 후 직접 부피 계산
        """
        valid_mask = self.depth_data_undistorted > 0
        
        if not valid_mask.any():
            return 0
        
        # 카메라 파라미터
        fx = self.mtx[0, 0]
        fy = self.mtx[1, 1]
        
        # 각 픽셀에서의 부피 요소 계산
        total_volume = 0
        
        for i in range(self.depth_data_undistorted.shape[0]):
            for j in range(self.depth_data_undistorted.shape[1]):
                z = self.depth_data_undistorted[i, j]
                sorted_z =np.sort(self.depth_data_undistorted[valid_mask])
                if len(sorted_z) > 10:
                    lowest_10 = sorted_z[10]
                baseline_depth = lowest_10 if len(sorted_z) > 10 else 0
                z = z - baseline_depth
                if z < 0:
                    z = 0
                if z > 0:
                    # 해당 깊이에서 1픽셀의 실제 크기
                    dx = z / fx  # mm
                    dy = z / fy  # mm
                    
                    # 부피 요소 dV = dx * dy * dz
                    # 여기서 dz는 깊이값 자체
                    dV = dx * dy * z
                    total_volume += dV
        
        return total_volume


if __name__ == "__main__":
    print("=" * 60)
    print("Arducam Depth Camera Volume Calculation (개선된 버전)")
    print("=" * 60)
    
    # 컴퓨터 객체 생성
    example = computeVolume('masked_depth.npy')
    
    print("\n=== Method 1: Scale Correction ===")
    volume1 = example.estimate_cup_volume_improved(cup_diameter_mm=90)
    print(f"Calculated Volume: {volume1/1000:.2f} mL")
    
    print("\n=== Method 4: Calibration Only (수정됨) ===")
    volume4 = example.compute_volume_calibration_only()
    print(f"Calculated Volume: {volume4/1000:.2f} mL")
    
    print("\n=== Method 5: Simple Integration ===")
    volume5 = example.compute_volume_simple_integration()
    print(f"Calculated Volume: {volume5/1000:.2f} mL")
    
    print("\n" + "=" * 60)
    print("실제 부피: 385 mL")
    print(f"오차율:")
    print(f"  Method 1: {abs(volume1/1000 - 385)/385*100:.1f}%")
    print(f"  Method 4: {abs(volume4/1000 - 385)/385*100:.1f}%")
    print(f"  Method 5: {abs(volume5/1000 - 385)/385*100:.1f}%")
    print("=" * 60)