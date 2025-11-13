import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

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
        #depth 스케일링
        self.depth_data = self.depth_data*3/4
        self.image_height, self.image_width = self.depth_data.shape

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

    def plot_3d_with_measurements(self, scale_factor=None, method_name=""):
        """
        3D 플롯과 측정값을 표시
        
        Parameters:
        - scale_factor: mm/pixel 변환 계수 (방법 1용, None이면 방법 4)
        - method_name: 방법 이름
        """
        valid_mask = self.depth_data > 0
        
        # 3D 좌표 생성
        y_indices, x_indices = np.where(valid_mask)
        z_values = self.depth_data[valid_mask]
        
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
        height_mm = np.max(z_mm)
        
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
        max_depth_idx = np.argmax(z_mm)
        x_at_max = x_mm[max_depth_idx]
        ax3.plot([x_at_max, x_at_max], [0, height_mm], 'r--', linewidth=2)
        ax3.text(x_at_max + 5, height_mm/2, f'Height: {height_mm:.1f} mm', 
                fontsize=12, color='red', weight='bold')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Depth (mm)')
        ax3.set_title('Side View (X-Z Plane)')
        ax3.grid(True)
        plt.colorbar(scatter3, ax=ax3, label='Depth (mm)')
        
        # 4. 깊이 히트맵
        ax4 = fig.add_subplot(2, 3, 4)
        depth_display = self.depth_data.copy()
        depth_display[~valid_mask] = np.nan
        im = ax4.imshow(depth_display, cmap='viridis')
        ax4.set_title('Depth Heatmap')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax4, label='Depth (mm)')
        
        # 컵 윤곽 표시
        pixel_diameter, (cx_pix, cy_pix) = self.find_cup_circle()
        circle_pix = plt.Circle((cx_pix, cy_pix), pixel_diameter/2, fill=False, color='red', linewidth=2)
        ax4.add_patch(circle_pix)
        
        
        # 6. 측정값 요약
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        === {method_name} Result ===
        
        Diameter: {diameter_mm:.2f} mm
        Height (Max Depth): {height_mm:.2f} mm
        
        Mean Depth: {np.mean(z_values):.2f} mm
        Min Depth: {np.min(z_values):.2f} mm
        Max Depth: {np.max(z_values):.2f} mm
        
        Valid Pixels: {len(z_values)}
        Total Pixels: {self.image_width * self.image_height}
        """
        
        if scale_factor is not None:
            summary_text += f"\nScale Factor: {scale_factor:.4f} mm/pixel"
            summary_text += f"\nPixel Diameter: {pixel_diameter:.1f} pixels"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'volume_analysis_{method_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return diameter_mm, height_mm

    def estimate_cup_volume_improved(self, cup_diameter_mm=90):
        """
        개선된 컵 부피 계산 방법
        """
        valid_mask = self.depth_data > 0
        
        # 컵의 정확한 픽셀 지름 측정
        pixel_diameter, center = self.find_cup_circle()
        
        # 스케일 팩터 계산
        scale_factor = cup_diameter_mm / pixel_diameter
        #scale_factor = 1.3488 #mm/px
        
        print(f"검출된 컵 픽셀 지름: {pixel_diameter:.1f} pixels")
        print(f"스케일 팩터: {scale_factor:.4f} mm/pixel")
        print(f"컵 중심: ({center[0]:.1f}, {center[1]:.1f})")
        
        # 각 픽셀의 실제 면적
        pixel_area = scale_factor * scale_factor  # mm^2
        
        # 깊이값 통계
        valid_depths = self.depth_data[valid_mask]
        print(f"유효 픽셀 수: {len(valid_depths)}")
        print(f"평균 깊이: {np.mean(valid_depths):.2f} mm")
        print(f"최대 깊이: {np.max(valid_depths):.2f} mm")
        
        # 부피 계산
        total_volume = np.sum(valid_depths) * pixel_area
        
        # 3D 플롯 및 측정값 표시
        print("\n3D 시각화 생성 중...")
        diameter, height = self.plot_3d_with_measurements(scale_factor=scale_factor, 
                                                           method_name="Method 1 (Scale Correction)")
        
        return total_volume


if __name__ == "__main__":
    print("=" * 60)
    print("Arducam Depth Camera Volume Calculation")
    print("=" * 60)
    
    # 컴퓨터 객체 생성
    example = computeVolume('masked_depth.npy')
    
    print("\n=== Method 1: Scale Correction ===")
    volume1 = example.estimate_cup_volume_improved(cup_diameter_mm=90)
    print(f"Calculated Volume: {volume1/1000:.2f} mL")
    
    print("\n" + "=" * 60)
    print("Actual Volume: 385 mL")
    print(f"Error Rate:")
    print(f"  Method 1: {abs(volume1/1000 - 385)/385*100:.1f}%")