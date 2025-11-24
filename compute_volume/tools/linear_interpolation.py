import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt # [추가] 플로팅 및 선형 회귀용
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
class FindMask:
    def __init__(self, file_path, savedata=True):
        self.file_path = file_path
        self.rawData = np.load(file_path)
        self.savedata = savedata

    def findMask(self, shrink_pixels=10, show_debug_images=False): # [수정] 플래그 추가
        # depth 데이터를 0~255로 정규화
        self.rawData = self.rawData[20:-20,20:-20]
        norm_data = cv.normalize(self.rawData, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
        # Otsu 알고리즘으로 임계값 계산
        if show_debug_images:
            cv.imshow('Normalized Depth', norm_data)
        ret, otsu = cv.threshold(norm_data, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        edges = cv.Canny(norm_data, ret * 0.5, ret * 1.5)
        print(f"Otsu 임계값: {ret}")
        
        if show_debug_images:
            cv.imshow('Edges', edges)

        # 컨투어 찾기
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("오류: 컨투어를 찾을 수 없습니다.")
            if show_debug_images: cv.destroyAllWindows()
            return None, None # [수정] 오류 시 None 반환
            
        largest_contour = max(contours, key=cv.contourArea)
        mask = np.zeros_like(self.rawData, dtype=np.uint8)
        cv.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
        
        # ... (Erosion, 동심원 마스크 생성 로직은 동일) ...
        (cx, cy), radius = cv.minEnclosingCircle(largest_contour)
        cx, cy = int(cx), int(cy)
        mask_circle = np.zeros_like(self.rawData, dtype=np.uint8)
        reduced_radius = int(radius - shrink_pixels)
        cv.circle(mask_circle, (cx, cy), reduced_radius, 255, -1)
        final_mask = mask_circle
        
        # ===== 1. 최종 마스크 픽셀 지름 측정 및 디버그 이미지 생성 =====
        contours_final, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        debug_image = cv.cvtColor(norm_data, cv.COLOR_GRAY2BGR)

        if not contours_final:
            print("경고: 최종 마스크에서 컨투어를 찾을 수 없습니다. (지름: 0)")
            pixel_diameter = 0
            return None, None # [수정] 오류 시 None 반환
        else:
            largest_final_contour = max(contours_final, key=cv.contourArea)
            (cx_final, cy_final), radius_final = cv.minEnclosingCircle(largest_final_contour)
            center_final = (int(cx_final), int(cy_final))
            radius_final_int = int(radius_final)
            pixel_diameter = radius_final_int * 2
            
            if show_debug_images:
                cv.circle(debug_image, center_final, radius_final_int, (0, 0, 255), 1) 
                cv.circle(debug_image, center_final, 2, (0, 255, 0), -1)

        print(f"========================================")
        print(f"최종 마스크 픽셀 지름: {pixel_diameter} px")
        print(f"========================================")
        
        # 시각화
        if show_debug_images:
            comparison = np.hstack([mask, final_mask])
            cv.imshow('Original vs Shrinked Mask', comparison)
            cv.imshow('Diameter Debug', debug_image) 
            cv.waitKey(0) # 모든 이미지 표시 후 한 번만 대기
            cv.destroyAllWindows()
        
        # 데이터 마스킹 및 기준점 보정
        masked_data = cv.bitwise_and(self.rawData, self.rawData, mask=final_mask)
        sorted_depths = np.sort(masked_data[masked_data > 0])
        
        if len(sorted_depths) < 20:
            print("오류: 유효한 데이터가 너무 적습니다.")
            return None, None # [수정] 오류 시 None 반환
            
        lowest_20_depths = sorted_depths[:20]
        baseline_depth = np.median(lowest_20_depths) # 이것이 기준 깊이(거리)
        
        # ... (평균 깊이 계산 로직은 회귀 분석에 직접 사용되진 않으므로 생략 가능) ...
        # ... (단, baseline_depth는 반환해야 하므로 계산은 유지) ...

        print(f"기준 깊이(거리): {baseline_depth:.2f} mm")
        
        # ... (파일 저장 로직) ...
        
        # [수정] 계산된 핵심 데이터 반환
        return pixel_diameter, baseline_depth

if __name__ == "__main__":
    
    # --- 설정 값 ---
    shrink_val = 5
    target_directory = './example/perspective_data'
    REAL_DIAMETER_MM = 70.0 # 컵의 실제 지름 (mm)
    SHOW_IMAGES_PER_FILE = False # False로 설정해야 배치 처리 중단 없음
    # ----------------
    
    file_pattern = os.path.join(target_directory, "*.npy")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"경고: '{target_directory}'에서 .npy 파일을 찾을 수 없습니다.")
    
    print(f"총 {len(file_list)}개의 파일을 처리합니다...")
    
    # [추가] 데이터 수집용 리스트
    depth_data = []       # X축: 기준 깊이(거리)
    mm_px_ratio_data = [] # Y축: mm/px 비율
    
    for file_path in sorted(file_list): 
        print(f"\n========================================")
        print(f"Processing: {file_path}")
        print(f"========================================")
        try:
            example = FindMask(file_path, savedata=False) 
            # [수정] show_debug_images 플래그 전달 및 반환값 받기
            result = example.findMask(shrink_pixels=shrink_val, 
                                      show_debug_images=SHOW_IMAGES_PER_FILE)

            # [추가] 결과 처리
            if result and result != (None, None):
                pixel_diameter, baseline_depth = result
                
                if pixel_diameter > 0:
                    # mm/px 비율 계산
                    mm_pp = REAL_DIAMETER_MM / pixel_diameter
                    
                    # 데이터 저장
                    depth_data.append(baseline_depth)
                    mm_px_ratio_data.append(mm_pp)
                    
                    print(f"-> 결과 저장: Depth={baseline_depth:.2f} mm, mm/px={mm_pp:.4f}")
                else:
                    print("-> 유효한 측정값을 얻지 못했습니다 (픽셀 지름 0).")
            else:
                print("-> 유효한 측정값을 얻지 못했습니다 (None 반환).")
                
        except Exception as e:
            print(f"예상치 못한 오류 발생 ({file_path}): {e}")

    print("\n--- 모든 파일 처리 완료 ---")
    cv.destroyAllWindows()

    # ===== [추가] 선형 회귀 및 플로팅 =====
    
    if len(depth_data) < 2:
        print("데이터가 2개 미만이라 선형 보정식을 계산할 수 없습니다.")
    else:
        # 리스트를 numpy 배열로 변환
        x_depths = np.array(depth_data)
        y_ratios = np.array(mm_px_ratio_data)
        
        # 1차 선형 회귀 (y = mx + b)
        # m: 기울기, b: y절편
        m, b = np.polyfit(x_depths, y_ratios, 1)
        
        print("\n--- 📈 선형 보정식 결과 ---")
        print(f"mm/px = {m:.6f} * depth + {b:.6f}")
        print("----------------------------\n")
        
        # 플로팅
        plt.figure(figsize=(10, 6))
        
        # 1. 원본 데이터 산점도
        plt.scatter(x_depths, y_ratios, label='측정 데이터 (Measured Data)')
        
        # 2. 선형 회귀선
        # x축 범위에 맞는 회귀선 생성
        x_fit = np.linspace(np.min(x_depths), np.max(x_depths), 100)
        y_fit = m * x_fit + b
        
        plt.plot(x_fit, y_fit, color='red', 
                 label=f'선형 보정식 (Fit):\ny = {m:.4f}x + {b:.4f}')
        
        # 그래프 설정
        plt.xlabel("측정된 컵과의 거리 (Depth, mm)")
        plt.ylabel("mm / pixel 비율")
        plt.title("깊이(거리)에 따른 mm/px 비율 변화 (Perspective Effect)")
        plt.legend()
        plt.grid(True)
        
        # 
        print("결과 플롯을 화면에 표시합니다...")
        plt.show()