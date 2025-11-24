import numpy as np
import cv2 as cv

class FindMask:
    def __init__(self, file_path, savedata=True):
        self.rawData = np.load(file_path)
        self.savedata = savedata

    # shrink_pixels: 이 값을 조절하여 림 두께만큼 깎아냅니다. (권장 시작값: 5~10)
    def findMask(self, shrink_pixels=8):
        # 1. 데이터 정규화 및 전처리
        # 슬라이싱은 데이터에 따라 필요 시 조정
        self.rawData = self.rawData[20:-20, 20:-20]
        norm_data = cv.normalize(self.rawData, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
        # 노이즈 제거를 위해 블러링을 약간 강하게 적용
        blurred = cv.GaussianBlur(norm_data, (11, 11), 0)
        
        # 2. 엣지 검출 (Otsu 자동 임계값)
        high_thresh, _ = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        edges = cv.Canny(blurred, low_thresh, high_thresh)
        
        # 3. 컨투어 찾기 (가장 바깥쪽 외경만 찾으면 됩니다)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("엣지(컨투어)를 찾지 못했습니다.")

        # 4. 외경 타원 피팅
        valid_contours = [cnt for cnt in contours if len(cnt) >= 5]
        if not valid_contours:
             raise ValueError("타원 피팅이 가능한 컨투어가 없습니다.")

        # 가장 면적이 큰 컨투어를 외경으로 간주
        target_contour = max(valid_contours, key=cv.contourArea)
        (cx, cy), (minor_axis, major_axis), angle = cv.fitEllipse(target_contour)
        
        print(f"외경 타원 검출: 중심({cx:.1f}, {cy:.1f}), 단축지름({minor_axis:.1f}), 장축지름({major_axis:.1f})")

        # 5. 마스크 생성 및 모폴로지 침식(Erosion) 연산 (핵심 변경)
        
        # 5-1. 외경 기준의 꽉 찬 마스크 생성
        mask_ellipse = np.zeros_like(self.rawData, dtype=np.uint8)
        # 주의: cv.ellipse는 (width, height)가 아닌 (major_axis, minor_axis) 전체 지름을 받습니다.
        cv.ellipse(mask_ellipse, ((cx, cy), (minor_axis, major_axis), angle), 255, -1)

        final_mask = mask_ellipse
        
        # 5-2. 모폴로지 침식 적용
        if shrink_pixels > 0:
            # 3x3 사각형 커널 사용 (일반적)
            kernel = np.ones((3, 3), np.uint8)
            
            # shrink_pixels 횟수만큼 마스크를 안쪽으로 깎아냅니다.
            # 이 횟수를 조절하여 두꺼운 림을 제거합니다.
            final_mask = cv.erode(mask_ellipse, kernel, iterations=shrink_pixels)
            print(f"모폴로지 침식 적용 완료 (강도: {shrink_pixels})")
        else:
             print("경고: shrink_pixels가 0입니다. 외경이 그대로 마스크로 사용됩니다.")

        
        # 시각화 (디버깅용)
        debug_img = cv.cvtColor(norm_data, cv.COLOR_GRAY2BGR)
        # 감지된 초기 외경 타원 (노란색)
        cv.ellipse(debug_img, ((cx, cy), (minor_axis, major_axis), angle), (0, 255, 255), 2)
        
        # 침식 후 최종 마스크 경계 (빨간색)
        # 최종 마스크에서 다시 컨투어를 추출해 그립니다.
        mask_contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(debug_img, mask_contours, -1, (0, 0, 255), 2)
        
        cv.imshow('Erosion Result (Yellow: Outer, Red: Inner)', debug_img)
        # 아무 키나 누르면 창이 닫히고 다음 단계로 진행합니다.
        cv.waitKey(0)
        cv.destroyAllWindows()

        # 6. 후처리 (데이터 마스킹 및 저장)
        masked_data = cv.bitwise_and(self.rawData, self.rawData, mask=final_mask)
        
        # 침식으로 인해 유효 데이터가 사라졌는지 확인
        valid_pixels = masked_data[masked_data > 0]
        if len(valid_pixels) < 20:
             raise ValueError("유효한 데이터가 너무 적습니다. shrink_pixels 값을 줄여보세요.")
            
        baseline_depth = np.median(np.sort(valid_pixels)[:20])
        masked_data[masked_data < 0] = 0 # 음수 값 제거

        if self.savedata:
            np.save('masked_depth.npy', masked_data)
            cv.imwrite('mask.png', final_mask)
            print("마스크 및 마스킹된 데이터 저장 완료.")
        
        return final_mask, masked_data, baseline_depth

if __name__ == "__main__":
    # 파일 경로를 실제 경로로 수정해주세요.
    file_path = 'C:/Users/rngyq/Documents/Capstone_Design/compute_volume/data/567_tumbler.npy'
    example = FindMask(file_path)
    
    # 핵심 튜닝 포인트: shrink_pixels
    # 이 값을 조절하여 빨간색 선이 림 안쪽으로 들어가도록 만드세요.
    # 림이 두꺼워 보이므로 8 정도로 시작해서 결과를 보고 증감시킵니다.
    example.findMask(shrink_pixels=1)