import numpy as np
import cv2 as cv

class FindMask:
    def __init__(self, file_path, savedata=True):
        self.rawData = np.load(file_path)
        self.savedata = savedata

    def findMask(self, shrink_pixels=10):
        # depth 데이터를 0~255로 정규화
        norm_data = cv.normalize(self.rawData, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
        # Otsu 알고리즘으로 임계값 계산
        ret, otsu = cv.threshold(norm_data, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        edges = cv.Canny(norm_data, ret * 0.5, ret)
        
        cv.imshow('Edges', edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # 컨투어 찾기
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("컨투어를 찾을 수 없습니다.")
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv.contourArea)
        
        # 원본 마스크 생성
        mask = np.zeros_like(self.rawData, dtype=np.uint8)
        cv.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
        
        # ===== 방법 1: Erosion으로 마스크 축소 =====
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (shrink_pixels*2+1, shrink_pixels*2+1))
        mask_shrinked = cv.erode(mask, kernel, iterations=1)
        
        # ===== 방법 2: 동심원 마스크 (더 정확한 원형 축소) =====
        # 최소 외접원 찾기
        (cx, cy), radius = cv.minEnclosingCircle(largest_contour)
        cx, cy = int(cx), int(cy)
        
        # 반지름을 줄인 동심원 마스크 생성
        mask_circle = np.zeros_like(self.rawData, dtype=np.uint8)
        reduced_radius = int(radius - shrink_pixels)  # 반지름 축소
        cv.circle(mask_circle, (cx, cy), reduced_radius, 255, -1)
        
        # 두 마스크 중 선택 (동심원 추천)
        final_mask = mask_circle  # 또는 mask_shrinked
        
        # 시각화
        comparison = np.hstack([mask, final_mask])
        cv.imshow('Original vs Shrinked Mask', comparison)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # 데이터 마스킹
        masked_data = cv.bitwise_and(self.rawData, self.rawData, mask=final_mask)
        
        # 기준점 보정 (가장 낮은 20개 값의 중간값)
        sorted_depths = np.sort(masked_data[masked_data > 0])
        if len(sorted_depths) < 20:
            raise ValueError("유효한 데이터가 너무 적습니다.")
        lowest_20_depths = sorted_depths[:20]
        baseline_depth = np.median(lowest_20_depths)
        masked_data[masked_data > 0] -= baseline_depth

        if self.savedata:
            np.save('masked_depth.npy', masked_data)
            cv.imwrite('mask.png', final_mask)
            print(f"마스크와 마스킹된 데이터를 저장했습니다. (축소: {shrink_pixels}px)")
        
        return final_mask, masked_data

if __name__ == "__main__":
    # shrink_pixels 값으로 축소 정도 조절 (픽셀 단위)
    example = FindMask('./example/smoothed_385_cup.npy')
    example.findMask(shrink_pixels=3)  # 15픽셀 만큼 안쪽으로 축소