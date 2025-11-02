#depth 데이터를 이용해 컵의 림 부분을 찾아주고 마스크를 만들어주는 코드
#마스크를 이용해 컵 데이터만 추출한 후, 저장할 수 있게 플래그를 통해 선택하게 해주는 클래스

import numpy as np
import cv2 as cv

class FindMask:
    def __init__(self, file_path, savedata = True):
        self.rawData = np.load(file_path)
        self.savedata = savedata

    def findMask(self):
        #depth 데이터를 엣지로 추출, 비최대 억제를 이용해 노이즈 
        #depth 데이터(범위 0~4000)를 0~255로 정규화
        norm_data = cv.normalize(self.rawData, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        #otsu 알고리즘을 이용해 임계값 계산
        ret, otsu = cv.threshold(norm_data, -1, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
        edges = cv.Canny(norm_data, ret*0.5, ret)
        cv.imshow('Edges', edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

        #컵의 림 컨투어 찾기
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("컨투어를 찾을 수 없습니다.")
        #가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv.contourArea)
        #가장 큰 컨투어의 지름 측정
        #(x,y), radius = cv.minEnclosingCircle(largest_contour)
        #print(2*radius)
        #컨투어 내부만 남기는 마스크 생성
        mask = np.zeros_like(self.rawData, dtype=np.uint8)
        cv.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
        #데이터 마스킹
        masked_data = cv.bitwise_and(self.rawData, self.rawData, mask=mask)
        #가장 낮은 높이 집합을 기준으로 거리 보정
        #min_depth = np.min(masked_data[masked_data > 0])
        #masked_data[masked_data > 0] -= min_depth

        #중간값 사용 방법
        sorted_depths = np.sort(masked_data[masked_data > 0])
        
        # 3. 정렬된 값들 중에서 가장 작은 20개를 선택합니다.
        lowest_20_depths = sorted_depths[:20]
        
        # 4. 선택된 20개 값의 중간값(median)을 계산합니다.
        baseline_depth = np.median(lowest_20_depths)
        masked_data[masked_data > 0] -= baseline_depth

        if self.savedata:
            np.save('masked_depth.npy', masked_data)
            cv.imwrite('mask.png', mask)
            print("마스크와 마스킹된 데이터를 저장했습니다.")
        return mask, masked_data

if __name__ == "__main__":
    example = FindMask('cup_example2.npy')
    example.findMask()