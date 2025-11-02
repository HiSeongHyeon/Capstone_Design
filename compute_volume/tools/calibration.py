# depth 데이터를 가공하고 캘리브레이션을 수행해 보정 행렬을 계산하는 모듈
import numpy as np
import cv2 as cv
class Calibration:
    def __init__(self, file_path):
        self.data = np.load(file_path)

    def normalize(self):
        #depth_data 0~4000 범위를 0~255로 정규화
        #norm_data = cv.normalize(self.data, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        norm_data = self.data
        #depth_map에서 하한값의 중앙값을 기준으로 10cm까지를 정규화
        sorted_depth = np.sort(norm_data[norm_data > 0])
        lowest_10 = sorted_depth[10]
        #lowest_50 보다 작은 값들은 삭제, 100*255/4000 값보다 큰 값들도 삭제
        norm_data[norm_data < lowest_10] = 0
        norm_data[norm_data > (lowest_10 + 300)] = 0
        #norm_data[norm_data > (100*255/4000)] = 0
        return norm_data#*4000/255
    def save_normalized(self, save_path):
        norm_data = self.normalize()
        np.save(save_path, norm_data)
        print(f"Normalized data saved to {save_path}")
    
    #임계값 이하는 검은색, 이상은 흰색으로 변환
    def convert2binary(self, threshold):
        norm_data = self.normalize()
        binary_data = np.zeros_like(norm_data, dtype=np.uint8)
        binary_data[norm_data >= threshold] = 255
        #.jpg 이미지 데이터로 변환
        cv.imwrite('cali_08.jpg', binary_data)
        print("Binary data saved to binary_data.jpg")
        return binary_data
if __name__ == "__main__":
    calib = Calibration('./example/smoothed_cali_08.npy')
    calib.save_normalized('./example/normalized_cali_08.npy')
    calib.convert2binary(328)