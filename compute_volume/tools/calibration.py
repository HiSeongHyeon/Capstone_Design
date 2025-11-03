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
    # def convert2binary(self, threshold):
    #     norm_data = self.normalize()
    #     binary_data = np.zeros_like(norm_data, dtype=np.uint8)
    #     binary_data[norm_data >= threshold] = 255
    #     # #비최대 억제 적용으로 노이즈 감소
    #     # kernel = np.ones((3,3), np.uint8)
    #     # binary_data = cv.dilate(binary_data, kernel, iterations=1)
    #     #.jpg 이미지 데이터로 변환
    #     cv.imwrite('cali_02.jpg', binary_data)
    #     print("Binary data saved to binary_data.jpg")
    #     return binary_data
    #임계하한, 상한 범위의 데이터를 0~255 그레이스케일 이미지로 변환
    def convert2grayscale(self, lower_bound, upper_bound):
        norm_data = self.normalize()
        gray_data = np.zeros_like(norm_data, dtype=np.uint8)
        #하한과 상한 사이의 값들을 0~255로 매핑
        mask = (norm_data >= lower_bound) & (norm_data <= upper_bound)
        gray_data[mask] = ((norm_data[mask] - lower_bound) * 255 / (upper_bound - lower_bound)).astype(np.uint8)
        #가우시안 블러링
        gray_data = cv.GaussianBlur(gray_data, (5, 5), 0)
        #.jpg 이미지 데이터로 변환
        cv.imwrite('grayscale_data.jpg', gray_data)
        print("Grayscale data saved to grayscale_data.jpg")
        return gray_data
    def can_calib(self, file_path):
        #이미지를 받아 캘리브레이션 패턴을 찾는지 확인해주는 함수
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        pattern_size = (7, 7)  # 체스보드 패턴의 내부 코너 수
        ret, corners = cv.findChessboardCorners(image, pattern_size, None)
        #캘리브레이션 패턴이 발견되었는지 여부와 코너 좌표 반환
        if ret:
            cv.drawChessboardCorners(image, pattern_size, corners, ret)
            cv.imshow('Calibration Pattern', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Calibration pattern not found.")
        return ret, corners

if __name__ == "__main__":
    calib = Calibration('./example/1103/calibration_04.npy')
    # calib.save_normalized('./example/normalized_cali_08.npy')
    # calib.convert2binary(328)
    calib.save_normalized('./example/1103/normalized_cali_04.npy')
    calib.convert2grayscale(340, 400)
    found, corners = calib.can_calib(f'grayscale_data.jpg')
    if found:
        print("Calibration pattern found.")