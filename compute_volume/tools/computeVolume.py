#구분구적법을 활용해 픽셀의 넓이를 계산하고 깊이값과 곱해 부피를 계산하는 클래스
import numpy as np
import cv2 as cv

class computeVolume:
    def __init__(self, depth_file_path, calibration_file_path=None):
        self.depth_data = np.load(depth_file_path)
        self.pixel_size =  90/71.3 # mm 단위, 기본값

        if calibration_file_path is not None:
            self.calibration_data = np.load(calibration_file_path)
            self.pixel_size = self.get_pixel_size(self.calibration_data)

    def get_pixel_size(self, calibration_data):
        #캘리브레이션 데이터를 이용해 픽셀 크기 계산
        return 90/71.3 #mm 단위(임시) 9cm 컵 지름

    def computeVolume(self):
        #영역 계산
        area = np.sum(self.depth_data > 0) * (self.pixel_size ** 2) # mm^2 단위
        average_depth = np.mean(self.depth_data[self.depth_data > 0])
        print(f"average_depth : {average_depth}")
        volume = np.sum(self.depth_data[self.depth_data > 0]) * (self.pixel_size ** 2) # mm^3 단위
        return area, volume

if __name__ == "__main__":
    example = computeVolume('masked_depth.npy')
    area, volume = example.computeVolume()
    print(f"Area: {area} mm^2")
    print(f"Volume: {volume/1000} mL") # cm^3 단위로 출력