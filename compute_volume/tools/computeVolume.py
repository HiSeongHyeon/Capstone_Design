#구분구적법을 활용해 픽셀의 넓이를 계산하고 깊이값과 곱해 부피를 계산하는 클래스
import numpy as np
import cv2 as cv

class computeVolume:
    def __init__(self, depth_file_path, calibration_file_path=None):
        self.depth_data = np.load(depth_file_path)
        self.pixel_size =  1#90/71.3 # mm 단위, 기본값

        if calibration_file_path is not None:
            self.calibration_data = np.load(calibration_file_path)
            self.pixel_size = self.get_pixel_size(self.calibration_data)

    def get_pixel_size(self, calibration_data):
        #캘리브레이션 데이터를 이용해 픽셀 크기 계산
        
        return 1#90/71.3 #mm 단위(임시) 9cm 컵 지름

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
# import numpy as np
# import cv2 as cv

# class computeVolume:
#     def __init__(self, depth_file_path, calibration_file_path=None):
#         self.depth_data = np.load(depth_file_path)
        
#         if calibration_file_path is not None:
#             calibration = np.load(calibration_file_path, allow_pickle=True).item()
#             self.mtx = calibration['mtx']
#             self.dist = calibration['dist']
            
#             # 왜곡 보정
#             self.depth_data = self.undistort_depth(self.depth_data)
#         else:
#             # 기본 카메라 행렬 (제공된 값)
#             self.mtx = np.array([[1.99323272e+03, 0.00000000e+00, 1.20578782e+02],
#                                   [0.00000000e+00, 3.10382446e+03, 1.03703125e+02],
#                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#             self.dist = np.array([[ -3.66790241e+01,  2.50492157e+04, -2.82306609e-01,  2.20440346e-02,
#    1.02634669e+02]])
            
#             # 왜곡 보정
#             self.depth_data = self.undistort_depth(self.depth_data)

#     def undistort_depth(self, depth_data):
#         """
#         depth 데이터의 왜곡을 보정합니다.
#         """
#         h, w = depth_data.shape
        
#         # 최적 카메라 행렬 계산
#         new_mtx, roi = cv.getOptimalNewCameraMatrix(
#             self.mtx, self.dist, (w, h), 1, (w, h)
#         )
        
#         # 왜곡 보정 맵 생성
#         mapx, mapy = cv.initUndistortRectifyMap(
#             self.mtx, self.dist, None, new_mtx, (w, h), cv.CV_32FC1
#         )
        
#         # depth 데이터 왜곡 보정 (nearest neighbor 보간 사용)
#         undistorted_depth = cv.remap(
#             depth_data, mapx, mapy, cv.INTER_NEAREST
#         )
        
#         # 업데이트된 카메라 행렬 저장
#         self.mtx = new_mtx
        
#         return undistorted_depth

#     def get_pixel_size_at_depth(self, depth_mm):
#         """
#         특정 깊이에서의 픽셀 크기를 계산합니다 (mm 단위).
        
#         카메라 행렬에서:
#         fx = mtx[0, 0] : x방향 초점거리 (픽셀)
#         fy = mtx[1, 1] : y방향 초점거리 (픽셀)
        
#         실제 크기 = (픽셀 크기 * 깊이) / 초점거리
#         """
#         fx = self.mtx[0, 0]
#         fy = self.mtx[1, 1]
        
#         # x, y 방향 픽셀 크기 계산
#         pixel_size_x = depth_mm / fx * 10  # mm/pixel
#         pixel_size_y = depth_mm / fy * 10  # mm/pixel

#         return pixel_size_x, pixel_size_y

#     def computeVolume(self):
#         """
#         구분구적법을 사용하여 부피를 계산합니다.
#         각 픽셀의 깊이에 따라 다른 픽셀 크기를 적용합니다.
#         """
#         # 유효한 깊이값만 선택
#         valid_mask = self.depth_data > 0
#         valid_depths = self.depth_data[valid_mask]
        
#         if len(valid_depths) == 0:
#             return 0, 0
        
#         # 평균 깊이 계산
#         average_depth = np.mean(valid_depths)
#         print(f"Average depth: {average_depth:.2f} mm")
        
#         # 각 픽셀에 대해 부피 계산
#         total_volume = 0
#         total_area = 0
        
#         # 방법 1: 각 픽셀마다 개별 계산 (더 정확)
#         for i in range(self.depth_data.shape[0]):
#             for j in range(self.depth_data.shape[1]):
#                 depth_value = self.depth_data[i, j]
                
#                 if depth_value > 0:
#                     # 해당 깊이에서의 픽셀 크기 계산
#                     pixel_size_x, pixel_size_y = self.get_pixel_size_at_depth(depth_value)
                    
#                     # 픽셀의 면적 (mm^2)
#                     pixel_area = pixel_size_x * pixel_size_y
                    
#                     # 부피 기여분 (mm^3)
#                     total_area += pixel_area
#                     total_volume += pixel_area * depth_value
        
#         return total_area, total_volume

#     def computeVolume_fast(self):
#         """
#         평균 깊이를 사용한 빠른 부피 계산 (근사값).
#         """
#         valid_mask = self.depth_data > 0
#         valid_depths = self.depth_data[valid_mask]
        
#         if len(valid_depths) == 0:
#             return 0, 0
        
#         # 평균 깊이 계산
#         average_depth = np.mean(valid_depths)
#         print(f"Average depth: {average_depth:.2f} mm")
        
#         # 평균 깊이에서의 픽셀 크기 계산
#         pixel_size_x, pixel_size_y = self.get_pixel_size_at_depth(average_depth)
#         pixel_area = pixel_size_x * pixel_size_y
        
#         print(f"Pixel size at average depth: {pixel_size_x:.4f} x {pixel_size_y:.4f} mm")
#         print(f"Pixel area: {pixel_area:.6f} mm^2")
        
#         # 면적 계산
#         total_area = np.sum(valid_mask) * pixel_area  # mm^2
        
#         # 부피 계산
#         total_volume = np.sum(valid_depths) * pixel_area  # mm^3
        
#         return total_area, total_volume


# if __name__ == "__main__":
#     # 캘리브레이션 데이터 저장 예시
#     # calibration_data = {
#     #     'mtx': mtx,
#     #     'dist': dist
#     # }
#     # np.save('calibration.npy', calibration_data)
    
#     # 사용 예시
#     example = computeVolume('masked_depth.npy')
    
#     print("=== 정확한 계산 (각 픽셀별) ===")
#     area, volume = example.computeVolume()
#     print(f"Area: {area:.2f} mm^2 ({area/100:.2f} cm^2)")
#     print(f"Volume: {volume/1000:.2f} mL")
    
#     print("\n=== 빠른 계산 (평균 깊이 사용) ===")
#     area_fast, volume_fast = example.computeVolume_fast()
#     print(f"Area: {area_fast:.2f} mm^2 ({area_fast/100:.2f} cm^2)")
#     print(f"Volume: {volume_fast/1000:.2f} mL")