# #FFT 변환 후 Phase 데이터에서 고주파 영역 마스킹 후 반환하는 코드

# import numpy as np
# import cv2 as cv

# class PhaseMask:    
#     def __init__(self, file_path):
#         self.phase_data = np.load(file_path)

#     def apply_mask(self):
#         # FFT 변환
#         f_transform = np.fft.fft2(self.phase_data)
#         f_transform_shifted = np.fft.fftshift(f_transform)

#         # 마스크 생성 (예: 중앙의 저주파 영역만 남기고 고주파 영역 제거)
#         rows, cols = self.phase_data.shape
#         crow, ccol = rows // 2, cols // 2
#         mask = np.zeros((rows, cols), dtype=np.uint8)
#         r = 30  # 마스크 반지름 (조절 가능)
#         cv.circle(mask, (ccol, crow), r, 1, -1)

#         # 마스크 적용
#         f_transform_shifted_masked = f_transform_shifted * mask

#         # 역 FFT 변환
#         f_ishift = np.fft.ifftshift(f_transform_shifted_masked)
#         img_back = np.fft.ifft2(f_ishift)
#         img_back = np.abs(img_back)

#         return img_back

# if __name__ == "__main__":
#     # 예시 데이터 로드 (실제 phase_data로 대체 필요)
#     # 여기서는 임의의 노이즈가 있는 이미지 생성
#     img = np.load('./calibration/calib_1.npy')
#     phase_mask = PhaseMask('./calibration/calib_1.npy')
#     masked_phase = phase_mask.apply_mask()
#     np.save('smoothed_phase.npy', masked_phase)
#     print("Masked phase data saved to smoothed_phase.npy")
    
#데이터를 3*3 가우시안 필터로 스무딩 후 저장하는 코드
import numpy as np
import cv2 as cv
class Smooth:
    def __init__(self, file_path):
        self.data = np.load(file_path)

    def smooth_data(self):
        # 3x3 가우시안 필터 적용
        smoothed = cv.GaussianBlur(self.data, (3, 3), 0)
        return smoothed

    def save_smoothed(self, save_path):
        smoothed_data = self.smooth_data()
        np.save(save_path, smoothed_data)
        print(f"Smoothed data saved to {save_path}")
if __name__ == "__main__":
    smoother = Smooth('./example/cali_08.npy')
    smoother.save_smoothed('./example/smoothed_cali_08.npy')