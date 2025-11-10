import numpy as np
import cv2 as cv

class Calibration:
    # 1. __init__ 수정: .npy 파일이 아닐 경우 self.data를 None으로 설정
    def __init__(self, file_path):
        if file_path.endswith('.npy'):
            self.data = np.load(file_path)
        else:
            self.data = None
            print(f"Info: Initialized with non-npy file. '{file_path}'. Assuming method-only use.")

    def normalize(self):
        # normalize 메서드는 self.data가 .npy일 때만 호출되어야 함
        if self.data is None:
            print("Error: normalize() called on an object initialized with a non-npy file.")
            return np.array([]) # 빈 배열 반환
            
        # depth_data 0~4000 범위를 0~255로 정규화
        # norm_data = cv.normalize(self.data, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        norm_data = self.data.copy() # 원본 데이터 수정을 방지하기 위해 .copy() 사용
        # depth_map에서 하한값의 중앙값을 기준으로 10cm까지를 정규화
        sorted_depth = np.sort(norm_data[norm_data > 0])
        
        # 데이터가 충분히 있는지 확인
        if len(sorted_depth) > 10:
            lowest_10 = sorted_depth[10]
        else:
            lowest_10 = 0 # 데이터가 부족할 경우 기본값 사용
            
        # lowest_50 보다 작은 값들은 삭제, 100*255/4000 값보다 큰 값들도 삭제
        #norm_data[norm_data < lowest_10] = 0
        #norm_data[norm_data > (lowest_10 + 300)] = 0
        # norm_data[norm_data > (100*255/4000)] = 0
        return norm_data#*4000/255
    
    def save_normalized(self, save_path):
        norm_data = self.normalize()
        if norm_data.size == 0: return # normalize 실패 시 중단
        
        np.save(save_path, norm_data)
        print(f"Normalized data saved to {save_path}")

    def convert2grayscale(self, lower_bound, upper_bound, save_path):
        norm_data = self.normalize()
        if norm_data.size == 0: return None # normalize 실패 시 중단

        gray_data = np.zeros_like(norm_data, dtype=np.uint8)
        
        # 하한과 상한 사이의 값들을 0~255로 매핑
        mask = (norm_data >= lower_bound) & (norm_data <= upper_bound)
        
        # 마스크에 해당하는 데이터가 있을 경우에만 정규화 수행
        if np.any(mask) and (upper_bound > lower_bound):
            gray_data[mask] = ((norm_data[mask] - lower_bound) * 255 / (upper_bound - lower_bound)).astype(np.uint8)
            
        # 가우시안 블러링
        gray_data = cv.GaussianBlur(gray_data, (3, 3), 0)
        
        # .jpg 이미지 데이터로 변환 (지정된 경로로 저장)
        cv.imwrite(save_path, gray_data)
        print(f"Grayscale data saved to {save_path}") # 저장 경로 출력
        return gray_data

    def can_calib(self, file_path):
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not read image from {file_path}")
            return False, None, None # 3개의 값 반환

        pattern_size = (19, 14)  # 체스보드 패턴의 내부 코너 수
        ret, corners = cv.findChessboardCorners(image, pattern_size, None)
        
        # 코너를 컬러로 그리기 위해 BGR 이미지로 변환
        annotated_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        if ret:
            cv.drawChessboardCorners(annotated_image, pattern_size, corners, ret)
            # cv.imshow('Calibration Pattern', annotated_image) # 루프에서 계속 창이 뜨는 것을 방지하기 위해 주석 처리
            # cv.waitKey(500) 
            # cv.destroyAllWindows()
        else:
            print(f"Calibration pattern not found in {file_path}.")
            
        return ret, corners, annotated_image # 코너가 그려진 이미지 반환

    # 2. calibrate_camera 수정: 단일 포인트(image_points) 대신 포인트 '리스트'를 받도록 변경
    def calibrate_camera(self, objpoints_list, imgpoints_list, image_size, save_path):
        
        # 3D 점들과 2D 점들이 이미 리스트로 전달됨
        # (내부에서 리스트를 새로 만들고 append 하던 코드 삭제)

        print(f"Calibrating using {len(imgpoints_list)} images.")
        
        # 카메라 보정 수행
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints_list, imgpoints_list, image_size, None, None)

        if ret:
            # 보정 행렬과 왜곡 계수를 파일에 저장
            np.savez(save_path, camera_matrix=mtx, dist_coeffs=dist)
            print(f"Camera calibration data saved to {save_path}")
            print("Camera Matrix (mtx):\n", mtx)
            print("Distortion Coefficients (dist):\n", dist)
        else:
            print("Camera calibration failed.")

if __name__ == "__main__":
    
    # ... (주석 처리된 npy 변환 루프) ...
            
    # 3. __main__ 루프 수정: 여러 이미지에서 코너를 찾아 리스트에 누적
    
    print("--- Starting Camera Calibration ---")
    
    # 실제 체스보드 패턴의 3D 좌표 생성 (모든 이미지에 동일하게 적용됨)
    objp = np.zeros((14*19, 3), np.float32)
    objp[:, :2] = np.mgrid[0:19, 0:14].T.reshape(-1, 2)
    objp *= 9  # 체스보드 각 사각형의 크기 (단위: mm)

    all_objpoints = []  # 모든 이미지의 3D 점들을 담을 리스트
    all_imgpoints = []  # 모든 이미지의 2D 점들(코너)을 담을 리스트
    
    image_size = (240, 180)  # 이미지 크기 (너비, 높이). 이 크기가 정확한지 확인하세요.
    num_images = 8           # 사용할 이미지 개수 (00, 01, 02, 03)

    # Calibration 객체 생성. (이제 .jpg 경로를 넣어도 __init__에서 오류가 나지 않음)
    # 이 객체는 can_calib와 calibrate_camera 메서드를 호출하기 위해 사용됩니다.
    calib = Calibration('./example/1110/calibration_00.jpg')

    for i in range(num_images):
        image_file_path = f'./example/1110/calibration_{i:02d}.jpg'
        print(f"Processing {image_file_path}...")
        
        # can_calib를 호출하여 코너 찾기
        found, corners, _ = calib.can_calib(image_file_path)
        
        if found:
            print(f"Pattern found in {image_file_path}")
            all_objpoints.append(objp)      # 3D 점 추가
            all_imgpoints.append(corners)   # 2D 코너 점 추가
        # else: (can_calib에서 이미 "not found" 메시지를 출력함)
        #     print(f"Pattern NOT found in {image_file_path}")

    # 루프가 끝난 후, 수집된 포인트가 있는지 확인
    if len(all_imgpoints) > 0:
        # 수집된 모든 포인트 리스트를 calibrate_camera에 전달
        calib.calibrate_camera(all_objpoints, all_imgpoints, image_size, './example/1110/camera_calibration_data.npz')
    else:
        print("Calibration failed: No valid chessboard patterns were found in any of the images.")

    # cv.destroyAllWindows() # can_calib에서 imshow를 주석 처리했으므로 필요 시 활성화