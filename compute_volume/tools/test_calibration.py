import cv2
import numpy as np

def test_undistortion():
    # --- 1. 사용자 제공 값 ---
    # 카메라 행렬 (mtx)
    mtx = np.array([[211.46873334,   0.,         119.88074471],
                        [  0.,         210.801416,    86.90851171],
                        [  0.,           0.,           1.        ]])
    dist = np.array([[ 4.93083651e-01, -1.25632226e+00,  2.28374174e-03, -1.58899167e-05,
                                5.29625368e-01]])

    # --- 2. 테스트 이미지 로드 ---
    # !!! 중요: 'your_distorted_image.jpg'를 실제 왜곡된 체커보드 이미지 파일 경로로 변경하세요.
    image_path = 'Checkerboard_pattern.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (240, 180))  # 이미지 크기를 조절 (필요시)
    
    if img is None:
        print(f"Error: 이미지를 로드할 수 없습니다. '{image_path}' 경로를 확인하세요.")
        print("이 코드를 사용하려면 캘리브레이션에 사용했던 원본 체커보드 이미지 파일이 필요합니다.")
        return

    print(f"이미지 로드 완료: {image_path} (크기: {img.shape[1]}x{img.shape[0]})")

    # --- 3. 왜곡 보정 ---
    h, w = img.shape[:2]
    
    # getOptimalNewCameraMatrix: 보정 후 이미지의 유효 픽셀 영역을 계산하고, 
    # 불필요한 검은 영역을 제거하거나(alpha=0) 모든 픽셀을 유지(alpha=1)하도록 새 카메라 매트릭스를 최적화합니다.
    # alpha=1: 모든 원본 픽셀을 유지 (검은색 테두리가 생길 수 있음)
    # alpha=0: 왜곡 보정 후 유효한 픽셀 영역만 잘라냄
    alpha = 1.0 
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    
    if new_camera_mtx is None:
        print("Error: new_camera_mtx를 계산할 수 없습니다. mtx 또는 dist 값이 잘못되었을 수 있습니다.")
        return
        
    # cv2.undistort() 함수를 사용하여 왜곡을 보정합니다.
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # (선택 사항) ROI(Region of Interest)를 사용하여 유효한 픽셀 영역만 잘라내기
    # x, y, w_roi, h_roi = roi
    # undistorted_img_cropped = undistorted_img[y:y+h_roi, x:x+w_roi]

    # --- 4. 결과 비교 ---
    
    # 보기 편하도록 이미지 크기를 조절 (원본 이미지가 너무 클 경우)
    scale = 2 # 200%로 확대 (이미지 크기에 맞게 조절하세요)
    img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    undistorted_resized = cv2.resize(undistorted_img, (0, 0), fx=scale, fy=scale)
    
    # 원본과 보정본을 나란히 붙여서 비교
    comparison_image = np.hstack((img_resized, undistorted_resized))
    
    # 창 제목에 정보 표시
    cv2.putText(comparison_image, 'Original (Distorted)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison_image, 'Undistorted', (img_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Distortion Correction Test (Original vs Undistorted)', comparison_image)
    print("결과 창이 표시되었습니다. 'q' 키를 누르면 종료됩니다.")
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_undistortion()