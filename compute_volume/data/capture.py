import cv2
import numpy as np
import ArducamDepthCamera as ac
import time  # 시간 측정을 위해 추가

# MAX_DISTANCE value modifiable  is 2000 or 4000
MAX_DISTANCE = 2000


class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


confidence_value = 30
selectRect, followRect = UserRect(), UserRect()


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_mouse(event, x, y, flags, param):
    global selectRect, followRect

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4
        selectRect.start_y = y - 4
        selectRect.end_x = x + 4
        selectRect.end_y = y + 4
    else:
        followRect.start_x = x - 4
        followRect.start_y = y - 4
        followRect.end_x = x + 4
        followRect.end_y = y + 4


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


def usage(argv0):
    print("Usage: python " + argv0 + " [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")


def main():
    print("Arducam Depth Camera Demo.")
    print("  SDK version:", ac.__version__)

    cam = ac.ArducamCamera()
    cfg_path = None
    # cfg_path = "file.cfg"

    black_color = (0, 0, 0)
    white_color = (255, 255, 255)

    ret = 0
    if cfg_path is not None:
        ret = cam.openWithFile(cfg_path, 0)
    else:
        ret = cam.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to open camera. Error code:", ret)
        return

    ret = cam.start(ac.FrameType.DEPTH)
    if ret != 0:
        print("Failed to start camera. Error code:", ret)
        cam.close()
        return

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)

    r = cam.getControl(ac.Control.RANGE)

    info = cam.getCameraInfo()
    print(f"Camera resolution: {info.width}x{info.height}")

    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("preview", on_mouse)

    if info.device_type == ac.DeviceType.VGA:
        # Only VGA support confidence
        cv2.createTrackbar(
            "confidence", "preview", confidence_value, 255, on_confidence_changed
        )

    while True:
        # 일반 프리뷰용 프레임 요청
        frame = cam.requestFrame(2000)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            result_image = getPreviewRGB(result_image, confidence_buf)

            cv2.normalize(confidence_buf, confidence_buf, 1, 0, cv2.NORM_MINMAX)

            cv2.imshow("preview_confidence", confidence_buf)

            cv2.rectangle(result_image, followRect.rect, white_color, 1)
            if not selectRect.empty:
                cv2.rectangle(result_image, selectRect.rect, black_color, 2)
                print("select Rect distance:", np.mean(depth_buf[selectRect.slice]))

            cv2.imshow("preview", result_image)
            cam.releaseFrame(frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        
        # 's' 키를 눌렀을 때 0.3초간 데이터 수집 및 평균 저장 로직
        elif key == ord("s"):
            print("Start capturing for 0.3 seconds...")
            captured_frames = []
            start_time = time.time()
            
            # 0.3초 동안 루프를 돌며 가능한 많은 프레임을 수집
            while (time.time() - start_time) < 0.3:
                # 빠른 수집을 위해 timeout을 짧게 설정 (예: 100ms)
                frame_tmp = cam.requestFrame(100)
                if frame_tmp is not None and isinstance(frame_tmp, ac.DepthData):
                    # C++ 버퍼 포인터 문제가 생기지 않도록 .copy()를 사용하여 깊은 복사 수행
                    captured_frames.append(frame_tmp.depth_data.copy())
                    cam.releaseFrame(frame_tmp)
            
            if captured_frames:
                # (N, H, W) 형태의 배열을 시간 축(axis=0) 기준으로 평균 계산
                # 제공해주신 샘플 데이터와 동일하게 float32 타입으로 변환
                avg_depth = np.mean(captured_frames, axis=0).astype(np.float32)
                
                timestamp = int(time.time())
                filename = f"depth_avg_{timestamp}.npy"
                
                # .npy 포맷으로 저장
                np.save(filename, avg_depth)
                print(f"Saved: {len(captured_frames)} frames averaged to '{filename}'")
                print(f"Data Shape: {avg_depth.shape}, Dtype: {avg_depth.dtype}")
            else:
                print("Warning: No frames captured during the interval.")

    cam.stop()
    cam.close()


if __name__ == "__main__":
    main()