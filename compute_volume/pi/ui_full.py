import sys
import os
import datetime
import time
import numpy as np
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGridLayout, QFrame, QSizePolicy, QGraphicsDropShadowEffect, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QFontDatabase

# === [변경 1] Arducam 라이브러리 임포트 ===
try:
    import ArducamDepthCamera as ac
    CAMERA_AVAILABLE = True
except ImportError:
    print("[시스템] ArducamDepthCamera 라이브러리를 찾을 수 없습니다. 카메라 기능을 사용할 수 없습니다.")
    CAMERA_AVAILABLE = False

# === [변경 2] 경량화된 NumPy 추론 엔진 임포트 ===
# PyTorch 의존성을 완전히 제거하고, inference_pure.py를 사용합니다.
try:
    from inference_pure import NumpyVolumePredictor, FeatureExtractor
    
    class SmartVolumeEstimator:
        """
        UI와 순수 NumPy 추론 엔진을 연결하는 래퍼 클래스
        FeatureExtractor와 Predictor를 결합하여 사용
        """
        def __init__(self, model_path):
            # 학습 때 사용한 고정 ROI (이 값이 맞지 않으면 정확도 떨어짐)
            self.FIXED_ROI = (14, 30, 165, 151)
            
            print(f"[시스템] NumPy 추론 엔진 로드 중... ({model_path})")
            self.extractor = FeatureExtractor()
            self.predictor = NumpyVolumePredictor(model_path)
            
        def predict(self, data_path):
            # 1. 특징 추출 (ROI 적용)
            feats = self.extractor.process(data_path, roi=self.FIXED_ROI)
            
            if feats is None:
                print("[시스템] 특징 추출 실패 (데이터 품질 낮음)")
                return {'volume_ml': 0, 'error': True}
            
            # 2. 부피 예측
            result = self.predictor.predict(feats)
            return result

except ImportError:
    print("[시스템] inference_pure.py를 찾을 수 없습니다. 더미 모드로 동작합니다.")
    SmartVolumeEstimator = None


class ModernDispenserUI(QMainWindow):
    # 폰트 설정
    FONT_FILES = [
        "NotoSansKR-Bold.ttf",
        "NotoSansKR-ExtraBold.ttf",
        "NotoSansKR-Black.ttf",
        "NotoSansKR-Medium.ttf",
        "NotoSansKR-Regular.ttf"
    ]
    MAIN_FONT_FAMILY = "Noto Sans KR" 
    
    WEIGHT_NORMAL = 50
    WEIGHT_MEDIUM = 57
    WEIGHT_BOLD = 75
    WEIGHT_BLACK = 87

    # 카메라 설정
    MAX_DISTANCE = 2000

    def __init__(self):
        super().__init__()
        
        # === [변경 3] 시스템 설정 (모델 확장자 변경) ===
        self.MODEL_PATH = "model_weights.npz"  # .pth -> .npz 로 변경
        self.TEMP_DATA_PATH = "latest_scan.npy" 
        
        # === 아두이노 시리얼 설정 ===
        self.arduino = None
        self.init_serial_connection()

        # === 상태 관리 ===
        self.current_mode = 'VOLUME'  
        self.current_value = 120.0    
        self.is_dispensing = False 
        
        self.init_inference_engine()
        self.init_ui()
        
        self.ratio_buttons = [self.btn_r1, self.btn_r2, self.btn_r3]
        self.volume_buttons = [self.btn_v1, self.btn_v2, self.btn_v3]

        self.update_display()
        self.highlight_button('VOLUME', 120.0)

    def init_serial_connection(self):
        """아두이노와 시리얼 연결 시도"""
        try:
            ports = list(serial.tools.list_ports.comports())
            target_port = None
            
            for p in ports:
                if "Arduino" in p.description or "ttyACM" in p.device or "ttyUSB" in p.device:
                    target_port = p.device
                    break
            
            if target_port is None:
                target_port = '/dev/ttyUSB0' 

            print(f"[시리얼] 아두이노 연결 시도: {target_port}")
            self.arduino = serial.Serial(port=target_port, baudrate=9600, timeout=1)
            time.sleep(2) # 아두이노 리셋 대기
            print("[시리얼] 아두이노 연결 성공")
            
        except Exception as e:
            print(f"[시리얼] 아두이노 연결 실패: {e}")
            self.arduino = None

    def init_inference_engine(self):
        self.estimator = None 
        if SmartVolumeEstimator and os.path.exists(self.MODEL_PATH):
            try:
                self.estimator = SmartVolumeEstimator(self.MODEL_PATH)
                print("[시스템] 추론 엔진 준비 완료.")
            except Exception as e:
                print(f"[오류] 모델 로드 실패: {e}")
        else:
            print(f"[시스템] 경고: 추론 엔진을 로드할 수 없습니다. (파일 없음: {self.MODEL_PATH})")

    def load_custom_fonts(self):
        font_db = QFontDatabase()
        for font_file in self.FONT_FILES:
            if os.path.exists(font_file):
                font_db.addApplicationFont(font_file)

    def init_ui(self):
        self.setWindowTitle("Premium Water Dispenser")
        self.setGeometry(100, 100, 850, 550)
        self.load_custom_fonts()

        self.setStyleSheet(f"""
            QWidget {{
                font-family: "{self.MAIN_FONT_FAMILY}";
                background-color: #F2F4F7;
                color: #333333;
            }}
            QFrame#display_frame {{
                background-color: #FFFFFF;
                border-radius: 30px;
                border: 1px solid #E5E5EA;
            }}
            QFrame.info_card {{
                background-color: #FFFFFF;
                border-radius: 15px;
                border: 1px solid #D1D1D6;
            }}
            QLabel[class="section_title"] {{
                color: #8E8E93;
                font-size: 13px;
                font-weight: bold;
                letter-spacing: 1px;
                margin-top: 10px;
            }}
            QPushButton {{
                background-color: #FFFFFF;
                border: 1px solid #D1D1D6;
                border-radius: 15px;
                font-size: 16px;
                color: #1C1C1E;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #F2F2F7;
                border: 1px solid #AEAEB2;
            }}
            QPushButton[selected="true"] {{
                background-color: #007AFF; 
                border: 1px solid #007AFF;
                color: #FFFFFF;
            }}
            QPushButton[class="control_button"] {{
                background-color: #FFFFFF;
                color: #007AFF;
                border: 1px solid #D1D1D6;
                font-size: 20px;
                border-radius: 12px;
            }}
            QPushButton[class="control_button"]:pressed {{
                background-color: #E5F1FF;
            }}
            QPushButton#dispense_button {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00C6FF, stop:1 #0072FF);
                border: none;
                border-radius: 28px;
                color: white;
                font-size: 22px;
                font-weight: 900;
                letter-spacing: 2px;
            }}
            QPushButton#dispense_button:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0055FF, stop:1 #0044CC);
            }}
            QPushButton#dispense_button[state="stop"] {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF453A, stop:1 #FF3B30);
                border: 2px solid #FF3B30;
            }}
            QPushButton#dispense_button[state="stop"]:pressed {{
                background: #D70015;
            }}
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_h_layout = QHBoxLayout(central_widget)
        main_h_layout.setContentsMargins(40, 40, 40, 40)
        main_h_layout.setSpacing(30)

        # 좌측 디스플레이
        self.display_frame = QFrame(objectName="display_frame")
        self.display_frame.setFixedWidth(320)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 15)) 
        shadow.setOffset(0, 5)
        self.display_frame.setGraphicsEffect(shadow)

        disp_layout = QVBoxLayout(self.display_frame)
        disp_layout.setContentsMargins(30, 40, 30, 40)
        disp_layout.setSpacing(5)

        self.lbl_status = QLabel("MANUAL MODE")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setFont(QFont(self.MAIN_FONT_FAMILY, 12, self.WEIGHT_BOLD))
        self.lbl_status.setStyleSheet("background-color: transparent; color: #007AFF; letter-spacing: 1px;")
        
        self.lbl_main_value = QLabel("120")
        self.lbl_main_value.setAlignment(Qt.AlignCenter)
        self.lbl_main_value.setFont(QFont(self.MAIN_FONT_FAMILY, 75, self.WEIGHT_BLACK)) 
        self.lbl_main_value.setStyleSheet("background-color: transparent; color: #1D1D1F; margin-bottom: -15px;")
        
        self.lbl_unit = QLabel("ml")
        self.lbl_unit.setAlignment(Qt.AlignCenter)
        self.lbl_unit.setFont(QFont(self.MAIN_FONT_FAMILY, 24, self.WEIGHT_MEDIUM))
        self.lbl_unit.setStyleSheet("background-color: transparent; color: #86868B;")
        
        self.lbl_sub_info = QLabel("고정 정량 출수")
        self.lbl_sub_info.setAlignment(Qt.AlignCenter)
        self.lbl_sub_info.setFont(QFont(self.MAIN_FONT_FAMILY, 14, self.WEIGHT_NORMAL))
        self.lbl_sub_info.setStyleSheet("background-color: transparent; color: #AEAEB2; margin-top: 10px;")

        disp_layout.addStretch(1)
        disp_layout.addWidget(self.lbl_status)
        disp_layout.addWidget(self.lbl_main_value)
        disp_layout.addWidget(self.lbl_unit)
        disp_layout.addWidget(self.lbl_sub_info)
        disp_layout.addStretch(1)
        
        main_h_layout.addWidget(self.display_frame)

        # 우측 컨트롤 패널
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(20)
        right_panel_layout.setContentsMargins(0, 10, 0, 10)

        control_grid = QGridLayout()
        control_grid.setVerticalSpacing(12)
        control_grid.setHorizontalSpacing(12)

        today = datetime.date.today()
        next_care = today + datetime.timedelta(days=15)
        next_care_str = next_care.strftime("%m.%d")

        self.card_usage = self.create_info_card("사용 기간", "10개월", "#1C1C1E")
        control_grid.addWidget(self.card_usage, 2, 0)

        self.card_date = self.create_info_card("다음 관리", f"{next_care_str}", "#007AFF")
        control_grid.addWidget(self.card_date, 2, 1)

        self.card_temp = self.create_info_card("물 온도", "4°C", "#30D158")
        control_grid.addWidget(self.card_temp, 2, 2)

        self.btn_up = QPushButton("▲")
        self.btn_up.clicked.connect(lambda: self.adjust_value(1))
        self.btn_up.setProperty("class", "control_button")
        
        self.btn_down = QPushButton("▼")
        self.btn_down.clicked.connect(lambda: self.adjust_value(-1))
        self.btn_down.setProperty("class", "control_button")

        self.btn_up.setFixedHeight(55)
        self.btn_down.setFixedHeight(55)

        lbl_ratio = QLabel("SMART FILL (컵 용량 자동 인식)")
        lbl_ratio.setProperty("class", "section_title")
        control_grid.addWidget(lbl_ratio, 5, 0, 1, 4)

        self.btn_r1 = self.create_button("50%", lambda: self.set_mode('RATIO', 0.5))
        self.btn_r2 = self.create_button("75%", lambda: self.set_mode('RATIO', 0.75))
        self.btn_r3 = self.create_button("90%", lambda: self.set_mode('RATIO', 0.90))

        control_grid.addWidget(self.btn_r1, 6, 0)
        control_grid.addWidget(self.btn_r2, 6, 1)
        control_grid.addWidget(self.btn_r3, 6, 2)
        
        control_grid.addWidget(self.btn_up, 6, 3)

        lbl_vol = QLabel("PRESET VOLUME")
        lbl_vol.setProperty("class", "section_title")
        control_grid.addWidget(lbl_vol, 7, 0, 1, 4)

        self.btn_v1 = self.create_button("120ml", lambda: self.set_mode('VOLUME', 120))
        self.btn_v2 = self.create_button("250ml", lambda: self.set_mode('VOLUME', 250))
        self.btn_v3 = self.create_button("500ml", lambda: self.set_mode('VOLUME', 500))
        
        control_grid.addWidget(self.btn_v1, 8, 0)
        control_grid.addWidget(self.btn_v2, 8, 1)
        control_grid.addWidget(self.btn_v3, 8, 2)

        control_grid.addWidget(self.btn_down, 8, 3)

        control_grid.setColumnStretch(0, 1)
        control_grid.setColumnStretch(1, 1)
        control_grid.setColumnStretch(2, 1)
        control_grid.setColumnStretch(3, 0)

        right_panel_layout.addLayout(control_grid)
        right_panel_layout.addStretch(1) 

        self.btn_pour = QPushButton("SCAN & DISPENSE")
        self.btn_pour.setObjectName("dispense_button")
        self.btn_pour.setFixedHeight(65)
        self.btn_pour.setCursor(Qt.PointingHandCursor)
        self.btn_pour.clicked.connect(self.process_dispense_sequence)
        
        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(15)
        btn_shadow.setColor(QColor(0, 122, 255, 80)) 
        btn_shadow.setOffset(0, 4)
        self.btn_pour.setGraphicsEffect(btn_shadow)

        right_panel_layout.addWidget(self.btn_pour)

        main_h_layout.addLayout(right_panel_layout)

    def create_info_card(self, title, value, value_color):
        frame = QFrame()
        frame.setProperty("class", "info_card")
        frame.setFixedHeight(85)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(2)
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont(self.MAIN_FONT_FAMILY, 11, self.WEIGHT_MEDIUM))
        lbl_title.setStyleSheet("color: #8E8E93; border: none; background: transparent;")
        lbl_title.setAlignment(Qt.AlignLeft)
        lbl_value = QLabel(value)
        lbl_value.setFont(QFont(self.MAIN_FONT_FAMILY, 16, self.WEIGHT_BOLD))
        lbl_value.setStyleSheet(f"color: {value_color}; border: none; background: transparent;")
        lbl_value.setAlignment(Qt.AlignLeft)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        return frame

    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.setMinimumHeight(55)
        btn.setFont(QFont(self.MAIN_FONT_FAMILY, 15, self.WEIGHT_BOLD))
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def set_mode(self, mode, value):
        if self.is_dispensing: return
        self.current_mode = mode
        self.current_value = float(value)
        self.update_display()
        self.highlight_button(mode, value)

    def highlight_button(self, mode, value):
        all_buttons = self.ratio_buttons + self.volume_buttons
        for btn in all_buttons:
            btn.setProperty("selected", False)
            btn.style().polish(btn)

        if mode in ('RATIO', 'VOLUME'):
            search_str = f"{int(round(value * 100))}%" if mode == 'RATIO' else f"{int(value)}ml"
            target_list = self.ratio_buttons if mode == 'RATIO' else self.volume_buttons
            for btn in target_list:
                if btn.text() == search_str:
                    btn.setProperty("selected", True)
                    btn.style().polish(btn)
                    break

    def adjust_value(self, direction):
        if self.is_dispensing: return
        if self.current_mode == 'RATIO':
            new_val = self.current_value + (0.1 * direction)
            new_val = max(0.1, min(1.0, new_val))
            self.current_value = round(new_val, 2)
        elif self.current_mode == 'VOLUME':
            new_val = self.current_value + (50 * direction)
            new_val = max(50, min(2000, new_val))
            self.current_value = int(new_val)
        self.update_display()
        self.highlight_button(None, None)

    def update_display(self):
        if self.current_mode == 'RATIO':
            percentage = int(self.current_value * 100)
            self.lbl_main_value.setText(f"{percentage}")
            self.lbl_unit.setText("%")
            self.lbl_sub_info.setText("용기 대비 출수량 (SCAN 필요)")
            self.lbl_status.setText("SMART FILL")
            self.lbl_status.setStyleSheet("background-color: transparent; color: #30D158; letter-spacing: 1px;")
        else:
            self.lbl_main_value.setText(f"{int(self.current_value)}")
            self.lbl_unit.setText("ml")
            self.lbl_sub_info.setText("설정 용량 출수")
            self.lbl_status.setText("MANUAL MODE")
            self.lbl_status.setStyleSheet("background-color: transparent; color: #007AFF; letter-spacing: 1px;")

    def capture_depth_image(self, duration=0.3):
        if not CAMERA_AVAILABLE:
            print("[경고] 카메라 라이브러리가 없어 촬영을 건너뜁니다.")
            return False

        print("[카메라] 초기화 중...")
        try:
            cam = ac.ArducamCamera()
            ret = cam.open(ac.Connection.CSI, 0)
            if ret != 0:
                print(f"[오류] 카메라 열기 실패. 코드: {ret}")
                return False

            ret = cam.start(ac.FrameType.DEPTH)
            if ret != 0:
                print(f"[오류] 카메라 시작 실패. 코드: {ret}")
                cam.close()
                return False

            cam.setControl(ac.Control.RANGE, self.MAX_DISTANCE)
            captured_frames = []
            start_time = time.time()
            
            print(f"[카메라] {duration}초간 데이터 수집 시작...")
            while (time.time() - start_time) < duration:
                frame = cam.requestFrame(100)
                if frame is not None and isinstance(frame, ac.DepthData):
                    captured_frames.append(frame.depth_data.copy())
                    cam.releaseFrame(frame)
                QApplication.processEvents()

            cam.stop()
            cam.close()

            if captured_frames:
                avg_depth = np.mean(captured_frames, axis=0).astype(np.float32)
                np.save(self.TEMP_DATA_PATH, avg_depth)
                print(f"[카메라] 저장 완료: {self.TEMP_DATA_PATH}")
                return True
            else:
                return False
        except Exception as e:
            print(f"[오류] 카메라 예외 발생: {e}")
            return False

    def send_to_arduino(self, volume):
        if self.arduino and self.arduino.is_open:
            try:
                msg = f"{volume}\n"
                self.arduino.write(msg.encode('utf-8'))
                print(f"[시리얼] 전송: {msg.strip()}")
            except Exception as e:
                print(f"[시리얼] 전송 실패: {e}")
        else:
            print("[시리얼] 연결 안됨.")

    def process_dispense_sequence(self):
        if self.is_dispensing:
            print("[시스템] 사용자 중지 요청!")
            self.send_to_arduino(-1) 
            self.reset_ui_state()    
            return

        original_text = self.lbl_status.text()
        self.lbl_status.setText("SCANNING...")
        self.btn_pour.setEnabled(False) 
        self.btn_pour.setText("Processing...")
        QApplication.processEvents() 

        # 1. 카메라 촬영
        success = self.capture_depth_image(duration=0.3)
        if not success and CAMERA_AVAILABLE:
            QMessageBox.warning(self, "오류", "카메라 촬영에 실패했습니다.")
            self.reset_ui_state()
            return

        # 2. 추론 (순수 NumPy 엔진 사용)
        final_volume_ml = 0
        predicted_capacity = 0
        
        if self.estimator:
            if os.path.exists(self.TEMP_DATA_PATH):
                print(f"[시스템] 추론 시작: {self.TEMP_DATA_PATH}")
                # NumPy 예측기 호출
                result = self.estimator.predict(self.TEMP_DATA_PATH)
                
                if result.get('error'):
                    predicted_capacity = 350.0 # 에러 시 기본값
                else:
                    predicted_capacity = result.get('volume_ml', 0)
                
                print(f"[시스템] 추론된 컵 용량: {predicted_capacity}ml (모드: {result.get('mode', 'N/A')})")
            else:
                print("[시스템] 데이터 없음. 더미 값 사용.")
                predicted_capacity = 350.0 
        else:
            print("[시스템] 추론 엔진 없음. 더미 값 사용.")
            predicted_capacity = 350.0

        # 3. 출수량 계산
        if self.current_mode == 'RATIO':
            final_volume_ml = predicted_capacity * self.current_value
            print(f"[출수] RATIO ({self.current_value*100}%) | 컵: {predicted_capacity}ml -> 출수: {final_volume_ml:.1f}ml")
            self.lbl_main_value.setText(f"{int(final_volume_ml)}")
            self.lbl_unit.setText("ml (Auto)")
            QApplication.processEvents()
        else:
            final_volume_ml = self.current_value
            print(f"[출수] MANUAL | 설정: {final_volume_ml}ml")

        # 4. 아두이노 전송 및 상태 변경
        self.send_to_arduino(final_volume_ml)
        self.enable_stop_mode()

    def enable_stop_mode(self):
        self.is_dispensing = True
        self.btn_pour.setEnabled(True)
        self.btn_pour.setText("STOP (정지)")
        self.btn_pour.setProperty("state", "stop") 
        self.btn_pour.style().polish(self.btn_pour) 
        
        self.lbl_status.setText("DISPENSING...")
        self.lbl_status.setStyleSheet("color: #FF3B30; letter-spacing: 1px;")

        QTimer.singleShot(15000, self.auto_reset_check)

    def auto_reset_check(self):
        if self.is_dispensing:
            print("[시스템] 타임아웃: UI 자동 초기화")
            self.reset_ui_state()

    def reset_ui_state(self, original_status_text=None):
        self.is_dispensing = False
        self.btn_pour.setText("SCAN & DISPENSE")
        self.btn_pour.setEnabled(True)
        self.btn_pour.setProperty("state", "normal") 
        self.btn_pour.style().polish(self.btn_pour)
        self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.arduino:
                self.arduino.close()
            self.close()

if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    window = ModernDispenserUI()
    window.showFullScreen() 
    sys.exit(app.exec_())