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

# === [설정] 테스트 모드 (True면 카메라 없이 더미 데이터 사용) ===
TEST_MODE = True 

# === 라이브러리 로드 (Arducam & Inference) ===
try:
    if not TEST_MODE:
        import ArducamDepthCamera as ac
        CAMERA_AVAILABLE = True
    else:
        print("[시스템] 테스트 모드: 카메라 라이브러리 로드 안 함")
        CAMERA_AVAILABLE = False
except ImportError:
    print("[시스템] ArducamDepthCamera 라이브러리 없음.")
    CAMERA_AVAILABLE = False

try:
    from inference_pure import NumpyVolumePredictor, FeatureExtractor
    
    class SmartVolumeEstimator:
        def __init__(self, model_path):
            self.FIXED_ROI = (14, 30, 165, 151)
            print(f"[시스템] NumPy 추론 엔진 로드 중... ({model_path})")
            self.extractor = FeatureExtractor()
            self.predictor = NumpyVolumePredictor(model_path)
            
        def predict(self, data_path):
            feats = self.extractor.process(data_path, roi=self.FIXED_ROI)
            if feats is None:
                return {'volume_ml': 0, 'error': True}
            result = self.predictor.predict(feats)
            return result

except ImportError:
    print("[시스템] inference_pure.py 없음. 더미 모드 동작.")
    SmartVolumeEstimator = None


class ModernDispenserUI(QMainWindow):
    FONT_FILES = [
        "NotoSansKR-Bold.ttf", "NotoSansKR-ExtraBold.ttf",
        "NotoSansKR-Black.ttf", "NotoSansKR-Medium.ttf", "NotoSansKR-Regular.ttf"
    ]
    MAIN_FONT_FAMILY = "Noto Sans KR" 
    
    WEIGHT_NORMAL = 50
    WEIGHT_MEDIUM = 57
    WEIGHT_BOLD = 75
    WEIGHT_BLACK = 87

    MAX_DISTANCE = 2000

    def __init__(self):
        super().__init__()
        
        # === [물 용량 관리 변수] ===
        self.MAX_CAPACITY = 2500.0  # 전체 용량 2.5L
        self.current_water = 2500.0 # 현재 남은 물
        self.pending_deduction = 0.0 # 출수 중인 양

        self.MODEL_PATH = "model_weights.npz"
        self.TEMP_DATA_PATH = "712_wine_11_depth.npy" 
        
        self.arduino = None
        self.init_serial_connection()

        self.current_mode = 'VOLUME'  
        self.current_value = 120.0    
        self.is_dispensing = False 
        
        # 시리얼 데이터를 주기적으로 체크하는 타이머
        self.check_timer = QTimer()
        self.check_timer.setInterval(100) # 0.1초마다 확인
        self.check_timer.timeout.connect(self.check_serial_feedback)

        self.init_inference_engine()
        self.init_ui()
        
        self.ratio_buttons = [self.btn_r1, self.btn_r2, self.btn_r3]
        self.volume_buttons = [self.btn_v1, self.btn_v2, self.btn_v3]

        self.update_display()
        self.update_water_card() # 초기 물 용량 UI 갱신
        self.highlight_button('VOLUME', 120.0)

    def init_serial_connection(self):
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
            self.arduino = serial.Serial(port=target_port, baudrate=9600, timeout=0.1)
            time.sleep(2)
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
            print(f"[시스템] 경고: 추론 엔진 로드 불가.")

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
            QWidget {{ font-family: "{self.MAIN_FONT_FAMILY}"; background-color: #F2F4F7; color: #333333; }}
            QFrame#display_frame {{ background-color: #FFFFFF; border-radius: 30px; border: 1px solid #E5E5EA; }}
            QFrame.info_card {{ background-color: #FFFFFF; border-radius: 15px; border: 1px solid #D1D1D6; }}
            QLabel[class="section_title"] {{ color: #8E8E93; font-size: 13px; font-weight: bold; letter-spacing: 1px; margin-top: 10px; }}
            QPushButton {{ background-color: #FFFFFF; border: 1px solid #D1D1D6; border-radius: 15px; font-size: 16px; color: #1C1C1E; font-weight: bold; }}
            QPushButton:hover {{ background-color: #F2F2F7; border: 1px solid #AEAEB2; }}
            QPushButton[selected="true"] {{ background-color: #007AFF; border: 1px solid #007AFF; color: #FFFFFF; }}
            QPushButton[class="control_button"] {{ background-color: #FFFFFF; color: #007AFF; border: 1px solid #D1D1D6; font-size: 20px; border-radius: 12px; }}
            QPushButton[class="control_button"]:pressed {{ background-color: #E5F1FF; }}
            QPushButton#dispense_button {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00C6FF, stop:1 #0072FF); border: none; border-radius: 28px; color: white; font-size: 22px; font-weight: 900; letter-spacing: 2px; }}
            QPushButton#dispense_button:pressed {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0055FF, stop:1 #0044CC); }}
            QPushButton#dispense_button[state="stop"] {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF453A, stop:1 #FF3B30); border: 2px solid #FF3B30; }}
            QPushButton#dispense_button[state="stop"]:pressed {{ background: #D70015; }}
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_h_layout = QHBoxLayout(central_widget)
        main_h_layout.setContentsMargins(40, 40, 40, 40)
        main_h_layout.setSpacing(30)

        # Left Display
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

        # Right Panel
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(20)
        right_panel_layout.setContentsMargins(0, 10, 0, 10)
        control_grid = QGridLayout()
        control_grid.setVerticalSpacing(12)
        control_grid.setHorizontalSpacing(12)

        today = datetime.date.today()
        next_care = today + datetime.timedelta(days=15)
        
        # === [카드 UI] ===
        self.card_usage = self.create_info_card("사용 기간", "10개월", "#1C1C1E")
        control_grid.addWidget(self.card_usage, 2, 0)
        
        self.card_date = self.create_info_card("다음 관리", f"{next_care.strftime('%m.%d')}", "#007AFF")
        control_grid.addWidget(self.card_date, 2, 1)
        
        # [수정됨] 물 용량 카드 (클릭 시 리셋 기능 연결)
        self.card_temp = self.create_info_card("물 용량", "100%", "#30D158", on_click=self.reset_water_capacity)
        control_grid.addWidget(self.card_temp, 2, 2)
        # ==========================

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

    # [수정됨] 클릭 이벤트를 받을 수 있도록 on_click 인자 추가
    def create_info_card(self, title, value, value_color, on_click=None):
        frame = QFrame()
        frame.setProperty("class", "info_card")
        frame.setFixedHeight(85)
        
        # 클릭 이벤트 처리 로직 추가
        if on_click:
            frame.setCursor(Qt.PointingHandCursor)
            # 마우스 클릭 이벤트 오버라이드
            frame.mousePressEvent = lambda event: on_click()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(2)
        
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont(self.MAIN_FONT_FAMILY, 11, self.WEIGHT_MEDIUM))
        lbl_title.setStyleSheet("color: #8E8E93; border: none; background: transparent;")
        lbl_title.setAlignment(Qt.AlignLeft)
        
        lbl_value = QLabel(value)
        lbl_value.setObjectName("value_label") 
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

    # === [신규 기능] 물 용량 리셋 (100% 충전) ===
    def reset_water_capacity(self):
        print("[시스템] 물 보충 완료: 용량을 100%로 리셋합니다.")
        self.current_water = self.MAX_CAPACITY
        self.update_water_card()

    # === [핵심 기능] 물 용량 UI 업데이트 ===
    def update_water_card(self):
        percentage = (self.current_water / self.MAX_CAPACITY) * 100
        
        # 0% 이하 방지
        if percentage < 0: percentage = 0
        
        # 20% 미만이면 붉은색(#FF3B30), 아니면 초록색(#30D158)
        color = "#FF3B30" if percentage < 20 else "#30D158"
        
        # 카드 내부의 값 라벨 찾기
        lbl_value = self.card_temp.findChild(QLabel, "value_label")
        if lbl_value:
            lbl_value.setText(f"{percentage:.0f}%")
            lbl_value.setStyleSheet(f"color: {color}; border: none; background: transparent;")

    def capture_depth_image(self, duration=0.3):
        if TEST_MODE:
            print(f"[테스트 모드] 카메라 촬영 건너뜀. 기존 파일 사용: {self.TEMP_DATA_PATH}")
            return os.path.exists(self.TEMP_DATA_PATH)

        if not CAMERA_AVAILABLE:
            return False

        try:
            cam = ac.ArducamCamera()
            if cam.open(ac.Connection.CSI, 0) != 0: return False
            if cam.start(ac.FrameType.DEPTH) != 0: 
                cam.close()
                return False

            cam.setControl(ac.Control.RANGE, self.MAX_DISTANCE)
            captured_frames = []
            start_time = time.time()
            
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
                return True
            else:
                return False
        except Exception:
            return False

    def send_to_arduino(self, volume):
        if self.arduino and self.arduino.is_open:
            try:
                # 버퍼 비우기 (이전 데이터 삭제)
                self.arduino.reset_input_buffer()
                
                vol_int = int(volume) 
                msg = f"{vol_int}\n"
                self.arduino.write(msg.encode('utf-8'))
                print(f"[시리얼] 전송: {msg.strip()}")
            except Exception as e:
                print(f"[시리얼] 전송 실패: {e}")

    # === [핵심 기능] 아두이노 DONE 신호 처리 ===
    def check_serial_feedback(self):
        if self.arduino and self.arduino.is_open and self.arduino.in_waiting:
            try:
                line = self.arduino.readline().decode('utf-8').strip()
                if line == "DONE":
                    print("[시스템] 출수 완료 신호 수신 (DONE)")
                    
                    # 1. 물 용량 차감
                    self.current_water -= self.pending_deduction
                    if self.current_water < 0: self.current_water = 0
                    
                    # 2. UI 업데이트
                    self.update_water_card()
                    
                    # 3. 상태 리셋
                    self.reset_ui_state()
            except Exception as e:
                print(f"[오류] 시리얼 읽기 실패: {e}")

    def process_dispense_sequence(self):
        if self.is_dispensing:
            print("[시스템] 사용자 중지 요청!")
            self.send_to_arduino(-1) 
            self.reset_ui_state()    
            return

        self.lbl_status.setText("SCANNING...")
        self.btn_pour.setEnabled(False) 
        self.btn_pour.setText("Processing...")
        QApplication.processEvents() 

        # 1. 카메라 촬영
        success = self.capture_depth_image(duration=0.3)
        if not success and not TEST_MODE and CAMERA_AVAILABLE:
            QMessageBox.warning(self, "오류", "카메라 촬영에 실패했습니다.")
            self.reset_ui_state()
            return
        elif not success and TEST_MODE:
             QMessageBox.warning(self, "오류", f"테스트 파일이 없습니다.\n{self.TEMP_DATA_PATH}")
             self.reset_ui_state()
             return

        # 2. 추론
        predicted_capacity = 350.0
        if self.estimator and os.path.exists(self.TEMP_DATA_PATH):
            result = self.estimator.predict(self.TEMP_DATA_PATH)
            if not result.get('error'):
                predicted_capacity = result.get('volume_ml', 0)
            print(f"[시스템] 추론된 컵 용량: {predicted_capacity}ml")
        
        # 3. 출수량 계산 및 저장 (pending_deduction)
        final_volume_ml = 0
        if self.current_mode == 'RATIO':
            final_volume_ml = predicted_capacity * self.current_value
            self.lbl_main_value.setText(f"{int(final_volume_ml)}")
            self.lbl_unit.setText("ml (Auto)")
        else:
            final_volume_ml = self.current_value

        # 출수 예정량 저장
        self.pending_deduction = final_volume_ml 
        print(f"[출수] 명령: {final_volume_ml}ml")

        # 4. 아두이노 전송 및 모니터링 시작
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

        # 기존 15초 타임아웃 + 시리얼 감지 시작
        self.check_timer.start() 
        QTimer.singleShot(15000, self.auto_reset_check)

    def auto_reset_check(self):
        if self.is_dispensing:
            print("[시스템] 타임아웃: 강제 UI 초기화")
            self.reset_ui_state()

    def reset_ui_state(self, original_status_text=None):
        self.is_dispensing = False
        self.check_timer.stop() # 시리얼 감지 중단
        
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
    window.show()
    sys.exit(app.exec_())