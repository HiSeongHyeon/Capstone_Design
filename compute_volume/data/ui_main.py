import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGridLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

# 추론 엔진 Import (같은 폴더에 inference.py 필요)
try:
    from inference import VolumeEstimator
except ImportError:
    print("[시스템] inference.py를 찾을 수 없어 더미 모드로 실행합니다.")
    VolumeEstimator = None

class ModernDispenserUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # === 시스템 설정 ===
        self.MODEL_PATH = "my_cup_model.pth"
        self.DATA_PATH = "567_tumbler_04_depth.npy"
        
        # === 상태 관리 변수 (State Management) ===
        # mode: 'RATIO' (비율 모드) 또는 'VOLUME' (고정 출수 모드)
        self.current_mode = 'VOLUME'  
        self.current_value = 120.0    # 초기값
        self.is_scanning = False      # 스캔 중 플래그

        # 엔진 초기화
        self.estimator = None
        self.init_inference_engine()
        
        # UI 구성
        self.init_ui()
        
        # 초기 상태 표시
        self.update_display()

    def init_inference_engine(self):
        if VolumeEstimator and os.path.exists(self.MODEL_PATH):
            try:
                self.estimator = VolumeEstimator(self.MODEL_PATH)
                print(f"[시스템] 모델 로드 완료: {self.MODEL_PATH}")
            except Exception as e:
                print(f"[오류] 모델 로드 실패: {e}")
        else:
            print("[시스템] 경고: 추론 엔진을 로드할 수 없습니다.")

    def init_ui(self):
        """Apple Liquid Glass 스타일 UI 초기화"""
        self.setWindowTitle("Smart Water Dispenser")
        self.setGeometry(100, 100, 480, 800) # 세로형 터치스크린 비율
        
        # 1. 배경 및 전체 스타일 (Dark Glassmorphism)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1c1c1e, stop:1 #2c2c2e);
            }
            QLabel {
                color: #ffffff;
                font-family: 'Arial';
            }
            /* 공통 버튼 스타일 (유리 질감) */
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.2);
            }
            /* 섹션 라벨 스타일 */
            QLabel.section_title {
                color: #8e8e93;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 40, 30, 40)
        main_layout.setSpacing(25)

        # 2. 상단 디스플레이 영역 (정보 표시)
        self.display_frame = QFrame()
        self.display_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)
        self.display_frame.setFixedHeight(220)
        disp_layout = QVBoxLayout(self.display_frame)
        
        self.lbl_status = QLabel("READY")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #0a84ff; font-size: 16px; font-weight: bold;")
        
        self.lbl_main_value = QLabel("120")
        self.lbl_main_value.setAlignment(Qt.AlignCenter)
        self.lbl_main_value.setStyleSheet("font-size: 72px; font-weight: 300; color: white;")
        
        self.lbl_unit = QLabel("ml")
        self.lbl_unit.setAlignment(Qt.AlignCenter)
        self.lbl_unit.setStyleSheet("font-size: 24px; color: #8e8e93;")
        
        self.lbl_sub_info = QLabel("고정 출수 모드")
        self.lbl_sub_info.setAlignment(Qt.AlignCenter)
        self.lbl_sub_info.setStyleSheet("font-size: 14px; color: #636366;")

        disp_layout.addWidget(self.lbl_status)
        disp_layout.addWidget(self.lbl_main_value)
        disp_layout.addWidget(self.lbl_unit)
        disp_layout.addWidget(self.lbl_sub_info)
        
        main_layout.addWidget(self.display_frame)

        # 3. 컨트롤 패널 (버튼 영역)
        control_layout = QGridLayout()
        control_layout.setSpacing(15)

        # --- Section A: Smart Ratio (지능형 출수) ---
        lbl_ratio = QLabel("SMART FILL (Ratio)", objectName="title")
        lbl_ratio.setProperty("class", "section_title")
        control_layout.addWidget(lbl_ratio, 0, 0, 1, 3)

        self.btn_r1 = self.create_button("50%", lambda: self.set_mode('RATIO', 0.5))
        self.btn_r2 = self.create_button("65%", lambda: self.set_mode('RATIO', 0.65))
        self.btn_r3 = self.create_button("90%", lambda: self.set_mode('RATIO', 0.90))
        
        control_layout.addWidget(self.btn_r1, 1, 0)
        control_layout.addWidget(self.btn_r2, 1, 1)
        control_layout.addWidget(self.btn_r3, 1, 2)

        # --- Section B: Fixed Volume (고정 출수) ---
        lbl_vol = QLabel("FIXED VOLUME", objectName="title")
        lbl_vol.setStyleSheet("color: #8e8e93; font-size: 14px; font-weight: bold; margin-top: 20px;")
        control_layout.addWidget(lbl_vol, 2, 0, 1, 3)

        self.btn_v1 = self.create_button("120ml", lambda: self.set_mode('VOLUME', 120))
        self.btn_v2 = self.create_button("250ml", lambda: self.set_mode('VOLUME', 250))
        self.btn_v3 = self.create_button("500ml", lambda: self.set_mode('VOLUME', 500))

        control_layout.addWidget(self.btn_v1, 3, 0)
        control_layout.addWidget(self.btn_v2, 3, 1)
        control_layout.addWidget(self.btn_v3, 3, 2)

        # --- Section C: Control Pad (Right Side) ---
        # 상하 조절 버튼을 우측에 세로로 길게 배치
        self.btn_up = QPushButton("▲")
        self.btn_up.clicked.connect(lambda: self.adjust_value(1))
        self.btn_up.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 0.5); border: none; border-top-left-radius: 15px; border-top-right-radius: 15px; border-bottom-left-radius: 0; border-bottom-right-radius: 0; font-size: 24px;
            }
            QPushButton:pressed { background-color: #0a84ff; }
        """)
        
        self.btn_down = QPushButton("▼")
        self.btn_down.clicked.connect(lambda: self.adjust_value(-1))
        self.btn_down.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 0.5); border: none; border-bottom-left-radius: 15px; border-bottom-right-radius: 15px; border-top-left-radius: 0; border-top-right-radius: 0; font-size: 24px;
            }
            QPushButton:pressed { background-color: #0a84ff; }
        """)

        # 우측 열(Column 3)에 상하 버튼 배치 (Row 1~3 span)
        control_layout.addWidget(self.btn_up, 1, 3, 1, 1)   # Ratio 옆
        control_layout.addWidget(self.btn_down, 3, 3, 1, 1) # Volume 옆
        
        # 버튼 높이 조정
        self.btn_up.setMinimumHeight(60)
        self.btn_down.setMinimumHeight(60)

        main_layout.addLayout(control_layout)
        
        # 4. 하단 실행 버튼 (물방울 모양 등)
        self.btn_pour = QPushButton("DISPENSE")
        self.btn_pour.setFixedHeight(70)
        self.btn_pour.setStyleSheet("""
            QPushButton {
                background-color: #0a84ff;
                border: none;
                border-radius: 35px;
                font-size: 22px;
                font-weight: bold;
                color: white;
            }
            QPushButton:pressed {
                background-color: #0070d1;
            }
        """)
        self.btn_pour.clicked.connect(self.dispense_water)
        main_layout.addWidget(self.btn_pour)

    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.setMinimumHeight(60)
        btn.clicked.connect(callback)
        return btn

    # ==========================================================
    # 로직 (Logic) 구현
    # ==========================================================
    
    def set_mode(self, mode, value):
        """사용자가 버튼을 눌러 모드와 값을 변경했을 때"""
        self.current_mode = mode
        self.current_value = float(value)
        self.update_display()

    def adjust_value(self, direction):
        """
        방향키(▲, ▼) 로직
        - direction: +1 (Up), -1 (Down)
        - RATIO 모드: 10% (0.1) 단위 증감
        - VOLUME 모드: 50ml 단위 증감
        """
        if self.current_mode == 'RATIO':
            # 10% 단위 조절, 범위: 10% ~ 100%
            new_val = self.current_value + (0.1 * direction)
            new_val = max(0.1, min(1.0, new_val)) # Clamp 0.1 ~ 1.0
            self.current_value = round(new_val, 2)
            
        elif self.current_mode == 'VOLUME':
            # 50ml 단위 조절, 범위: 50ml ~ 2000ml
            new_val = self.current_value + (50 * direction)
            new_val = max(50, new_val)
            self.current_value = int(new_val)
            
        self.update_display()

    def update_display(self):
        """현재 모드와 값에 따라 디스플레이 갱신"""
        if self.current_mode == 'RATIO':
            percentage = int(self.current_value * 100)
            self.lbl_main_value.setText(f"{percentage}")
            self.lbl_unit.setText("% (Ratio)")
            self.lbl_sub_info.setText("스캔된 컵 용량 대비 비율")
            self.lbl_status.setText("SMART MODE")
            self.lbl_status.setStyleSheet("color: #30d158; font-size: 16px; font-weight: bold;")
            
        else: # VOLUME
            self.lbl_main_value.setText(f"{int(self.current_value)}")
            self.lbl_unit.setText("ml")
            self.lbl_sub_info.setText("고정 정량 출수")
            self.lbl_status.setText("MANUAL MODE")
            self.lbl_status.setStyleSheet("color: #0a84ff; font-size: 16px; font-weight: bold;")

    def dispense_water(self):
        """출수 실행 (DISPENSE 버튼 클릭 시 최종 계산 및 출력)"""
        final_volume = 0.0
        
        if self.current_mode == 'VOLUME':
            # 고정 모드: 현재 값 그대로 사용
            final_volume = self.current_value
            print(f"{int(final_volume)}") # 프롬프트 출력
            
        elif self.current_mode == 'RATIO':
            # 비율 모드: 추론 엔진 실행
            self.lbl_status.setText("SCANNING...")
            QApplication.processEvents() # UI 갱신
            
            # 데이터 파일 확인
            if not os.path.exists(self.DATA_PATH):
                print(f"[Error] 데이터 없음: {self.DATA_PATH}")
                self.lbl_sub_info.setText("Error: No Sensor Data")
                return

            if self.estimator:
                try:
                    # 부피 추정
                    result = self.estimator.predict(self.DATA_PATH)
                    cup_vol = result['volume_ml']
                    
                    # 비율 계산
                    final_volume = cup_vol * self.current_value
                    
                    # 결과 표시
                    self.lbl_sub_info.setText(f"Cup: {cup_vol}ml / Target: {int(final_volume)}ml")
                    
                    # [요구사항] 프롬프트 출력
                    print(f"{final_volume:.2f}")
                    
                except Exception as e:
                    print(f"[Error] Inference failed: {e}")
            else:
                print("[Error] No Inference Engine")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # High DPI Scaling (해상도 높은 화면 대응)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    window = ModernDispenserUI()
    window.show()
    sys.exit(app.exec_())