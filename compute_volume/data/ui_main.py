import sys
import os
import datetime # 날짜 계산용
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGridLayout, QFrame, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QFontDatabase

# === 추론 엔진 Dummy ===
try:
    class DummyVolumeEstimator:
        def __init__(self, model_path):
            print(f"[시스템] 더미 추론 엔진 초기화: {model_path}")
        def predict(self, data_path):
            print(f"[시스템] 더미 데이터 스캔: {data_path}")
            return {'volume_ml': 350.0}
    try:
        from inference import VolumeEstimator
    except ImportError:
        VolumeEstimator = DummyVolumeEstimator 
except Exception as e:
    print(f"[시스템] 추론 엔진 초기화 중 오류 발생: {e}")
    VolumeEstimator = None


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
    
    # PyQt5 호환성을 위한 폰트 웨이트 상수 정의
    WEIGHT_NORMAL = 50
    WEIGHT_MEDIUM = 57
    WEIGHT_BOLD = 75
    WEIGHT_BLACK = 87

    def __init__(self):
        super().__init__()
        
        # === 시스템 설정 ===
        self.MODEL_PATH = "my_cup_model.pth"
        self.DATA_PATH = "380_4000_paper.npy"
        
        # === 상태 관리 ===
        self.current_mode = 'VOLUME'  
        self.current_value = 120.0    
        
        self.init_inference_engine()
        self.init_ui()
        
        # 버튼 그룹화
        self.ratio_buttons = [self.btn_r1, self.btn_r2, self.btn_r3]
        self.volume_buttons = [self.btn_v1, self.btn_v2, self.btn_v3]

        # 초기 상태 업데이트
        self.update_display()
        self.highlight_button('VOLUME', 120.0)

    def init_inference_engine(self):
        if VolumeEstimator:
            try:
                self.estimator = VolumeEstimator(self.MODEL_PATH)
            except Exception as e:
                print(f"[오류] 모델 로드 실패: {e}")
        else:
            print("[시스템] 경고: 추론 엔진 로드 불가.")

    def load_custom_fonts(self):
        font_db = QFontDatabase()
        for font_file in self.FONT_FILES:
            if os.path.exists(font_file):
                font_db.addApplicationFont(font_file)

    def init_ui(self):
        """밝고 고급스러운 Premium Light UI"""
        self.setWindowTitle("Premium Water Dispenser")
        self.setGeometry(100, 100, 850, 550) # 위젯 추가로 공간 확보를 위해 높이/너비 약간 증가
        self.load_custom_fonts()

        # === Global Stylesheet ===
        self.setStyleSheet(f"""
            QWidget {{
                font-family: "{self.MAIN_FONT_FAMILY}";
                background-color: #F2F4F7;
                color: #333333;
            }}
            
            /* 좌측 정보 표시 패널 */
            QFrame#display_frame {{
                background-color: #FFFFFF;
                border-radius: 30px;
                border: 1px solid #E5E5EA;
            }}
            
            /* 상단 정보 카드 (새로 추가됨) */
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
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_h_layout = QHBoxLayout(central_widget)
        main_h_layout.setContentsMargins(40, 40, 40, 40)
        main_h_layout.setSpacing(30)

        # ---------------------------------------------------------
        # 1. 좌측 디스플레이 영역
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 2. 우측 컨트롤 패널
        # ---------------------------------------------------------
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(20)
        right_panel_layout.setContentsMargins(0, 10, 0, 10)

        control_grid = QGridLayout()
        control_grid.setVerticalSpacing(12) # 간격 미세 조정
        control_grid.setHorizontalSpacing(12)

        # === [NEW] 상단 정보 위젯 (Row 1, Col 0~2) ===
        # 오늘 날짜 기반 계산
        today = datetime.date.today()
        next_care = today + datetime.timedelta(days=15)
        next_care_str = next_care.strftime("%m.%d") # 예: 11.25

        # 1. 사용시간 (Filter/Usage)
        self.card_usage = self.create_info_card("사용 기간", "10개월", "#1C1C1E")
        control_grid.addWidget(self.card_usage, 2, 0)

        # 2. 다음 관리일 (D-Day)
        self.card_date = self.create_info_card("다음 관리", f"{next_care_str}", "#007AFF")
        control_grid.addWidget(self.card_date, 2, 1)

        # 3. 물 온도 (Cold)
        self.card_temp = self.create_info_card("물 온도", "4°C", "#30D158") # 시원한 느낌의 Green/Blue
        control_grid.addWidget(self.card_temp, 2, 2)


        # === 기존 버튼 재배치 ===
        
        # Up/Down Control (Row 1~4, Col 3) - 위치 유지
        self.btn_up = QPushButton("▲")
        self.btn_up.clicked.connect(lambda: self.adjust_value(1))
        self.btn_up.setProperty("class", "control_button")
        
        self.btn_down = QPushButton("▼")
        self.btn_down.clicked.connect(lambda: self.adjust_value(-1))
        self.btn_down.setProperty("class", "control_button")

        # 버튼 높이 자동 조절을 위해 Expanding 정책 사용
        self.btn_up.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.btn_down.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        # 최소 높이 설정 (정보 카드와 높이 맞춤)
        self.btn_up.setMinimumHeight(60)
        self.btn_down.setMinimumHeight(60)

        control_grid.addWidget(self.btn_up, 5, 3, 2, 1)   # Row 1, 2 차지
        control_grid.addWidget(self.btn_down, 7, 3, 2, 1) # Row 3, 4 차지
        
        # -- Section 1: Smart Fill (Ratio) --
        # 겹치지 않게 Row 5부터 시작하도록 밀어줌
        lbl_ratio = QLabel("SMART FILL")
        lbl_ratio.setProperty("class", "section_title")
        control_grid.addWidget(lbl_ratio, 5, 0, 1, 4) # Row 5

        self.btn_r1 = self.create_button("50%", lambda: self.set_mode('RATIO', 0.5))
        self.btn_r2 = self.create_button("75%", lambda: self.set_mode('RATIO', 0.75))
        self.btn_r3 = self.create_button("90%", lambda: self.set_mode('RATIO', 0.90))

        control_grid.addWidget(self.btn_r1, 6, 0) # Row 6
        control_grid.addWidget(self.btn_r2, 6, 1)
        control_grid.addWidget(self.btn_r3, 6, 2)

        # -- Section 2: Fixed Volume --
        lbl_vol = QLabel("PRESET VOLUME")
        lbl_vol.setProperty("class", "section_title")
        control_grid.addWidget(lbl_vol, 7, 0, 1, 4) # Row 7

        self.btn_v1 = self.create_button("120ml", lambda: self.set_mode('VOLUME', 120))
        self.btn_v2 = self.create_button("250ml", lambda: self.set_mode('VOLUME', 250))
        self.btn_v3 = self.create_button("500ml", lambda: self.set_mode('VOLUME', 500))
        
        control_grid.addWidget(self.btn_v1, 8, 0) # Row 8
        control_grid.addWidget(self.btn_v2, 8, 1)
        control_grid.addWidget(self.btn_v3, 8, 2)

        # 레이아웃 비율 설정
        control_grid.setColumnStretch(0, 1)
        control_grid.setColumnStretch(1, 1)
        control_grid.setColumnStretch(2, 1)
        control_grid.setColumnStretch(3, 0) # 화살표 버튼은 폭 고정

        right_panel_layout.addLayout(control_grid)
        right_panel_layout.addStretch(1) 

        # -- Dispense Button --
        self.btn_pour = QPushButton("DISPENSE")
        self.btn_pour.setObjectName("dispense_button")
        self.btn_pour.setFixedHeight(65)
        self.btn_pour.setCursor(Qt.PointingHandCursor)
        self.btn_pour.clicked.connect(self.dispense_water)
        
        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(15)
        btn_shadow.setColor(QColor(0, 122, 255, 80)) 
        btn_shadow.setOffset(0, 4)
        self.btn_pour.setGraphicsEffect(btn_shadow)

        right_panel_layout.addWidget(self.btn_pour)

        main_h_layout.addLayout(right_panel_layout)

    def create_info_card(self, title, value, value_color):
        """작은 정보 표시용 카드 위젯 생성"""
        frame = QFrame()
        frame.setProperty("class", "info_card")
        frame.setFixedHeight(85) # 높이 고정
        
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
            self.lbl_sub_info.setText("용기 대비 출수량")
            self.lbl_status.setText("SMART FILL")
            self.lbl_status.setStyleSheet("background-color: transparent; color: #30D158; letter-spacing: 1px;")
        else:
            self.lbl_main_value.setText(f"{int(self.current_value)}")
            self.lbl_unit.setText("ml")
            self.lbl_sub_info.setText("설정 용량 출수")
            self.lbl_status.setText("MANUAL MODE")
            self.lbl_status.setStyleSheet("background-color: transparent; color: #007AFF; letter-spacing: 1px;")

    def dispense_water(self):
        print(f"Dispensing {self.current_value} ({self.current_mode})")

if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    window = ModernDispenserUI()
    window.show()
    sys.exit(app.exec_())