// [수정됨] LCD 라이브러리 및 관련 변수 제거 완료

const int flowSensor = 2;   // 유량 센서 핀
const int IN1 = 5;          // 모터 드라이버 핀
const int IN2 = 4;          // 모터 드라이버 핀 
const int ENA = 6;          // 모터 속도 제어(PWM)

// 물리 버튼 핀
const int START_BTN = 8;    // 시작 버튼
const int STOP_BTN  = 9;    // 정지 버튼

// 유량 센서 보정값 (LCD용 별도 상수는 제거함)
const float PULSES_PER_LITER = 565.0f;

// 변수 초기화
volatile unsigned long pulseCount = 0;
float total_mL_FLOW = 0.0f; 
unsigned long lastFlowMillis = 0;
const unsigned long flowInterval = 100;

// 펌프 상태 변수
bool runEnabled = false;

// 목표 출수량 (파이썬에서 받음)
float Limit_Flow = 0.0f; 

// 인터럽트 서비스 루틴
void pulseCounter() { 
  pulseCount++; 
}

void setup() {
  // 시리얼 통신 설정
  Serial.begin(9600); 
  Serial.setTimeout(50); 

  // 핀 모드 설정
  pinMode(flowSensor, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(flowSensor), pulseCounter, FALLING);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(STOP_BTN, INPUT_PULLUP);
  pinMode(START_BTN, INPUT_PULLUP);

  lastFlowMillis = millis();
}

void loop() {
  // 1. 시리얼 명령 수신 (파이썬 -> 아두이노)
  if (Serial.available() > 0) {
    float receivedValue = Serial.parseFloat(); 

    // 정지 신호 (-1)
    if (receivedValue == -1.0f) {
      runEnabled = false;
      Motor1_Brake(); 
      while(Serial.available()) { Serial.read(); } // 버퍼 비우기
    }
    // 출수 명령 (양수)
    else if (receivedValue > 0) {
      Limit_Flow = receivedValue;
      total_mL_FLOW = 0.0f; 
      runEnabled = true; 
    }
  }

  // 2. 물리 버튼 제어
  if (digitalRead(START_BTN) == LOW) {
    // 목표량이 설정 안 된 상태에서 물리버튼 누르면 기본 300ml 출수
    if (Limit_Flow <= 0) Limit_Flow = 300.0f; 
    total_mL_FLOW = 0.0f; 
    runEnabled = true;
  }
  
  if (digitalRead(STOP_BTN) == LOW) {
    runEnabled = false;
    Motor1_Brake();
  }

  // 3. 유량 계산 및 모터 제어 (0.1초 간격)
  unsigned long now = millis();

  if (now - lastFlowMillis >= flowInterval) {

    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    // 펄스를 ml로 변환
    float tick_FLOW = (float)pulses * (1000.0f / PULSES_PER_LITER);

    if (runEnabled) {
      total_mL_FLOW += tick_FLOW;     
    }

    // === 목표량 도달 체크 ===
    if (total_mL_FLOW >= Limit_Flow) {
      // 정상적으로 출수가 완료된 순간에만 신호 전송
      if (runEnabled) {
        Serial.println("DONE");  // 파이썬으로 완료 신호 전송
      }
      runEnabled = false;
      Motor1_Brake();
    }

    // 모터 구동
    if (runEnabled) {
      Motor1_Forward(255); // 최대 속도
    } else {
      Motor1_Brake();
    }

    lastFlowMillis = now;
  }
}

void Motor1_Forward(int Speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, Speed); 
}

void Motor1_Brake() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}