#include <LiquidCrystal_I2C.h>
#include <Wire.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int flowSensor = 2;   // Flow sensor pin
const int IN1=5;  // motor pin
const int IN2=4;  // motor pin 
const int ENA=6;  // motor pin

// button pin
const int START_BTN = 8;   // START button
const int STOP_BTN  = 9;   // STOP button

// Lcd control 
const float PULSES_PER_LITER_LCD  = 660.0f;  

// Flow control 
const float PULSES_PER_LITER_FLOW = 565.0f;

// initialization
volatile unsigned long pulseCount = 0;
float total_mL_LCD  = 0.0f;     
float total_mL_FLOW = 0.0f; 
unsigned long lastFlowMillis = 0;
const unsigned long flowInterval = 100;

// pump initialization
bool runEnabled = false;

// Limit_Flow = Rpi5 input (기본값 0으로 설정, 시리얼 입력 대기)
float Limit_Flow = 0.0f; 

void pulseCounter() { 
  pulseCount++; 
}

void setup() {
  // [수정] 시리얼 통신 시작 (라즈베리파이와 통신용)
  Serial.begin(9600); 
  // 타임아웃 설정: parseFloat가 너무 오래 기다리지 않도록 (기본 1000ms -> 50ms)
  Serial.setTimeout(50); 

  pinMode(flowSensor, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(flowSensor), pulseCounter, FALLING);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(STOP_BTN, INPUT_PULLUP);
  pinMode(START_BTN, INPUT_PULLUP);

  lcd.init();
  lcd.backlight();
  lcd.clear();

  lastFlowMillis = millis();
}

void loop() {
  // [수정] 시리얼 데이터 수신 로직 강화
  if (Serial.available() > 0) {
    float receivedValue = Serial.parseFloat(); // 데이터 읽기

    // === 1. 정지 신호 수신 (-1) ===
    if (receivedValue == -1.0f) {
      runEnabled = false;
      Motor1_Brake(); // 즉시 정지
      
      // 버퍼에 남은 찌꺼기 데이터 비우기 (엔터키 등)
      while(Serial.available()) { Serial.read(); }
    }
    // === 2. 출수 명령 수신 (양수) ===
    else if (receivedValue > 0) {
      Limit_Flow = receivedValue;
      
      // 새로운 출수를 위해 유량 카운터 초기화
      total_mL_FLOW = 0.0f; 
      
      runEnabled = true; // 모터 작동 시작 플래그
    }
  }

  // 물리 버튼 로직 (기존 유지)
  if (digitalRead(START_BTN) == LOW) {
    if (Limit_Flow <= 0) Limit_Flow = 300.0f; 
    total_mL_FLOW = 0.0f; 
    runEnabled = true;
  }
  
  if (digitalRead(STOP_BTN) == LOW) {
    runEnabled = false;
    Motor1_Brake();
  }

  unsigned long now = millis();

  if (now - lastFlowMillis >= flowInterval) {

    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    // LCD, Flow separate
    float tick_LCD  = (float)pulses * (1000.0f / PULSES_PER_LITER_LCD);
    float tick_FLOW = (float)pulses * (1000.0f / PULSES_PER_LITER_FLOW);

    if (runEnabled) {
      total_mL_LCD  += tick_LCD;      
      total_mL_FLOW += tick_FLOW;     
    }

    // Flow Control Check
    if (total_mL_FLOW >= Limit_Flow) {
      runEnabled = false;
      Motor1_Brake();
    }

    // 실제 모터 구동 명령 (타이머 내부에서도 상태 확인)
    if (runEnabled) {
      Motor1_Forward(255);
    } else {
      Motor1_Brake();
    }

    // Lcd display 
    lcd.setCursor(0, 0);
    lcd.print("Tot:      mL");
    lcd.setCursor(5, 0);
    lcd.print("    ");
    lcd.setCursor(5, 0);
    lcd.print((unsigned long)total_mL_LCD); 

    lcd.setCursor(0, 1);
    lcd.print("Set:          "); 
    lcd.setCursor(5, 1);
    lcd.print((int)Limit_Flow);
    
    // 상태 표시
    lcd.setCursor(10, 1);
    if (runEnabled) {
      lcd.print(" RUN  ");
    } else {
      lcd.print(" STOP ");
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
  analogWrite(ENA, 0); // PWM도 0으로 확실히 끔
}