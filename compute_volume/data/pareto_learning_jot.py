import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os
from scipy import ndimage
from skimage import measure

# ==========================================
# 1. Feature Extraction (물리적 지표 추출)
# ==========================================
def extract_features(filename):
    try:
        data = np.load(filename)
    except:
        return None
    
    # 전처리
    flat_data = data.flatten()
    hist, bins = np.histogram(flat_data, bins=100)
    # 배경값 추출 로직
    bg_idx = np.argmax(hist)
    bg_val = (bins[bg_idx] + bins[bg_idx+1]) / 2
    
    mask = data < (bg_val - 15)
    labels = measure.label(mask)
    if labels.max() == 0: return None
    largest_region = max(measure.regionprops(labels), key=lambda r: r.area)
    mask = labels == largest_region.label
    mask = ndimage.binary_fill_holes(mask)
    
    object_depths = data[mask]
    if len(object_depths) == 0: return None
    
    # ------------------------------------------------
    # 핵심 Feature 계산
    # ------------------------------------------------
    # 1. Z_rim (거리)
    z_rim = np.percentile(object_depths, 5)
    
    # 2. Shape Factor (형상)
    z_max = np.max(object_depths)
    pixels = len(object_depths)
    if pixels == 0: return None
    shape_factor = (z_max - z_rim) / np.sqrt(pixels)
    
    # 3. Cubic Index Phys (물리적 부피 지표)
    # 공식: Sum( (Z - Z_rim) * Z^2 )
    valid_mask = object_depths > z_rim
    z_valid = object_depths[valid_mask]
    cubic_index_phys = np.sum((z_valid - z_rim) * (z_valid**2))

    return [z_rim, shape_factor, cubic_index_phys]

# ==========================================
# 2. PyTorch 모델 (PINN 구조)
# ==========================================
class PhysicalVolumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 파라미터 초기화
        self.w_z = nn.Parameter(torch.tensor(0.0)) # 거리 가중치
        self.w_s = nn.Parameter(torch.tensor(0.0)) # 형상 가중치
        self.bias = nn.Parameter(torch.tensor(1.0)) # 기본 스케일
        self.exponent = nn.Parameter(torch.tensor(1.0)) # 지수 (보통 부피는 선형비례하므로 1.0 근처)

    def forward(self, x):
        # x columns: [z_rim_norm, shape_norm, index_norm]
        z = x[:, 0]
        s = x[:, 1]
        idx = x[:, 2]
        
        # 수식: Volume = Scale(z, s) * (Index ^ exponent)
        # Softplus: 스케일이 음수가 되는 것을 방지 (log(1 + exp(x)))
        scale = torch.nn.functional.softplus(self.w_z * z + self.w_s * s + self.bias)
        
        # 예측값 계산 (0~1 사이로 예측됨)
        vol_pred = scale * (idx ** self.exponent)
        return vol_pred

# ==========================================
# 3. 데이터 로드 및 전처리
# ==========================================
files = glob.glob('*.npy')
data = []
targets = []
filenames = []

print(f"파일 검색 중... {os.getcwd()}")
for f in files:
    try:
        # 파일명 규칙: "508_mug.npy" -> 508.0
        vol = float(os.path.basename(f).split('_')[0])
        feats = extract_features(f)
        if feats:
            data.append(feats)
            targets.append(vol)
            filenames.append(f)
            print(f"Loaded: {f} -> {vol}ml")
    except Exception as e:
        print(f"Error {f}: {e}")
        continue

if len(data) < 2:
    print("데이터가 부족합니다. (최소 2개 이상)")
    exit()

# Numpy 변환
X_raw = np.array(data, dtype=np.float32)
y_raw = np.array(targets, dtype=np.float32).reshape(-1, 1)

# --- 스케일링 (Normalization) 중요 ---
# 1. 입력 변수 (Z-score)
z_mean, z_std = X_raw[:, 0].mean(), X_raw[:, 0].std()
s_mean, s_std = X_raw[:, 1].mean(), X_raw[:, 1].std()
# 분모가 0이 되는 것 방지
if z_std == 0: z_std = 1.0
if s_std == 0: s_std = 1.0

# 2. 입력 변수 (MinMax) - Index
idx_max = X_raw[:, 2].max()

# 3. 타겟 변수 (MinMax) - Volume **[추가된 핵심]**
# 정답(y)도 0~1 사이로 맞춰야 학습이 안정적입니다.
y_max = y_raw.max()

X_scaled = np.zeros_like(X_raw)
X_scaled[:, 0] = (X_raw[:, 0] - z_mean) / z_std
X_scaled[:, 1] = (X_raw[:, 1] - s_mean) / s_std
X_scaled[:, 2] = X_raw[:, 2] / idx_max

y_scaled = y_raw / y_max

# Tensor 변환
X_tensor = torch.from_numpy(X_scaled)
y_tensor = torch.from_numpy(y_scaled)

# ==========================================
# 4. 학습 (Training)
# ==========================================
model = PhysicalVolumeModel()
# Adam Optimizer 사용, 학습률 0.01로 상향
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n[학습 시작]")
epochs = 12000  # 100만번은 너무 많음, 1만번이면 충분

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward
    y_pred_scaled = model(X_tensor).unsqueeze(1)
    
    # Loss: MAPE
    loss = torch.mean(torch.abs((y_pred_scaled - y_tensor) / y_tensor))
    
    # Backward
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | MAPE Loss: {loss.item()*100:.2f}%")

# ==========================================
# 5. 결과 평가 및 복원
# ==========================================
model.eval()
with torch.no_grad():
    # 예측 (0~1 범위)
    pred_scaled = model(X_tensor).numpy()
    
    # 복원 (Real Scale)
    pred_real = pred_scaled * y_max

print("\n" + "="*60)
print(f"{'Filename':<20} | {'Real':<6} | {'Pred':<6} | {'Error':<8}")
print("-" * 60)

total_mape = 0
for i in range(len(filenames)):
    real = y_raw[i][0]
    pred = pred_real[i]
    
    # 오차율 계산
    err_percent = abs(pred - real) / real * 100
    total_mape += err_percent
    
    print(f"{filenames[i]:<20} | {real:<6.0f} | {pred:<6.1f} | {err_percent:<6.1f}%")

print("-" * 60)
print(f"평균 오차율(MAPE): {total_mape / len(filenames):.2f}%")

# ==========================================
# 6. 파라미터 출력 (Application 적용용)
# ==========================================
print("\n[ 최종 적용 파라미터 (이 값을 복사해서 사용하세요) ]")
print("1. Preprocessing Constants:")
print(f"   IDX_MAX  = {idx_max:.6e}")
print(f"   Y_MAX    = {y_max:.6f}")
print(f"   Z_MEAN   = {z_mean:.6f}, Z_STD = {z_std:.6f}")
print(f"   SHP_MEAN = {s_mean:.6f}, SHP_STD = {s_std:.6f}")
print("\n2. Model Weights:")
print(f"   w_z (Dist) = {model.w_z.item():.6f}")
print(f"   w_s (Shape)= {model.w_s.item():.6f}")
print(f"   bias       = {model.bias.item():.6f}")
print(f"   exponent   = {model.exponent.item():.6f}")