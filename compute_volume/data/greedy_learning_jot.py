import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os
from scipy import ndimage
from skimage import measure

# ==========================================
# 1. Feature Extraction (변수 5개 추출)
# ==========================================
def extract_all_features(filename):
    try:
        data = np.load(filename)
    except:
        return None
    
    # ------------------------------------------------
    # 전처리 (Segmentation)
    # ------------------------------------------------
    flat_data = data.flatten()
    hist, bins = np.histogram(flat_data, bins=100)
    bg_idx = np.argmax(hist)
    bg_val = (bins[bg_idx] + bins[bg_idx+1]) / 2
    
    thresh = bg_val - 15
    mask = data < thresh
    
    labels = measure.label(mask)
    if labels.max() == 0: return None
    largest_region = max(measure.regionprops(labels), key=lambda r: r.area)
    mask = labels == largest_region.label
    mask = ndimage.binary_fill_holes(mask)
    
    object_depths = data[mask]
    if len(object_depths) == 0: return None
    
    # ------------------------------------------------
    # [1] 기존 Feature
    # ------------------------------------------------
    z_rim = np.percentile(object_depths, 5)
    
    z_max = np.max(object_depths)
    pixels = len(object_depths)
    shape_factor = (z_max - z_rim) / np.sqrt(pixels) if pixels > 0 else 0
    
    valid_mask = object_depths > z_rim
    z_valid = object_depths[valid_mask]
    cubic_index = np.sum((z_valid - z_rim) * (z_valid**2))

    # ------------------------------------------------
    # [2] 신규 추가 Feature (렌즈 왜곡 & 재질 보정)
    # ------------------------------------------------
    # A. Radial Distance (중심에서 떨어진 거리) - 렌즈 왜곡 보정용
    # 이미지 센터 좌표
    cy, cx = data.shape[0] / 2, data.shape[1] / 2
    # 물체 무게중심 좌표
    obj_cy, obj_cx = ndimage.center_of_mass(mask)
    # 유클리드 거리 계산
    r_dist = np.sqrt((obj_cy - cy)**2 + (obj_cx - cx)**2)
    
    # B. Depth Sigma (깊이 표준편차) - 재질(종이 vs 도자기) 추정용
    # 표면이 매끄러운지, 노이즈가 많은지를 나타냄
    z_sigma = np.std(object_depths)

    # 반환: [Z, Shape, R_dist, Z_sigma, Index]
    return [z_rim, shape_factor, r_dist, z_sigma, cubic_index]

# ==========================================
# 2. PyTorch 모델 (항이 5개로 늘어남)
# ==========================================
class FullPhysicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 가중치 파라미터 5개 (초기값 0.0)
        self.w_z = nn.Parameter(torch.tensor(0.0)) # 거리
        self.w_s = nn.Parameter(torch.tensor(0.0)) # 형상
        self.w_r = nn.Parameter(torch.tensor(0.0)) # 렌즈왜곡 (New)
        self.w_n = nn.Parameter(torch.tensor(0.0)) # 재질노이즈 (New)
        
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.exponent = nn.Parameter(torch.tensor(0.62)) # 이전 학습값 참고

    def forward(self, x):
        # x: [z, shape, r_dist, z_sigma, index] (Normalized)
        z = x[:, 0]
        s = x[:, 1]
        r = x[:, 2] # Radial Dist
        n = x[:, 3] # Noise/Sigma
        idx = x[:, 4]
        
        # 선형 결합: 거리 + 형상 + 렌즈위치 + 재질
        linear_comb = (self.w_z * z) + (self.w_s * s) + (self.w_r * r) + (self.w_n * n) + self.bias
        
        # Scale Factor (Softplus로 양수 보장)
        scale = torch.nn.functional.softplus(linear_comb)
        
        # 최종 부피
        vol = scale * (idx ** self.exponent)
        return vol

# ==========================================
# 3. 데이터 로드 및 정규화
# ==========================================
files = glob.glob('*.npy')
data_list = []
target_list = []
filenames = []

print("데이터 로딩 및 Feature 추출 중...")
for f in files:
    try:
        vol = float(os.path.basename(f).split('_')[0])
        feats = extract_all_features(f)
        if feats:
            data_list.append(feats)
            target_list.append(vol)
            filenames.append(f)
            print(f" -> {f}: Features extracted.")
    except Exception as e:
        print(f"Skip {f}: {e}")

if len(data_list) < 2:
    print("데이터 부족.")
    exit()

X_raw = np.array(data_list, dtype=np.float32)
y_raw = np.array(target_list, dtype=np.float32).reshape(-1, 1)

# --- Normalization Stats 저장 (나중에 C++/Python 적용 시 필요) ---
stats = {
    'z_mean': X_raw[:, 0].mean(), 'z_std': X_raw[:, 0].std() + 1e-6,
    's_mean': X_raw[:, 1].mean(), 's_std': X_raw[:, 1].std() + 1e-6,
    'r_mean': X_raw[:, 2].mean(), 'r_std': X_raw[:, 2].std() + 1e-6,
    'n_mean': X_raw[:, 3].mean(), 'n_std': X_raw[:, 3].std() + 1e-6,
    'idx_max': X_raw[:, 4].max(),
    'y_max': y_raw.max()
}

# 데이터 정규화 적용
X_scaled = np.zeros_like(X_raw)
X_scaled[:, 0] = (X_raw[:, 0] - stats['z_mean']) / stats['z_std']
X_scaled[:, 1] = (X_raw[:, 1] - stats['s_mean']) / stats['s_std']
X_scaled[:, 2] = (X_raw[:, 2] - stats['r_mean']) / stats['r_std']
X_scaled[:, 3] = (X_raw[:, 3] - stats['n_mean']) / stats['n_std']
X_scaled[:, 4] = X_raw[:, 4] / stats['idx_max']

y_scaled = y_raw / stats['y_max']

# Tensor 변환
X_tensor = torch.from_numpy(X_scaled)
y_tensor = torch.from_numpy(y_scaled)

# ==========================================
# 4. 학습 (Training)
# ==========================================
model = FullPhysicalModel()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("\n[풀 옵션 모델 학습 시작]")
epochs = 15000

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_tensor).unsqueeze(1)
    loss = torch.mean(torch.abs((y_pred - y_tensor) / y_tensor)) # MAPE
    loss.backward()
    optimizer.step()
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | MAPE: {loss.item()*100:.2f}%")

# ==========================================
# 5. 최종 결과 확인
# ==========================================
model.eval()
with torch.no_grad():
    pred_norm = model(X_tensor).numpy()
    pred_real = pred_norm * stats['y_max']

print("\n" + "="*60)
print(f"{'Filename':<20} | {'Real':<5} | {'Pred':<5} | {'Error':<8}")
print("-" * 60)

total_err = 0
for i in range(len(filenames)):
    real = y_raw[i][0]
    pred = pred_real[i]
    err = abs(pred - real) / real * 100
    total_err += err
    print(f"{filenames[i]:<20} | {real:<5.0f} | {pred:<5.1f} | {err:<6.2f}%")

print("-" * 60)
print(f"최종 평균 오차(MAPE): {total_err/len(filenames):.2f}%")

# ==========================================
# 6. 파라미터 출력 (적용을 위해 복사하세요)
# ==========================================
print("\n[ Updated Parameters for Application ]")
print("1. Stats:")
for k, v in stats.items():
    print(f"   {k.upper():<10} = {v:.6f}")

print("\n2. Weights:")
print(f"   W_Z (Dist)   = {model.w_z.item():.6f}")
print(f"   W_S (Shape)  = {model.w_s.item():.6f}")
print(f"   W_R (Radial) = {model.w_r.item():.6f}  <-- New!")
print(f"   W_N (Noise)  = {model.w_n.item():.6f}  <-- New!")
print(f"   BIAS         = {model.bias.item():.6f}")
print(f"   EXPONENT     = {model.exponent.item():.6f}")