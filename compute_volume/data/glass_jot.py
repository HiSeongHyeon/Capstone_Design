import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import measure, filters, morphology

# ==========================================================
# 1. 개선된 Feature Extraction (Smart Segmentation 적용)
# ==========================================================
def extract_fusion_features_improved(filename):
    try:
        data = np.load(filename)
    except:
        return None
    
    # --- [Step 1] 전처리 및 마스크 추출 (Robust Masking) ---
    # 유효 데이터만 추출하여 Otsu 임계값 계산 (배경/물체 자동 분리)
    valid_data = data[data > 0]
    if len(valid_data) == 0: return None
    
    try:
        thresh = filters.threshold_otsu(valid_data)
    except:
        thresh = np.mean(valid_data) # Otsu 실패 시 평균값 사용

    # 물체 마스크 생성 (카메라와 가까운 쪽)
    mask = data < thresh
    
    # 노이즈 제거 (작은 점들 삭제)
    mask = morphology.binary_opening(mask, morphology.disk(1))

    # 가장 큰 덩어리만 남기기 (배경 잡음 제거)
    labels = measure.label(mask)
    if labels.max() == 0: return None
    
    props = measure.regionprops(labels)
    largest_region = max(props, key=lambda r: r.area)
    
    # --- [Step 2] 핵심: Convex Hull로 구멍 메우기 (유리컵 보정) ---
    # 유리컵은 가운데가 뚫려있으므로, 외곽선을 기준으로 내부를 채워줌
    mask_convex = largest_region.image
    mask_convex = morphology.convex_hull_image(mask_convex)
    
    # 전체 이미지 크기에 맞춰 마스크 복원
    min_r, min_c, max_r, max_c = largest_region.bbox
    full_mask = np.zeros_like(mask)
    full_mask[min_r:max_r, min_c:max_c] = mask_convex
    
    # --- [Step 3] Feature 계산 ---
    # 보정된 마스크 내의 깊이 데이터만 추출
    object_depths = data[full_mask]
    valid_depths = object_depths[object_depths > 0]
    
    if len(valid_depths) == 0: return None

    # 1. Depth Features (적분 방식)
    z_rim = np.percentile(valid_depths, 5)
    z_max = np.max(valid_depths)
    
    # Cubic Index: 구멍 난 데이터 대신, '평균 깊이 기여도' X '채워진 픽셀 수' 사용
    # 이 방식은 유리컵처럼 데이터가 듬성듬성해도 전체 부피를 잘 추정함
    valid_mask_z = valid_depths > z_rim
    z_v = valid_depths[valid_mask_z]
    
    if len(z_v) == 0: 
        avg_cubic_contribution = 0
    else:
        # 픽셀당 기여 부피: (거리 차) * (면적 가중치 Z^2)
        avg_cubic_contribution = np.mean((z_v - z_rim) * (z_v**2))
        
    pixels_filled = np.sum(full_mask) # Convex Hull로 채워진 픽셀 수
    cubic_index = avg_cubic_contribution * pixels_filled # 보정된 적분값
    
    # Shape Factor
    height = z_max - z_rim
    shape_factor = height / np.sqrt(pixels_filled) if pixels_filled > 0 else 0

    # 2. Geometric Features (원통 근사 방식)
    h_pixel = max_r - min_r
    w_pixel = max_c - min_c
    geom_index = (w_pixel ** 2) * h_pixel

    # 3. Gating Features (재질 판별용)
    # Fill Rate: (원래 잡힌 픽셀 수) / (Convex Hull로 채워진 픽셀 수)
    # 유리컵은 이 값이 낮고(구멍 많음), 머그컵은 높음(꽉 참)
    fill_rate = largest_region.area / pixels_filled
    z_sigma = np.std(valid_depths)

    # 4. Label Generation (파일명 기반 지도학습)
    basename = os.path.basename(filename).lower()
    # glass, tumbler, mug는 깊이 센서 오차가 크므로 '기하학 모드(1)'로 유도
    is_geom_label = 1.0 if ('glass' in basename or 'tumbler' in basename or 'mug' in basename) else 0.0

    return [cubic_index, geom_index, z_rim, shape_factor, fill_rate, z_sigma, is_geom_label]

# ==========================================================
# 2. Supervised Gated PINN 모델 정의
# ==========================================================
class SupervisedGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Expert 1: Depth Model (일반 도자기/종이컵용)
        self.w_z = nn.Parameter(torch.tensor(0.0))
        self.w_s = nn.Parameter(torch.tensor(0.0))
        self.b_depth = nn.Parameter(torch.tensor(1.0))
        self.exp_depth = nn.Parameter(torch.tensor(0.62))
        
        # Expert 2: Geometric Model (유리/텀블러/검은색용)
        self.b_geom = nn.Parameter(torch.tensor(0.01))
        self.exp_geom = nn.Parameter(torch.tensor(1.0)) # 기하학적 부피는 보통 선형 비례
        
        # Gate: 상황 판단기 (0: Depth, 1: Geom)
        self.gate_w_fill = nn.Parameter(torch.tensor(-1.0)) # Fill rate 낮으면 -> Geom(1)
        self.gate_w_sig  = nn.Parameter(torch.tensor(1.0))  # 노이즈 심하면 -> Geom(1)
        self.gate_bias   = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Input x: [idx_d, idx_g, z, s, fill, sig]
        idx_d, idx_g = x[:, 0], x[:, 1]
        z, s = x[:, 2], x[:, 3]
        fill, sig = x[:, 4], x[:, 5]
        
        # 1. Gate Calculation (Mode Selection)
        gate_logit = (self.gate_w_fill * fill) + (self.gate_w_sig * sig) + self.gate_bias
        alpha = torch.sigmoid(gate_logit) 
        
        # 2. Prediction from Experts
        # Depth Expert
        scale_d = torch.nn.functional.softplus(self.w_z * z + self.w_s * s + self.b_depth)
        pred_depth = scale_d * (idx_d ** self.exp_depth)
        
        # Geometric Expert
        scale_g = torch.nn.functional.softplus(self.b_geom)
        pred_geom = scale_g * (idx_g ** self.exp_geom)
        
        # 3. Weighted Sum (Sensor Fusion)
        final_pred = (1 - alpha) * pred_depth + (alpha) * pred_geom
        
        return final_pred, alpha

# ==========================================================
# 3. 학습 및 실행 파이프라인
# ==========================================================
if __name__ == "__main__":
    # 1. 데이터 로드
    files = glob.glob('*.npy')
    print(f"검색된 파일 개수: {len(files)}개")
    
    data_list, target_list, label_list, filenames = [], [], [], []

    for f in files:
        try:
            # 파일명에서 용량 추출 (예: 508_mug.npy -> 508)
            vol_str = os.path.basename(f).split('_')[0]
            if not vol_str.isdigit(): continue
            vol = float(vol_str)
            
            # 특징 추출
            feats = extract_fusion_features_improved(f)
            if feats:
                data_list.append(feats[:-1]) # 입력 변수
                label_list.append(feats[-1]) # 정답 라벨 (Mode)
                target_list.append(vol)      # 정답 부피
                filenames.append(f)
                print(f"Loaded: {f} (Mode: {'GEOM' if feats[-1] else 'DEPTH'})")
        except Exception as e:
            print(f"Skipped {f}: {e}")

    if len(data_list) < 1:
        print("학습할 데이터가 없습니다.")
        exit()

    # 2. 데이터 정규화 (Normalization)
    X_raw = np.array(data_list, dtype=np.float32)
    y_raw = np.array(target_list, dtype=np.float32).reshape(-1, 1)
    labels_raw = np.array(label_list, dtype=np.float32).reshape(-1, 1)

    stats = {
        'idx_d_max': X_raw[:, 0].max(),
        'idx_g_max': X_raw[:, 1].max(),
        'z_mean': X_raw[:, 2].mean(), 'z_std': X_raw[:, 2].std() + 1e-6,
        's_mean': X_raw[:, 3].mean(), 's_std': X_raw[:, 3].std() + 1e-6,
        'fill_mean': X_raw[:, 4].mean(), 'fill_std': X_raw[:, 4].std() + 1e-6,
        'sig_mean': X_raw[:, 5].mean(), 'sig_std': X_raw[:, 5].std() + 1e-6,
        'y_max': y_raw.max()
    }

    X_scaled = np.zeros_like(X_raw)
    # Scale specific features appropriately
    X_scaled[:, 0] = X_raw[:, 0] / stats['idx_d_max']       # Cubic Index
    X_scaled[:, 1] = X_raw[:, 1] / stats['idx_g_max']       # Geom Index
    X_scaled[:, 2] = (X_raw[:, 2] - stats['z_mean']) / stats['z_std']
    X_scaled[:, 3] = (X_raw[:, 3] - stats['s_mean']) / stats['s_std']
    X_scaled[:, 4] = (X_raw[:, 4] - stats['fill_mean']) / stats['fill_std']
    X_scaled[:, 5] = (X_raw[:, 5] - stats['sig_mean']) / stats['sig_std']

    y_scaled = y_raw / stats['y_max']

    # Tensor 변환
    X_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y_scaled)
    label_tensor = torch.from_numpy(labels_raw)

    # 3. 모델 학습
    model = SupervisedGatedModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bce_loss = nn.BCELoss() # 분류 오차 함수

    print("\n[학습 시작]")
    for epoch in range(50001):
        optimizer.zero_grad()
        pred_vol, pred_alpha = model(X_tensor)
        
        # Loss 1: 부피 오차 (MAPE)
        loss_vol = torch.mean(torch.abs((pred_vol.unsqueeze(1) - y_tensor) / y_tensor))
        
        # Loss 2: 모드 분류 오차 (파일명 라벨과 Gate 출력 비교)
        loss_gate = bce_loss(pred_alpha.unsqueeze(1), label_tensor)
        
        # Total Loss
        loss = loss_vol + loss_gate
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Total: {loss.item():.4f} (Vol: {loss_vol.item():.4f}, Gate: {loss_gate.item():.4f})")

    # 4. 최종 결과 평가
    model.eval()
    with torch.no_grad():
        pred_norm, alphas = model(X_tensor)
        pred_real = pred_norm.numpy() * stats['y_max']
        alphas = alphas.numpy()

    print("\n" + "="*75)
    print(f"{'Filename':<25} | {'Real':<6} | {'Pred':<6} | {'Error':<8} | {'Mode (Prob)'}")
    print("-" * 75)

    for i in range(len(filenames)):
        real = y_raw[i][0]
        pred = pred_real[i]
        err = abs(pred - real) / real * 100
        
        # Gate 확률에 따른 모드 표시
        prob = alphas[i]
        mode_str = "GEOM" if prob > 0.5 else "DEPTH"
        
        print(f"{filenames[i]:<25} | {real:<6.0f} | {pred:<6.1f} | {err:<6.2f}%  | {mode_str} ({prob:.2f})")
    print("-" * 75)