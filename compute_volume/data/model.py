import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import measure, morphology
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
import copy

# ==========================================================
# [설정] 고정 ROI 및 하이퍼파라미터
# ==========================================================
FIXED_ROI = (14, 30, 165, 151)

def plot_parity(y_true, y_pred, title, filename):
    """(기존과 동일) Parity Plot 생성"""
    plt.figure(figsize=(10, 8), dpi=100)
    plt.scatter(y_pred, y_true, alpha=0.6, edgecolors='w', s=80, c='#007AFF')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', lw=2, label='Ideal (y=x)')
    plt.xlabel('Estimated Volume (ml)', fontsize=18, fontweight='bold')
    plt.ylabel('Actual Volume (ml)', fontsize=18, fontweight='bold')
    plt.title(title, fontsize=20, pad=15, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Graph] Saved parity plot to: {filename}")

# [추가] 학습 곡선 및 게이트 변화 시각화 함수
def plot_learning_curves(history, filename="learning_curves.png"):
    """
    [수정됨] 상단 여백을 확보하여 제목과 범례 겹침 현상 해결
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 세로 길이를 조금 늘려 여백 확보 (10, 6 -> 10, 7)
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # 왼쪽 축: Loss (MAPE)
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss (MAPE)', color=color, fontsize=15)
    ax1.plot(epochs, history['train_loss'], color=color, alpha=0.6, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='darkred', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 오른쪽 축: Gate Alpha
    lines_2, labels_2 = [], []
    if 'val_gate_mean' in history:
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Gate Alpha (0:Depth <-> 1:Geom)', color=color, fontsize=15)
        ax2.plot(epochs, history['val_gate_mean'], color=color, linewidth=2, label='Avg Gate (Val)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)
        lines_2, labels_2 = ax2.get_legend_handles_labels()

    # 2. 범례 위치 조정
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    
    # 범례를 그래프 '바로 위'에 배치 (bbox_to_anchor 조절)
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='lower center',             # 기준점: 하단 중앙
               bbox_to_anchor=(0.5, 1.02),     # 위치: 그래프 프레임 바로 위
               ncol=3, 
               frameon=True, 
               edgecolor='white')              # 깔끔하게 테두리 제거 효과

    # 3. 제목 위치 조정 (y 파라미터로 더 위로 올림)
    plt.title('Training Metrics & Model Behavior', fontsize=18, fontweight='bold', y=1.12)

    # 4. 레이아웃 여백 강제 지정 (핵심)
    # rect=[left, bottom, right, top] -> top을 0.9로 설정하여 상단 10%를 제목용으로 비워둠
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    plt.savefig(filename)
    plt.close()
    print(f"[Graph] Saved learning curves to: {filename}")

class FeatureExtractor:
    """(기존과 동일) 학습/추론 공용 특징 추출기"""
    def __init__(self, shrink_ratio: float = 0.02):
        self.shrink_ratio = shrink_ratio

    def get_pca_width(self, region_prop) -> float:
        try: return region_prop.minor_axis_length if region_prop.minor_axis_length > 0 else 0.0
        except: return 0.0

    def process(self, filename_or_data: Union[str, np.ndarray], roi: Optional[Tuple[int,int,int,int]] = None) -> Optional[List[float]]:
        # (기존 코드 생략 - 변경 없음)
        try:
            if isinstance(filename_or_data, str): data = np.load(filename_or_data)
            else: data = filename_or_data
        except: return None
        
        if roi is not None:
            x, y, w, h = roi
            h_img, w_img = data.shape
            x = max(0, x); y = max(0, y)
            w = min(w, w_img - x); h = min(h, h_img - y)
            data = data[y:y+h, x:x+w]
            if data.size < 100: return None

        valid_data = data[data > 0]
        if len(valid_data) == 0: return None

        thresh = 1.5*np.mean(valid_data) + 0.2 * np.std(valid_data)
        mask = data < thresh
        mask = morphology.binary_opening(mask, morphology.disk(1))
        
        labels = measure.label(mask)
        if labels.max() == 0: return None
        
        props = measure.regionprops(labels)
        largest_region = max(props, key=lambda r: r.area)
        
        mask_convex = morphology.convex_hull_image(largest_region.image)
        bbox_h = largest_region.bbox[2] - largest_region.bbox[0]
        bbox_w = largest_region.bbox[3] - largest_region.bbox[1]
        img_size = max(bbox_h, bbox_w)
        shrink_pixels = max(1, int(img_size * self.shrink_ratio))
        mask_eroded = morphology.binary_erosion(mask_convex, morphology.disk(shrink_pixels))
        final_mask_patch = mask_eroded if np.sum(mask_eroded) > 0 else mask_convex

        min_r, min_c, max_r, max_c = largest_region.bbox
        full_mask = np.zeros_like(mask)
        full_mask[min_r:min_r+final_mask_patch.shape[0], min_c:min_c+final_mask_patch.shape[1]] = final_mask_patch

        object_depths = data[full_mask]
        valid_depths = object_depths[object_depths > 0]
        if len(valid_depths) == 0: return None

        z_rim = np.percentile(valid_depths, 1) 
        z_bottom = np.percentile(valid_depths, 99)
        
        pixel_volumes = (valid_depths - z_rim) * (valid_depths ** 2)
        pixel_volumes = pixel_volumes[pixel_volumes > 0]
        raw_volume_depth = np.sum(pixel_volumes) if len(pixel_volumes) > 0 else 0.0

        z_mid = (z_rim + z_bottom) / 2.0
        top_mask = full_mask & (data < z_mid) & (data > 0)
        if np.sum(top_mask) < 50:
             z_mid_relaxed = z_rim + (z_bottom - z_rim) * 0.8
             top_mask = full_mask & (data < z_mid_relaxed) & (data > 0)

        labels_top = measure.label(top_mask)
        if labels_top.max() > 0: prop_pca = max(measure.regionprops(labels_top), key=lambda r: r.area)
        else: prop_pca = measure.regionprops(measure.label(full_mask))[0]

        pca_width = self.get_pca_width(prop_pca)
        prop_full = measure.regionprops(measure.label(full_mask))[0]
        h_pixel = prop_full.bbox[2] - prop_full.bbox[0]
        raw_volume_geom = (pca_width ** 2) * h_pixel

        avg_depth = np.mean(valid_depths)
        pixels_filled = np.sum(full_mask)
        shape_factor = (z_bottom - z_rim) / np.sqrt(pixels_filled) if pixels_filled > 0 else 0
        view_factor = h_pixel / pca_width if pca_width > 0 else 0
        n_convex = np.sum(mask_convex)
        fill_rate = np.sum(mask) / n_convex if n_convex > 0 else 0
        z_sigma = np.std(valid_depths)

        return [raw_volume_depth, raw_volume_geom, avg_depth, shape_factor, view_factor, fill_rate, z_sigma]

class PhysicsGatedModel(nn.Module):
    """(기존과 동일)"""
    def __init__(self):
        super().__init__()
        self.fc_depth = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        self.exp_depth = nn.Parameter(torch.tensor(1.0)) 
        self.fc_geom = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        self.exp_geom = nn.Parameter(torch.tensor(1.0)) 
        self.gate_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        v_depth_raw = x[:, 0].unsqueeze(1)
        v_geom_raw  = x[:, 1].unsqueeze(1)
        inputs_kd = x[:, [0, 2, 3, 4]] 
        inputs_kg = x[:, [1, 4]]
        inputs_gate = x[:, [5, 6]]
        
        k_d = torch.nn.functional.softplus(self.fc_depth(inputs_kd))
        k_g = torch.nn.functional.softplus(self.fc_geom(inputs_kg))
        
        v_depth_base = torch.clamp(v_depth_raw, min=1e-6)
        v_geom_base = torch.clamp(v_geom_raw, min=1e-6)

        pred_depth = k_d * (v_depth_base ** self.exp_depth)
        pred_geom  = k_g * (v_geom_base  ** self.exp_geom)
        
        alpha_gate = torch.sigmoid(self.gate_mlp(inputs_gate))
        final_pred = (1 - alpha_gate) * pred_depth + alpha_gate * pred_geom
        return final_pred.squeeze(1), alpha_gate

def normalize_data(X_raw, y_raw, stats):
    """(기존과 동일)"""
    X_norm = (X_raw - stats['mean']) / stats['std']
    X_norm[:, 0] = X_raw[:, 0] / (stats['v_depth_max'] + 1e-6)
    X_norm[:, 1] = X_raw[:, 1] / (stats['v_geom_max'] + 1e-6)
    
    y_norm = None
    if y_raw is not None:
        y_norm = y_raw / stats['y_max']
    
    return X_norm, y_norm

def train_and_export(data_dir: str = '.', epochs: int = 5000):
    # 1. Data Preparation (기존과 동일)
    extractor = FeatureExtractor()
    files = glob.glob(os.path.join(data_dir, '*.npy'))
    data_list, target_list = [], []
    
    print(f"[System] Found {len(files)} files. Extracting features...")
    for f in files:
        try:
            vol_str = os.path.basename(f).split('_')[0]
            if not vol_str.replace('.', '', 1).isdigit(): continue
            feats = extractor.process(f, roi=FIXED_ROI)
            if feats:
                data_list.append(feats)
                target_list.append(float(vol_str))
        except: pass

    if not data_list: print("No data found."); return

    X_full = np.array(data_list, dtype=np.float32)
    y_full = np.array(target_list, dtype=np.float32)

    # 2. Data Splitting (기존과 동일)
    num_samples = len(X_full)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split1 = int(0.6 * num_samples)
    split2 = int(0.8 * num_samples)

    idx_train = indices[:split1]
    idx_val   = indices[split1:split2]
    idx_test  = indices[split2:]

    X_train_raw, y_train_raw = X_full[idx_train], y_full[idx_train]
    X_val_raw,   y_val_raw   = X_full[idx_val],   y_full[idx_val]
    X_test_raw,  y_test_raw  = X_full[idx_test],  y_full[idx_test]

    print(f"[Split] Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # 3. Stats (기존과 동일)
    stats = {
        'mean': X_train_raw.mean(axis=0), 
        'std': X_train_raw.std(axis=0) + 1e-6,
        'v_depth_max': X_train_raw[:, 0].max(), 
        'v_geom_max': X_train_raw[:, 1].max(), 
        'y_max': y_train_raw.max()
    }

    # 4. Normalization (기존과 동일)
    X_train, y_train = normalize_data(X_train_raw, y_train_raw, stats)
    X_val,   y_val   = normalize_data(X_val_raw,   y_val_raw,   stats)
    X_test,  y_test  = normalize_data(X_test_raw,  y_test_raw,  stats)

    # 5. DataLoaders (기존과 동일)
    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=len(X_train), shuffle=True)
    val_loader   = DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val)),   batch_size=len(X_val),   shuffle=False)

    # 6. Training (수정됨)
    model = PhysicsGatedModel()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0

    # [수정] 기록용 리스트
    history = {'train_loss': [], 'val_loss': [], 'val_gate_mean': []}

    print("[Train] Start Loop...")
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_batch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred, _ = model(bx)
            loss = torch.mean(torch.abs((pred - by) / (by + 1e-6))) # MAPE 형태의 Loss
            loss.backward()
            optimizer.step()
            train_batch_loss = loss.item()
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_bx, val_by = next(iter(val_loader))
            val_pred, val_alpha = model(val_bx)
            val_loss = torch.mean(torch.abs((val_pred - val_by) / (val_by + 1e-6)))
            
            # [수정] 기록
            history['train_loss'].append(train_batch_loss)
            history['val_loss'].append(val_loss.item())
            history['val_gate_mean'].append(val_alpha.mean().item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        if epoch % 1000 == 0: 
            print(f"Ep {epoch}: Train Loss {train_batch_loss:.4f} | Val Loss {val_loss.item():.4f}")

    print(f"\n[Result] Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    
    # 7. Visualization & Final Evaluation (수정됨)
    # [수정] 학습 곡선 그래프 저장 함수 호출
    plot_learning_curves(history, "learning_curves.png")

    model.load_state_dict(best_weights)
    model.eval()
    
    with torch.no_grad():
        # Validation Set Graph
        val_in = torch.FloatTensor(X_val)
        val_target_norm = torch.FloatTensor(y_val)
        val_pred_norm, _ = model(val_in)
        
        val_pred_real = val_pred_norm.numpy() * stats['y_max']
        val_target_real = val_target_norm.numpy() * stats['y_max']
        
        plot_parity(val_target_real, val_pred_real, "Validation Set: Actual vs Estimated", "validation_parity_plot.png")
        
        # Test Set Evaluation
        t_in = torch.FloatTensor(X_test)
        t_target_norm = torch.FloatTensor(y_test)
        test_pred_norm, _ = model(t_in)
        
        test_pred_real = test_pred_norm.numpy() * stats['y_max']
        test_target_real = t_target_norm.numpy() * stats['y_max']
        
        errors = np.abs(test_pred_real - test_target_real)
        mape = np.mean(errors / (test_target_real + 1e-6)) * 100
        mae = np.mean(errors)

    print(f"[Test] Final Evaluation on {len(X_test)} unseen samples:")
    print(f"       -> MAPE: {mape:.2f}%")
    print(f"       -> MAE : {mae:.2f} ml")

    # 8. Export (기존과 동일)
    print("\n[Export] Saving BEST model weights to 'model_weights.npz'...")
    numpy_weights = {k: v.cpu().detach().numpy() for k, v in best_weights.items()}
    np.savez('model_weights.npz', weights=numpy_weights, stats=stats)
    print("Done.")

if __name__ == "__main__":
    train_and_export(epochs=15000)