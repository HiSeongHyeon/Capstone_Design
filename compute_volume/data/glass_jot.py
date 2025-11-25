import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import measure, filters, morphology
from typing import List, Optional, Tuple, Dict, Union
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 1. Image Processing & Feature Extraction (Enhanced)
# ==========================================================
class FeatureExtractor:
    def __init__(self, shrink_ratio: float = 0.02, slices: int = 10):
        """
        Args:
            shrink_ratio: 마스크 침식 비율
            slices: Z축 슬라이싱 개수 (적층 적분용)
        """
        self.shrink_ratio = shrink_ratio
        self.slices = slices

    def get_pca_width(self, region_prop) -> float:
        try:
            if region_prop.minor_axis_length > 0:
                return region_prop.minor_axis_length
            else:
                min_r, min_c, max_r, max_c = region_prop.bbox
                return min(max_r - min_r, max_c - min_c)
        except:
            return 0.0

    def process(self, filename_or_data: Union[str, np.ndarray]) -> Optional[List[float]]:
        # 파일 경로 또는 numpy 배열 모두 처리 가능하도록 수정
        try:
            if isinstance(filename_or_data, str):
                data = np.load(filename_or_data)
                basename = os.path.basename(filename_or_data).lower()
            else:
                data = filename_or_data
                basename = "unknown"
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            return None
        
        valid_data = data[data > 0]
        if len(valid_data) == 0: return None
        
        thresh = np.mean(valid_data) + 0.2 * np.std(valid_data)
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
        patch_h, patch_w = final_mask_patch.shape
        full_mask[min_r:min_r+patch_h, min_c:min_c+patch_w] = final_mask_patch
        
        object_depths = data[full_mask]
        valid_depths = object_depths[object_depths > 0]
        if len(valid_depths) == 0: return None

        z_rim = np.percentile(valid_depths, 1) 
        z_bottom = np.percentile(valid_depths, 99)
        
        z_step = (z_bottom - z_rim) / self.slices
        sliced_volume_acc = 0.0
        
        if z_step > 0:
            for i in range(self.slices):
                z_start = z_rim + i * z_step
                z_end = z_rim + (i + 1) * z_step
                layer_pixels = np.sum((object_depths >= z_start) & (object_depths < z_end))
                sliced_volume_acc += layer_pixels * z_step
        
        valid_mask_z = valid_depths > z_rim
        z_v = valid_depths[valid_mask_z]
        avg_cubic = np.mean((z_v - z_rim)) if len(z_v) > 0 else 0
            
        pixels_filled = np.sum(full_mask)
        
        region_props_final = measure.regionprops(measure.label(full_mask))
        if not region_props_final: return None
        prop = region_props_final[0]
        
        pca_width = self.get_pca_width(prop)
        min_r_f, min_c_f, max_r_f, max_c_f = prop.bbox
        h_pixel = max_r_f - min_r_f
        
        geom_vol_pca = (pca_width ** 2) * h_pixel

        view_factor = h_pixel / pca_width if pca_width > 0 else 0
        shape_factor = (z_bottom - z_rim) / np.sqrt(pixels_filled) if pixels_filled > 0 else 0

        fill_rate = largest_region.area / pixels_filled 
        z_sigma = np.std(valid_depths)
        
        is_geom_label = 1.0 if any(x in basename for x in ['glass', 'tumbler', 'mug']) else 0.0

        return [sliced_volume_acc, geom_vol_pca, avg_cubic, shape_factor, view_factor, fill_rate, z_sigma, is_geom_label]

# ==========================================================
# 2. PyTorch Model Definition
# ==========================================================
class ViewpointGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense_depth = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.exp_depth = nn.Parameter(torch.tensor(1.0)) 
        
        self.dense_geom = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.exp_geom = nn.Parameter(torch.tensor(1.0))
        
        self.gate_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, 0.0)

    def forward(self, x):
        sliced_vol = x[:, 0].unsqueeze(1)
        geom_vol   = x[:, 1].unsqueeze(1)
        
        input_depth = torch.stack([x[:, 0], x[:, 2], x[:, 3], x[:, 4]], dim=1)
        input_geom  = torch.stack([x[:, 1], x[:, 4]], dim=1)
        
        gate_input = torch.stack([x[:, 5], x[:, 6]], dim=1)
        alpha = torch.sigmoid(self.gate_net(gate_input))
        
        scale_d = torch.nn.functional.softplus(self.dense_depth(input_depth))
        pred_depth = scale_d * (sliced_vol ** self.exp_depth)
        
        scale_g = torch.nn.functional.softplus(self.dense_geom(input_geom))
        pred_geom = scale_g * (geom_vol ** self.exp_geom)
        
        final_pred = (1 - alpha) * pred_depth + (alpha) * pred_geom
        
        return final_pred.squeeze(1), alpha

# ==========================================================
# 3. Inference Engine (New Class for Deployment)
# ==========================================================
class VolumeEstimator:
    def __init__(self, model_path: str):
        """
        학습된 모델(.pth)을 로드하여 추론 준비
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # [Fix for PyTorch 2.6+] weights_only=False 설정으로 Numpy 객체 로드 허용
        # stats 딕셔너리에 Numpy array가 포함되어 있어 기본값(True)에서는 에러 발생
        try:
            checkpoint = torch.load(model_path, weights_only=False)
        except TypeError:
            # 구버전 PyTorch 호환 (weights_only 인자가 없는 경우)
            checkpoint = torch.load(model_path)
        
        # 1. Load Config & Stats
        self.stats = checkpoint['stats']
        self.config = checkpoint.get('config', {'shrink_ratio': 0.03, 'slices': 10})
        
        # 2. Init Feature Extractor
        self.extractor = FeatureExtractor(
            shrink_ratio=self.config['shrink_ratio'], 
            slices=self.config['slices']
        )
        
        # 3. Load Model
        self.model = ViewpointGatedModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"[System] Model loaded from {model_path}")

    def predict(self, npy_input: Union[str, np.ndarray]) -> Dict:
        """
        입력: .npy 파일 경로 또는 numpy array
        출력: 예측 부피(ml), 모드(Geom/Depth), 신뢰도
        """
        # 1. Extract Features
        feats = self.extractor.process(npy_input)
        if feats is None:
            return {"error": "Feature extraction failed"}
            
        # feats: [sliced, geom, avg_d, shape, view, fill, sig, label]
        # Inference 시 label(마지막 인자)은 사용하지 않거나 dummy 처리
        x_raw = np.array(feats[:-1], dtype=np.float32) # label 제외 (Shape: 7)
        
        # 2. Normalize (Must match training logic exactly)
        # 전역 통계(Mean/Std) 정규화
        # [Fix] 훈련 시에도 feats[:-1]로 통계를 냈으므로 Shape가 (7,)로 일치함. 슬라이싱 제거.
        x_norm = (x_raw - self.stats['mean']) / self.stats['std']
        
        # 물리적 특성(부피 관련)은 Max Scaling으로 덮어쓰기
        # (학습 때 로직: X_scaled[:, 0] = X_raw[:, 0] / stats['vol_max'])
        x_norm[0] = x_raw[0] / (self.stats['vol_max'] + 1e-6)
        x_norm[1] = x_raw[1] / (self.stats['geom_max'] + 1e-6)
        
        # 3. Model Inference
        input_tensor = torch.from_numpy(x_norm).unsqueeze(0) # Batch dim 추가
        
        with torch.no_grad():
            pred_val, pred_alpha = self.model(input_tensor)
            
        # 4. Denormalize & Format
        pred_ml = pred_val.item() * self.stats['y_max']
        alpha = pred_alpha.item()
        
        mode = "GEOM" if alpha > 0.5 else "DEPTH"
        confidence = alpha if mode == "GEOM" else (1 - alpha)
        
        return {
            "volume_ml": round(pred_ml, 2),
            "mode": mode,
            "confidence_percent": round(confidence * 100, 1),
            "raw_features": x_raw.tolist()
        }

# ==========================================================
# 4. Training Pipeline with Save
# ==========================================================
class VolumeDataset(Dataset):
    def __init__(self, X, y, labels):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1)
        self.labels = torch.FloatTensor(labels).reshape(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.labels[idx]

def train_pipeline(data_dir: str = '.', epochs: int = 10000, save_path='volume_model.pth'):
    # --- 1. Data Prep ---
    extractor = FeatureExtractor(shrink_ratio=0.03, slices=10)
    files = glob.glob(os.path.join(data_dir, '*.npy'))
    
    data_list, target_list, label_list = [], [], []

    print(f"[Train] Loading {len(files)} files...")
    for f in files:
        try:
            vol_str = os.path.basename(f).split('_')[0]
            if not vol_str.replace('.', '', 1).isdigit(): continue
            feats = extractor.process(f)
            if feats:
                data_list.append(feats[:-1]) # drop label from features
                label_list.append(feats[-1]) # use label for gate
                target_list.append(float(vol_str))
        except: pass

    if len(data_list) < 5:
        print("Not enough data.")
        return

    # --- 2. Normalization Stats Calculation ---
    X_raw = np.array(data_list, dtype=np.float32)
    y_raw = np.array(target_list, dtype=np.float32)
    
    # 통계치 계산 (추론을 위해 저장 필수)
    stats = {
        'mean': X_raw.mean(axis=0),
        'std': X_raw.std(axis=0) + 1e-6,
        'vol_max': X_raw[:, 0].max(), # Sliced Volume Max
        'geom_max': X_raw[:, 1].max(), # Geom Volume Max
        'y_max': y_raw.max()          # Target Volume Max
    }
    
    # Apply Normalization
    X_scaled = (X_raw - stats['mean']) / stats['std']
    X_scaled[:, 0] = X_raw[:, 0] / (stats['vol_max'] + 1e-6)
    X_scaled[:, 1] = X_raw[:, 1] / (stats['geom_max'] + 1e-6)
    y_scaled = y_raw / stats['y_max']

    # --- 3. Train ---
    dataset = VolumeDataset(X_scaled, y_scaled, label_list)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    model = ViewpointGatedModel()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    bce_loss = nn.BCELoss()
    
    print("[Train] Training Started...")
    for epoch in range(epochs):
        for batch_X, batch_y, batch_lbl in dataloader:
            optimizer.zero_grad()
            pred_vol, pred_alpha = model(batch_X)
            loss = torch.mean(torch.abs((pred_vol - batch_y)/(batch_y+1e-6))) + 0.2 * bce_loss(pred_alpha.squeeze(), batch_lbl)
            loss.backward()
            optimizer.step()
            
        if epoch % 1000 == 0:
            print(f"Ep {epoch}: Loss {loss.item():.4f}")

    # --- 4. Save Model & Stats ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'config': {'shrink_ratio': 0.03, 'slices': 10}
    }, save_path)
    print(f"[System] Model saved to {save_path}")

# ==========================================================
# 5. Execution Example
# ==========================================================
if __name__ == "__main__":
    # A. 모델 학습 및 저장
    # 데이터가 있는 경우에만 실행
    if len(glob.glob('*.npy')) > 5:
        train_pipeline(epochs=5000, save_path='my_cup_model.pth')

        # B. 추론 (Inference) 테스트
        print("\n[Test] Running Inference...")
        estimator = VolumeEstimator('my_cup_model.pth')
        
        test_files = glob.glob('*.npy')[:3]
        for f in test_files:
            result = estimator.predict(f)
            print(f"File: {os.path.basename(f)} -> {result['volume_ml']}ml ({result['mode']})")
    else:
        print("데이터(.npy)가 부족하여 학습을 건너뜁니다.")