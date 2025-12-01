import numpy as np
import glob
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import measure, filters, morphology
from typing import List, Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# [설정] 고정 ROI (x, y, w, h)
# 모든 데이터는 이 영역만큼 잘라서 분석됩니다.
# ==========================================================
FIXED_ROI = (14, 30, 165, 151)

# ==========================================================
# 0. Research Debugger (Visualization Module)
# ==========================================================
class ResearchDebugger:
    """연구 및 디버깅을 위한 시각화 전담 클래스"""
    @staticmethod
    def show_preprocessing_steps(original, mask, mask_convex, mask_eroded, final_mask, bbox, title="Preprocessing"):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # 1. Raw Depth (Cropped)
        axes[0].imshow(original, cmap='viridis')
        min_r, min_c, max_r, max_c = bbox
        rect = Rectangle((min_c, min_r), max_c - min_c, max_r - min_r, fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title("1. Raw Depth (ROI Cropped)")
        
        # 2. Binary Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("2. Initial Mask")

        # 3. Convex Hull
        axes[2].imshow(mask_convex, cmap='gray')
        axes[2].set_title("3. Convex Hull (Anchor)")
        
        # 4. Eroded Mask
        axes[3].imshow(mask_eroded, cmap='gray')
        axes[3].set_title("4. Eroded")
        
        # 5. Final Patch
        axes[4].imshow(final_mask, cmap='magma')
        axes[4].set_title("5. Final Input Mask")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_geometric_analysis(full_mask, top_mask, valid_depths, region_prop, z_rim, z_bottom, slices):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # --- [Subplot 1] PCA Axis on TOP MASK ---
        y0, x0 = region_prop.centroid
        orientation = region_prop.orientation
        
        axes[0].imshow(full_mask, cmap='gray', alpha=0.3)
        axes[0].imshow(top_mask, cmap='autumn', alpha=0.7) 
        axes[0].plot(x0, y0, '.g', markersize=10)
        
        x1 = x0 + math.cos(orientation) * 0.5 * region_prop.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * region_prop.minor_axis_length
        x2 = x0 - math.cos(orientation) * 0.5 * region_prop.minor_axis_length
        y2 = y0 + math.sin(orientation) * 0.5 * region_prop.minor_axis_length
        axes[0].plot((x1, x2), (y1, y2), '-b', linewidth=3, label='PCA Width (Body)')
        
        axes[0].set_title(f"PCA Analysis (on Top Slice)\nWidth: {region_prop.minor_axis_length:.1f} px")
        axes[0].legend()

        # --- [Subplot 2] Cup Profile ---
        rows, cols = np.indices(full_mask.shape)
        masked_rows = rows[full_mask > 0]; masked_cols = cols[full_mask > 0]
        radii = np.sqrt((masked_rows - y0)**2 + (masked_cols - x0)**2)
        
        axes[1].scatter(radii, valid_depths, s=1, alpha=0.3, c='purple')
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Radius"); axes[1].set_ylabel("Depth")
        axes[1].set_title("Cup Profile (Full Height)")

        # --- [Subplot 3] Z-Slicing ---
        axes[2].hist(valid_depths, bins=50, color='skyblue', alpha=0.7)
        axes[2].axvline(z_rim, color='green', linewidth=2, label='Rim')
        axes[2].axvline(z_bottom, color='blue', linewidth=2, label='Bottom')
        axes[2].legend(); axes[2].set_title("Depth Histogram")

        plt.tight_layout(); plt.show()

    @staticmethod
    def show_training_analysis(loss_history, gt_vols, pred_vols, alphas):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1); plt.plot(loss_history); plt.title("Training Loss"); plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        sc = plt.scatter(gt_vols, pred_vols, c=alphas, cmap='coolwarm', alpha=0.8, edgecolors='k')
        plt.plot([min(gt_vols), max(gt_vols)], [min(gt_vols), max(gt_vols)], 'k--', alpha=0.5)
        plt.colorbar(sc, label='Alpha'); plt.title("GT vs Pred"); plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(alphas, bins=20, range=(0,1), color='purple', alpha=0.7, edgecolor='black')
        plt.title("Alpha Distribution"); plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.show()

# ==========================================================
# 1. Feature Extractor (Fixed ROI & Physics Logic)
# ==========================================================
class FeatureExtractor:
    def __init__(self, shrink_ratio: float = 0.02, slices: int = 10, debug: bool = False):
        self.shrink_ratio = shrink_ratio
        self.slices = slices
        self.debug = debug

    def get_pca_width(self, region_prop) -> float:
        try:
            return region_prop.minor_axis_length if region_prop.minor_axis_length > 0 else 0.0
        except: return 0.0

    def process(self, filename_or_data: Union[str, np.ndarray], roi: Optional[Tuple[int,int,int,int]] = None) -> Optional[List[float]]:
        try:
            if isinstance(filename_or_data, str):
                data = np.load(filename_or_data)
                basename = os.path.basename(filename_or_data).lower()
            else:
                data = filename_or_data
                basename = "unknown"
        except: return None
        
        # ---------------------------------------------------------
        # [CRITICAL] Apply Fixed ROI Crop
        # 데이터 로드 직후, 분석 전에 무조건 크롭 수행
        # ---------------------------------------------------------
        if roi is not None:
            x, y, w, h = roi
            # 이미지 경계 체크
            h_img, w_img = data.shape
            x = max(0, x); y = max(0, y)
            w = min(w, w_img - x); h = min(h, h_img - y)
            
            # Crop
            data = data[y:y+h, x:x+w]
            
            if data.size < 100: # 크롭 후 데이터가 너무 작으면 실패 처리
                return None
        # ---------------------------------------------------------

        valid_data = data[data > 0]
        if len(valid_data) == 0: return None

        # 1. Masking Strategy
        thresh = 1.5*np.mean(valid_data) + 0.2 * np.std(valid_data)
        mask = data < thresh
        mask = morphology.binary_opening(mask, morphology.disk(1))
        
        labels = measure.label(mask)
        # 유리컵 Fallback
        if labels.max() == 0 or max([r.area for r in measure.regionprops(labels)]) < 200:
            bg_depth = np.percentile(valid_data, 80)
            mask = data < (bg_depth - 2.0)
            mask = morphology.binary_opening(mask, morphology.disk(1))
            labels = measure.label(mask)

        if labels.max() == 0: return None
        
        props = measure.regionprops(labels)
        largest_region = max(props, key=lambda r: r.area)
        
        # 2. Convex Hull (Anchor)
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
        
        if self.debug:
            ResearchDebugger.show_preprocessing_steps(
                data, mask, mask_convex, mask_eroded, full_mask, 
                largest_region.bbox, title=f"Preprocessing: {basename}"
            )

        # 3. Feature Extraction
        object_depths = data[full_mask]
        valid_depths = object_depths[object_depths > 0]
        if len(valid_depths) == 0: return None

        z_rim = np.percentile(valid_depths, 1) 
        z_bottom = np.percentile(valid_depths, 99)
        
        # [Modified] Z^2 Weighted Volume Integration
        pixel_volumes = (valid_depths - z_rim) * (valid_depths ** 2)
        pixel_volumes = pixel_volumes[pixel_volumes > 0]
        raw_volume_depth = np.sum(pixel_volumes) if len(pixel_volumes) > 0 else 0.0

        # [Modified] Dual Mask Strategy for PCA
        z_mid = (z_rim + z_bottom) / 2.0
        top_mask = full_mask & (data < z_mid) & (data > 0)
        
        if np.sum(top_mask) < 50:
             z_mid_relaxed = z_rim + (z_bottom - z_rim) * 0.8
             top_mask = full_mask & (data < z_mid_relaxed) & (data > 0)

        labels_top = measure.label(top_mask)
        if labels_top.max() > 0:
            prop_pca = max(measure.regionprops(labels_top), key=lambda r: r.area)
        else:
            prop_pca = measure.regionprops(measure.label(full_mask))[0]
            top_mask = full_mask

        if self.debug:
            ResearchDebugger.show_geometric_analysis(
                full_mask, top_mask, valid_depths, prop_pca, z_rim, z_bottom, self.slices
            )
        
        pca_width = self.get_pca_width(prop_pca)
        
        region_props_full = measure.regionprops(measure.label(full_mask))
        prop_full = region_props_full[0]
        h_pixel = prop_full.bbox[2] - prop_full.bbox[0]
        
        raw_volume_geom = (pca_width ** 2) * h_pixel

        # Correction Parameters
        avg_depth = np.mean(valid_depths)
        pixels_filled = np.sum(full_mask)
        shape_factor = (z_bottom - z_rim) / np.sqrt(pixels_filled) if pixels_filled > 0 else 0
        view_factor = h_pixel / pca_width if pca_width > 0 else 0
        
        n_raw = np.sum(mask)
        n_convex = np.sum(mask_convex)
        fill_rate = n_raw / n_convex if n_convex > 0 else 0
        z_sigma = np.std(valid_depths)

        return [raw_volume_depth, raw_volume_geom, avg_depth, shape_factor, view_factor, fill_rate, z_sigma]

# ==========================================================
# 2. Physics-Informed Model
# ==========================================================
class PhysicsGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Depth Expert
        self.fc_depth = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        self.exp_depth = nn.Parameter(torch.tensor(1.0)) 
        
        # Geom Expert
        self.fc_geom = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        self.exp_geom = nn.Parameter(torch.tensor(1.0)) 
        
        # Gating (2-Layer MLP)
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

# ==========================================================
# 3. Volume Estimator
# ==========================================================
class VolumeEstimator:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try: checkpoint = torch.load(model_path, weights_only=False)
        except: checkpoint = torch.load(model_path)
        
        self.stats = checkpoint['stats']
        self.extractor = FeatureExtractor(shrink_ratio=0.03, slices=10)
        self.model = PhysicsGatedModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"[System] Model loaded from {model_path}")
        self.analyze_model_weights()

    def analyze_model_weights(self):
        print("\n" + "="*50)
        print(" [Model Weight Analysis] ")
        print("="*50)
        n_depth = self.model.exp_depth.item()
        n_geom = self.model.exp_geom.item()
        print(f"1. Physics Exponents (n)")
        print(f"   - Depth Mode: {n_depth:.4f}")
        print(f"   - Geom Mode : {n_geom:.4f}")
        print("-" * 50)
        w_depth = self.model.fc_depth[0].weight.detach().numpy()
        importance_depth = np.mean(np.abs(w_depth), axis=0)
        print(f"2. Depth Expert Importance (Kd)")
        for name, val in zip(["V_depth", "Avg Z", "Shape", "View"], importance_depth):
            print(f"   - {name:<12}: {val:.4f}")
        print("="*50 + "\n")

    def predict(self, npy_input: Union[str, np.ndarray], debug: bool = False) -> Dict:
        self.extractor.debug = debug
        # [ROI 적용] 설정된 FIXED_ROI를 사용하여 처리
        feats = self.extractor.process(npy_input, roi=FIXED_ROI)
        
        if feats is None: return {"error": "Extraction failed"}
        
        x_raw = np.array(feats, dtype=np.float32)
        x_norm = (x_raw - self.stats['mean']) / self.stats['std']
        x_norm[0] = x_raw[0] / (self.stats['v_depth_max'] + 1e-6)
        x_norm[1] = x_raw[1] / (self.stats['v_geom_max'] + 1e-6)
        
        input_tensor = torch.from_numpy(x_norm).unsqueeze(0)
        
        with torch.no_grad():
            pred_val, pred_alpha = self.model(input_tensor)
            
        pred_ml = pred_val.item() * self.stats['y_max']
        alpha = pred_alpha.item()
        mode = "GEOM" if alpha > 0.5 else "DEPTH"
        
        return {
            "volume_ml": round(pred_ml, 2),
            "mode": mode,
            "confidence": round((alpha if mode=="GEOM" else 1-alpha)*100, 1),
            "alpha": round(alpha, 3)
        }

# ==========================================================
# 4. Training Pipeline
# ==========================================================
class VolumeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_pipeline(data_dir: str = '.', epochs: int = 5000):
    extractor = FeatureExtractor(shrink_ratio=0.03, slices=10, debug=False)
    files = glob.glob(os.path.join(data_dir, '*.npy'))
    print(f"[Info] Found {len(files)} files.")
    
    data_list, target_list, filenames = [], [], []
    for f in files:
        try:
            vol_str = os.path.basename(f).split('_')[0]
            if not vol_str.replace('.', '', 1).isdigit(): continue
            # 학습 시에도 FIXED_ROI 적용 권장 (데이터 일관성 위함)
            # 단, 학습 데이터가 이미 crop된 상태라면 roi=None으로 두어야 함.
            # 여기서는 일관성을 위해 ROI 적용 (필요 없으면 None으로 변경)
            feats = extractor.process(f, roi=FIXED_ROI) 
            if feats:
                data_list.append(feats)
                target_list.append(float(vol_str)); filenames.append(f)
        except: pass

    if len(data_list) < 5: print("[Error] Need more data."); return

    X_raw = np.array(data_list, dtype=np.float32)
    y_raw = np.array(target_list, dtype=np.float32)
    
    stats = {
        'mean': X_raw.mean(axis=0), 'std': X_raw.std(axis=0) + 1e-6,
        'v_depth_max': X_raw[:, 0].max(), 'v_geom_max': X_raw[:, 1].max(), 'y_max': y_raw.max()
    }
    
    X_norm = (X_raw - stats['mean']) / stats['std']
    X_norm[:, 0] = X_raw[:, 0] / (stats['v_depth_max'] + 1e-6)
    X_norm[:, 1] = X_raw[:, 1] / (stats['v_geom_max'] + 1e-6)
    y_norm = y_raw / stats['y_max']

    dataset = VolumeDataset(X_norm, y_norm)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    model = PhysicsGatedModel(); optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    
    print(f"[Train] Start Training...")
    for epoch in range(1, epochs + 1):
        for bx, by in dataloader:
            optimizer.zero_grad(); pred_vol, _ = model(bx)
            loss = torch.mean(torch.abs((pred_vol - by) / (by + 1e-6)))
            loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        if epoch % 500 == 0: print(f"Ep {epoch}: Loss {loss.item():.4f}")

    torch.save({'model_state_dict': model.state_dict(), 'stats': stats}, 'volume_model_fixed.pth')
    
    # Visualization Analysis
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_norm)
        pred_vol, pred_alpha = model(inputs)
        ResearchDebugger.show_training_analysis(loss_history, y_raw, pred_vol.numpy().flatten() * stats['y_max'], pred_alpha.numpy().flatten())

    # Evaluation Text
    print("\n" + "="*80)
    res_real = pred_vol.numpy().flatten() * stats['y_max']
    alpha_val = pred_alpha.numpy().flatten()
    mape_sum = 0
    for i in range(len(filenames)):
        err = abs(res_real[i] - y_raw[i]) / y_raw[i] * 100
        mape_sum += err
        mode = "GEOM" if alpha_val[i] > 0.5 else "DPTH"
        print(f"{os.path.basename(filenames[i]):<25} | {y_raw[i]:<6.0f} | {res_real[i]:<6.1f} | {err:<6.2f} | {mode}")
    print("-" * 80)
    print(f"Average MAPE: {mape_sum / len(filenames):.2f}%")

if __name__ == "__main__":
    # 1. 학습 실행
    if len(glob.glob('*.npy')) > 5:
        train_pipeline(epochs=50000)
    
    # 2. 추론 테스트 (고정 ROI 적용됨)
    if os.path.exists('volume_model_fixed.pth'):
        print("\n[Test] Running Inference with Fixed ROI...")
        estimator = VolumeEstimator('volume_model_fixed.pth')
        test_files = glob.glob('*.npy')[15:17] 
        for f in test_files:
            print(f"--- Checking {f} ---")
            res = estimator.predict(f, debug=True)
            print(f"Result: {res['volume_ml']}ml")