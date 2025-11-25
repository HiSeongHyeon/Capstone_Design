import numpy as np
import torch
import torch.nn as nn
import os
from skimage import measure, morphology
from typing import List, Optional, Dict, Union
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 1. Feature Extractor (학습 코드와 동일해야 함)
# ==========================================================
class FeatureExtractor:
    def __init__(self, shrink_ratio: float = 0.02, slices: int = 10):
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
        #의 전처리 로직 그대로 유지
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
        
        # Inference 시 label은 dummy 값
        is_geom_label = 0.0 

        return [sliced_volume_acc, geom_vol_pca, avg_cubic, shape_factor, view_factor, fill_rate, z_sigma, is_geom_label]

# ==========================================================
# 2. Model Definition (로드할 때 아키텍처가 필요함)
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
        # Bias 초기화 주의
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
# 3. Inference Wrapper
# ==========================================================
class VolumeEstimator:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # [중요] PyTorch 버전에 따라 weights_only 옵션 처리
        try:
            checkpoint = torch.load(model_path, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path)
        
        # 저장된 통계치(Mean/Std/Max) 로드
        self.stats = checkpoint['stats']
        self.config = checkpoint.get('config', {'shrink_ratio': 0.03, 'slices': 10})
        
        # Feature Extractor 초기화 (저장된 config 사용)
        self.extractor = FeatureExtractor(
            shrink_ratio=self.config['shrink_ratio'], 
            slices=self.config['slices']
        )
        
        # 모델 로드
        self.model = ViewpointGatedModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() # 평가 모드 전환

    def predict(self, npy_input: Union[str, np.ndarray]) -> Dict:
        # 1. Feature Extraction
        feats = self.extractor.process(npy_input)
        if feats is None:
            return {"error": "Feature extraction failed"}
            
        # feats: [sliced, geom, avg_d, shape, view, fill, sig, label]
        x_raw = np.array(feats[:-1], dtype=np.float32) 
        
        # 2. Normalization (학습 때 저장한 stats 사용)
        x_norm = (x_raw - self.stats['mean']) / self.stats['std']
        
        # 물리 부피값은 Max Scaling (학습 로직과 일치시킴)
        x_norm[0] = x_raw[0] / (self.stats['vol_max'] + 1e-6)
        x_norm[1] = x_raw[1] / (self.stats['geom_max'] + 1e-6)
        
        # 3. Inference
        input_tensor = torch.from_numpy(x_norm).unsqueeze(0) # Batch Size 1
        
        with torch.no_grad():
            pred_val, pred_alpha = self.model(input_tensor)
            
        # 4. Denormalize & Result
        pred_ml = pred_val.item() * self.stats['y_max']
        alpha = pred_alpha.item()
        
        mode = "GEOM" if alpha > 0.5 else "DEPTH"
        confidence = alpha if mode == "GEOM" else (1 - alpha)
        
        return {
            "volume_ml": round(pred_ml, 2),
            "mode": mode,
            "confidence_percent": round(confidence * 100, 1),
            "alpha_value": round(alpha, 4)
        }

# ==========================================================
# 실행 예시
# ==========================================================
if __name__ == "__main__":
    # 1. 모델 경로 설정 (학습 후 생성된 파일)
    MODEL_PATH = "my_cup_model.pth" 
    
    # 2. 테스트할 npy 파일 경로
    TEST_FILE = "test_cup_data.npy" 

    # 3. 추론 실행
    try:
        estimator = VolumeEstimator(MODEL_PATH)
        
        # 파일이 실제로 존재한다고 가정하고 실행 (또는 numpy 배열 직접 전달 가능)
        if os.path.exists(TEST_FILE):
            result = estimator.predict(TEST_FILE)
            print("-" * 30)
            print(f"File: {TEST_FILE}")
            print(f"Estimated Volume: {result['volume_ml']} ml")
            print(f"Algorithm Used  : {result['mode']} (Weight: {result['confidence_percent']}%)")
            print("-" * 30)
        else:
            print(f"[Warning] Test file '{TEST_FILE}' not found.")
            print("To test, please place a .npy file path in TEST_FILE variable.")

    except Exception as e:
        print(f"[Error] {e}")