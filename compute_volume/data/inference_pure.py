import numpy as np
import os
import glob
from skimage import measure, morphology
from typing import List, Optional, Tuple, Dict, Union

# ==========================================================
# [설정] 고정 ROI (학습 코드와 동일해야 함)
# ==========================================================
FIXED_ROI = (14, 30, 165, 151)

class FeatureExtractor:
    """학습 코드와 완전히 동일한 로직을 가진 특징 추출기"""
    def __init__(self, shrink_ratio: float = 0.02):
        self.shrink_ratio = shrink_ratio

    def get_pca_width(self, region_prop) -> float:
        try: return region_prop.minor_axis_length if region_prop.minor_axis_length > 0 else 0.0
        except: return 0.0

    def process(self, filename_or_data: Union[str, np.ndarray], roi: Optional[Tuple[int,int,int,int]] = None) -> Optional[List[float]]:
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

class NumpyVolumePredictor:
    """PyTorch 의존성 없이 순수 NumPy로 구현된 예측 엔진"""
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found.")
        
        # Load .npz
        loaded = np.load(model_path, allow_pickle=True)
        # item()을 호출해야 0-d array 내부의 딕셔너리를 꺼낼 수 있음
        self.weights = loaded['weights'].item() 
        self.stats = loaded['stats'].item()
        
    def _relu(self, x): return np.maximum(0, x)
    
    def _softplus(self, x): return np.log(1 + np.exp(np.clip(x, -80, 80)))
    
    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    
    def _linear(self, x, w, b): return x @ w.T + b

    def predict(self, features: List[float]) -> Dict:
        # 1. Normalize
        x_raw = np.array(features, dtype=np.float32)
        x_norm = (x_raw - self.stats['mean']) / self.stats['std']
        x_norm[0] = x_raw[0] / (self.stats['v_depth_max'] + 1e-6)
        x_norm[1] = x_raw[1] / (self.stats['v_geom_max'] + 1e-6)
        x = x_norm.reshape(1, -1)

        # 2. Forward Pass (NumPy Implementation of PhysicsGatedModel)
        # Indices: [V_depth, V_geom, Avg_Z, Shape, View, Fill, Z_sigma]
        
        # Depth Expert
        inputs_kd = x[:, [0, 2, 3, 4]]
        h1_d = self._relu(self._linear(inputs_kd, self.weights['fc_depth.0.weight'], self.weights['fc_depth.0.bias']))
        k_d = self._softplus(self._linear(h1_d, self.weights['fc_depth.2.weight'], self.weights['fc_depth.2.bias']))
        
        # Geom Expert
        inputs_kg = x[:, [1, 4]]
        h1_g = self._relu(self._linear(inputs_kg, self.weights['fc_geom.0.weight'], self.weights['fc_geom.0.bias']))
        k_g = self._softplus(self._linear(h1_g, self.weights['fc_geom.2.weight'], self.weights['fc_geom.2.bias']))
        
        # Physics Formula
        v_depth_base = np.maximum(x[:, 0], 1e-6)
        v_geom_base = np.maximum(x[:, 1], 1e-6)
        
        pred_depth = k_d.flatten() * (v_depth_base ** self.weights['exp_depth'])
        pred_geom = k_g.flatten() * (v_geom_base ** self.weights['exp_geom'])
        
        # Gating
        inputs_gate = x[:, [5, 6]]
        h1_gate = self._relu(self._linear(inputs_gate, self.weights['gate_mlp.0.weight'], self.weights['gate_mlp.0.bias']))
        alpha_gate = self._sigmoid(self._linear(h1_gate, self.weights['gate_mlp.2.weight'], self.weights['gate_mlp.2.bias'])).flatten()
        
        final_pred_norm = (1 - alpha_gate) * pred_depth + alpha_gate * pred_geom
        volume_ml = final_pred_norm.item() * self.stats['y_max']
        
        return {
            "volume_ml": round(volume_ml, 2),
            "mode": "GEOM" if alpha_gate.item() > 0.5 else "DEPTH",
            "alpha": round(alpha_gate.item(), 3)
        }

if __name__ == "__main__":
    # Test Block
    model_file = 'model_weights.npz'
    if os.path.exists(model_file):
        print("[System] Loading pure NumPy Inference Engine...")
        predictor = NumpyVolumePredictor(model_file)
        extractor = FeatureExtractor()
        
        # 현재 폴더의 .npy 파일 하나로 테스트
        test_files = glob.glob('./glass/*.npy')
        if test_files:
            target_file = test_files[2]
            print(f"[Test] Processing: {target_file}")
            
            # [시간 측정 시작]
            start_total = time.perf_counter()

            # 1. 특징 추출 (Feature Extraction) 시간 측정
            t_extract_start = time.perf_counter()
            feats = extractor.process(target_file, roi=FIXED_ROI)
            t_extract_end = time.perf_counter()

            if feats:
                # 2. 추론 (Inference) 시간 측정
                t_infer_start = time.perf_counter()
                result = predictor.predict(feats)
                t_infer_end = time.perf_counter()
                
                # 전체 종료 시간
                end_total = time.perf_counter()

                # 시간 계산 (초 단위 -> 밀리초 단위 변환)
                time_extract = (t_extract_end - t_extract_start) * 1000
                time_infer = (t_infer_end - t_infer_start) * 1000
                time_total = (end_total - start_total) * 1000

                print(f"Prediction: {result}")
                print("-" * 50)
                print(f"[Performance Report]")
                print(f" > Total Time      : {time_total:.2f} ms")
                print(f" > Feature Extract : {time_extract:.2f} ms")
                print(f" > Model Inference : {time_infer:.2f} ms")
                print("-" * 50)

            else:
                print("Feature extraction failed.")
        else:
            print("No .npy files found for testing.")
    else:
        print(f"Error: {model_file} not found. Run train script first.")