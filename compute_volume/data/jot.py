import numpy as np
from scipy import stats, ndimage
from skimage import filters, measure

def estimate_volume_universal(filename):
    # 1. 데이터 로드
    try:
        data = np.load(filename)
    except:
        return 0, 0
        
    flat_data = data.flatten()
    
    # 2. 전처리 (Segmentation)
    # 배경 모드 추출
    hist, bins = np.histogram(flat_data, bins=100)
    bg_idx = np.argmax(hist)
    bg_val = (bins[bg_idx] + bins[bg_idx+1]) / 2
    
    # 물체 분리 (Thresholding)
    thresh = bg_val - 15
    mask = data < thresh
    
    # 노이즈 제거 및 구멍 채우기
    labels = measure.label(mask)
    if labels.max() == 0: return 0, 0
    largest_region = max(measure.regionprops(labels), key=lambda r: r.area)
    mask = labels == largest_region.label
    mask = ndimage.binary_fill_holes(mask)
    
    # 3. 특징 추출 (Feature Extraction)
    object_depths = data[mask]
    if len(object_depths) == 0: return 0, 0
    
    # Rim Depth (거리)
    z_rim = np.percentile(object_depths, 5)
    
    # Height & Pixels -> Shape Factor (형상)
    z_max = np.max(object_depths)
    height = z_max - z_rim
    pixels = len(object_depths)
    shape_factor = height / np.sqrt(pixels)
    
    # 4. 3차 부피 지표 (Cubic Index) 계산
    valid_mask = object_depths > z_rim
    z_valid = object_depths[valid_mask]
    
    # Index = Sum( (Z_valid^3 - Z_rim^3) / 3 )
    cubic_index = np.sum((np.maximum(z_valid, z_rim)**3 - z_rim**3) / 3.0)
    
    # 5. 통합 보정 모델 적용 (Universal Calibration)
    # 파라미터 적용
    p_a = 6.2519e-07
    p_b = 3.6637e-05
    p_c = -8.6922e-05
    p_d = 0.5926
    
    # 공식: K(Z, Shape) * Index^d
    k_factor = (p_a * z_rim) + (p_b * shape_factor) + p_c
    estimated_vol = k_factor * (cubic_index ** p_d)
    
    return estimated_vol, cubic_index

# --- 테스트 ---
files = [
    ('508_mug.npy', 508),
    ('413_dojagi.npy', 413),
    ('567_tumbler.npy', 567),
    ('380_bigpaper.npy', 380),
    ('170_smallcup.npy', 170)
]

print(f"{'File':<15} | {'Real':<5} | {'Est(ml)':<8} | {'Error(%)':<8}")
print("-" * 50)

for fname, real_vol in files:
    est, idx = estimate_volume_universal(fname)
    err = ((est - real_vol) / real_vol) * 100
    print(f"{fname.split('.')[0]:<15} | {real_vol:<5} | {est:<8.1f} | {err:<+8.1f}")