import numpy as np
from scipy import stats, ndimage
from skimage import filters, measure

def estimate_volume_precision(filename):
    # 1. 데이터 로드 및 전처리
    data = np.load(filename)
    flat_data = data.flatten()
    
    # 배경 및 객체 분리 (Segmentation)
    hist, bins = np.histogram(flat_data, bins=100)
    # 히스토그램에서 가장 빈도가 높은 곳이 배경일 확률이 큼
    bg_idx = np.argmax(hist)
    bg_val = (bins[bg_idx] + bins[bg_idx+1]) / 2
    thresh = bg_val - 15  # 노이즈 마진
    
    mask = data < thresh
    
    # 가장 큰 덩어리 추출
    labels = measure.label(mask)
    if labels.max() == 0: return 0, 0, 0
    largest_region = max(measure.regionprops(labels), key=lambda r: r.area)
    mask = labels == largest_region.label
    mask = ndimage.binary_fill_holes(mask) # 내부 구멍 메우기
    
    # 2. 특징 추출 (Feature Extraction)
    object_depths = data[mask]
    if len(object_depths) == 0: return 0, 0, 0
    
    z_min = np.min(object_depths)
    z_max = np.max(object_depths)
    
    # Rim Depth: 상위 5% 지점 (노이즈 제외한 최상단)
    z_rim = np.percentile(object_depths, 5)
    
    # Height: 바닥부터 림까지의 깊이 차이
    height = z_max - z_rim
    
    # Pixels: 컵이 차지하는 화면상 면적
    pixels = len(object_depths)
    
    # Shape Factor: 깊이/너비 비율 (가늘고 긴지, 넓고 얕은지)
    shape_factor = height / np.sqrt(pixels)
    
    # 3. 3차 부피 지표 (Cubic Index) 계산
    # 림보다 깊은(내부) 픽셀만 유효
    valid_mask = object_depths > z_rim
    z_valid = object_depths[valid_mask]
    
    # 3차 적분 (원근 보정된 기본 부피)
    # 림 위쪽(음수) 값은 물리적으로 불가능하므로 제외(np.maximum)
    cubic_index = np.sum((np.maximum(z_valid, z_rim)**3 - z_rim**3) / 3.0)
    
    # 4. 정밀 보정 모델 (Calibration)
    # 최적화된 파라미터 적용
    p_a = 4.0199e-11
    p_b = 1.5783e-09
    p_c = -8.3515e-09
    
    # K 계산
    k_factor = (p_a * z_rim) + (p_b * shape_factor) + p_c
    
    # 최종 부피 추정
    estimated_vol = cubic_index * k_factor
    
    return estimated_vol, z_rim, cubic_index

# --- 실행 및 검증 ---
files = [
    ('508_4000_mug.npy', 508),
    ('413_4000_dojagi.npy', 413),
    ('567_4000_tumbler.npy', 567),
    ('380_4000_paper.npy', 380)
]

print(f"{'File':<15} | {'Real':<5} | {'Est(ml)':<8} | {'Error(%)':<8} | {'Index':<10}")
print("-" * 65)

for fname, real_vol in files:
    est, rim, idx = estimate_volume_precision(fname)
    err = ((est - real_vol) / real_vol) * 100
    print(f"{fname.split('.')[0]:<15} | {real_vol:<5} | {est:<8.1f} | {err:<8.1f} | {idx:.1e}")