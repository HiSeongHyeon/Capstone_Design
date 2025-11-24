"""
데이터로부터 부피를 측정하는 통합 스크립트 (RBF_mesh 버전)

이 스크립트는 findMask와 RBF_mesh의 computeVolume을 사용하여
depth 데이터로부터 부피를 측정합니다.
"""

import numpy as np
import os
import sys
import glob
import re
import matplotlib.pyplot as plt

# tools 패키지에서 모듈 import
from tools import FindMask
from tools.RBF_mesh import computeVolume

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def measure_volume_from_data(
    depth_file_path,
    cup_diameter_mm=87,
    shrink_pixels=3,
    savedata=True,
    verbose=True
):
    """
    depth 데이터로부터 부피를 측정하는 통합 함수
    
    Parameters:
    -----------
    depth_file_path : str
        원본 depth 데이터 파일 경로 (.npy)
    cup_diameter_mm : float, optional
        컵의 실제 직경 (mm), 기본값 87mm
        RBF_mesh는 컵 직경을 기반으로 스케일 팩터를 계산합니다.
    shrink_pixels : int, optional
        마스크 축소 픽셀 수, 기본값 3
    savedata : bool, optional
        중간 결과물 저장 여부, 기본값 True
    verbose : bool, optional
        상세 출력 여부, 기본값 True
    
    Returns:
    --------
    dict
        부피 측정 결과를 담은 딕셔너리
        - 'riemann_volume': 구분구적법으로 계산한 부피 (mm³)
        - 'rbf_volume': RBF 보간으로 계산한 부피 (mm³)
        - 'mask': 생성된 마스크
        - 'masked_data': 마스킹된 depth 데이터
    """
    
    if verbose:
        print("=" * 80)
        print("부피 측정 통합 프로세스 시작 (RBF_mesh 버전)")
        print("=" * 80)
        print(f"입력 파일: {depth_file_path}")
        print(f"컵 직경: {cup_diameter_mm} mm")
        print(f"마스크 축소: {shrink_pixels} 픽셀")
        print("=" * 80)
    
    # 1단계: FindMask를 사용하여 마스크 생성 및 데이터 마스킹
    if verbose:
        print("\n[1단계] 마스크 생성 중...")
    try:
        find_mask_obj = FindMask(depth_file_path, savedata=savedata)
        mask, masked_data, baseline_depth = find_mask_obj.findMask(shrink_pixels=shrink_pixels)
        
        if mask is None or masked_data is None:
            raise ValueError("마스크 생성 실패")
        
        if verbose:
            print(f"마스크 생성 완료: {mask.shape}")
            print(f"마스킹된 데이터 범위: {np.min(masked_data[masked_data > 0]):.2f} ~ {np.max(masked_data):.2f} mm")
            print(f"기준점 보정값 (baseline_depth): {baseline_depth:.2f} mm")
        
    except Exception as e:
        print(f"마스크 생성 중 오류 발생: {e}")
        raise
    
    # 2단계: 마스킹된 데이터를 임시 파일로 저장 (computeVolume이 파일 경로를 요구)
    if verbose:
        print("\n[2단계] 마스킹된 데이터 준비 중...")
    temp_masked_file = 'masked_depth.npy'
    np.save(temp_masked_file, masked_data)
    if verbose:
        print(f"임시 파일 저장: {temp_masked_file}")
    
    # 2-1단계: 구분구적법용 데이터 준비 (baseline_depth를 뺀 데이터)
    masked_data_for_riemann = masked_data.copy()
    masked_data_for_riemann[masked_data_for_riemann > 0] -= baseline_depth
    masked_data_for_riemann[masked_data_for_riemann < 0] = 0  # 음수 방지
    temp_masked_file_riemann = 'masked_depth_riemann.npy'
    np.save(temp_masked_file_riemann, masked_data_for_riemann)
    if verbose:
        print(f"구분구적법용 데이터 저장 (baseline_depth 제거): {temp_masked_file_riemann}")
    
    # 3단계: computeVolume을 사용하여 부피 계산
    if verbose:
        print("\n[3단계] 부피 계산 중...")
    try:
        # 구분구적법용 computeVolume 객체 생성 (baseline_depth를 뺀 데이터 사용)
        if verbose:
            print("\n[3-1] 구분구적법 (Riemann Sum) 계산 중...")
        volume_calculator_riemann = computeVolume(temp_masked_file_riemann)
        riemann_volume, riemann_points = volume_calculator_riemann.estimate_cup_volume_improved(
            cup_diameter_mm=cup_diameter_mm
        )
        
        # RBF 보간용 computeVolume 객체 생성 (원본 데이터 사용)
        volume_calculator = computeVolume(temp_masked_file)
        
        # RBF 보간으로 부피 계산
        if verbose:
            print("\n[3-2] RBF 보간 (RBF Interpolation) 계산 중...")
        rbf_volume, rbf_mesh = volume_calculator.volume_rbf_interpolation(
            cup_diameter_mm=cup_diameter_mm
        )
        
        # 결과 출력
        if verbose:
            print("\n" + "=" * 80)
            print("부피 측정 결과")
            print("=" * 80)
            print(f"구분구적법 (Riemann Sum): {riemann_volume/1000:.2f} mL ({riemann_volume:.2f} mm³)")
            print(f"RBF 보간 (RBF Interpolation): {rbf_volume/1000:.2f} mL ({rbf_volume:.2f} mm³)")
            print("=" * 80)
        
        # 결과 딕셔너리 생성
        results = {
            'riemann_volume': riemann_volume,
            'rbf_volume': rbf_volume,
            'riemann_volume_ml': riemann_volume / 1000,
            'rbf_volume_ml': rbf_volume / 1000,
            'mask': mask,
            'masked_data': masked_data,
            'riemann_points': riemann_points,
            'rbf_mesh': rbf_mesh
        }
        
        return results
        
    except Exception as e:
        print(f"부피 계산 중 오류 발생: {e}")
        raise
    finally:
        # 임시 파일 정리 (선택사항)
        if not savedata:
            if os.path.exists(temp_masked_file):
                os.remove(temp_masked_file)
            if os.path.exists(temp_masked_file_riemann):
                os.remove(temp_masked_file_riemann)
            if verbose:
                print(f"\n임시 파일 삭제: {temp_masked_file}, {temp_masked_file_riemann}")


def extract_volume_from_filename(filename):
    """
    파일명에서 실제 부피를 추출합니다.
    형식: {부피}_{컵이름}.npy (예: 413_dojagi.npy -> 413)
    
    Parameters:
    -----------
    filename : str
        파일명 또는 파일 경로
    
    Returns:
    --------
    float or None
        추출된 부피 값, 실패 시 None
    """
    # 파일명만 추출 (경로 제거)
    basename = os.path.basename(filename)
    # 확장자 제거
    name_without_ext = os.path.splitext(basename)[0]
    # 첫 번째 숫자 부분 추출
    match = re.match(r'^(\d+(?:\.\d+)?)', name_without_ext)
    if match:
        return float(match.group(1))
    return None


def process_all_cups_in_data_folder(
    data_folder='data',
    cup_diameter_mm=87,
    shrink_pixels=3,
    savedata=False
):
    """
    data 폴더의 모든 .npy 파일을 처리하고 결과를 비교합니다.
    
    Parameters:
    -----------
    data_folder : str
        데이터 폴더 경로, 기본값 'data'
    cup_diameter_mm : float, optional
        컵의 실제 직경 (mm), 기본값 87mm
    shrink_pixels : int, optional
        마스크 축소 픽셀 수, 기본값 3
    savedata : bool, optional
        중간 결과물 저장 여부, 기본값 False
    
    Returns:
    --------
    dict
        모든 컵의 측정 결과를 담은 딕셔너리
    """
    # data 폴더의 모든 .npy 파일 찾기
    data_path = os.path.join(os.path.dirname(__file__), data_folder)
    if not os.path.exists(data_path):
        data_path = data_folder  # 절대 경로일 수도 있음
    
    npy_files = glob.glob(os.path.join(data_path, '*.npy'))
    
    if not npy_files:
        print(f"오류: {data_path} 폴더에서 .npy 파일을 찾을 수 없습니다.")
        return None
    
    print("=" * 80)
    print(f"배치 처리 시작: {len(npy_files)}개 파일 발견")
    print(f"컵 직경 설정: {cup_diameter_mm} mm")
    print("=" * 80)
    
    results = {}
    
    for idx, file_path in enumerate(npy_files, 1):
        filename = os.path.basename(file_path)
        actual_volume = extract_volume_from_filename(filename)
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(npy_files)}] 처리 중: {filename}")
        if actual_volume:
            print(f"실제 부피: {actual_volume} mL")
        print(f"{'='*80}")
        
        try:
            # 각 파일에 대해 부피 측정
            result = measure_volume_from_data(
                depth_file_path=file_path,
                cup_diameter_mm=cup_diameter_mm,
                shrink_pixels=shrink_pixels,
                savedata=savedata,
                verbose=False  # 배치 처리 시 상세 출력 비활성화
            )
            
            # 결과에 실제 부피와 파일명 추가
            result['filename'] = filename
            result['actual_volume_ml'] = actual_volume
            result['cup_name'] = os.path.splitext(filename)[0].split('_', 1)[1] if '_' in filename else filename
            
            results[filename] = result
            
            # 간단한 결과 출력
            if actual_volume:
                riemann_error = abs(result['riemann_volume_ml'] - actual_volume) / actual_volume * 100
                rbf_error = abs(result['rbf_volume_ml'] - actual_volume) / actual_volume * 100
                print(f"\n결과 요약:")
                print(f"  실제 부피: {actual_volume} mL")
                print(f"  구분구적법: {result['riemann_volume_ml']:.2f} mL (오차: {riemann_error:.1f}%)")
                print(f"  RBF 보간: {result['rbf_volume_ml']:.2f} mL (오차: {rbf_error:.1f}%)")
            
        except Exception as e:
            print(f"\n✗ {filename} 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results[filename] = {'error': str(e), 'filename': filename}
    
    return results


def compare_results(all_results):
    """
    모든 컵의 측정 결과를 비교하고 시각화합니다.
    
    Parameters:
    -----------
    all_results : dict
        process_all_cups_in_data_folder()의 반환값
    """
    if not all_results:
        print("비교할 결과가 없습니다.")
        return
    
    # 유효한 결과만 필터링
    valid_results = {
        k: v for k, v in all_results.items() 
        if 'error' not in v and 'riemann_volume_ml' in v
    }
    
    if not valid_results:
        print("유효한 결과가 없습니다.")
        return
    
    # 데이터 준비
    filenames = []
    cup_names = []
    actual_volumes = []
    riemann_volumes = []
    rbf_volumes = []
    riemann_errors = []
    rbf_errors = []
    
    for filename, result in valid_results.items():
        filenames.append(filename)
        cup_names.append(result.get('cup_name', filename))
        actual = result.get('actual_volume_ml', None)
        actual_volumes.append(actual)
        riemann_volumes.append(result['riemann_volume_ml'])
        rbf_volumes.append(result['rbf_volume_ml'])
        
        if actual:
            riemann_errors.append(abs(result['riemann_volume_ml'] - actual) / actual * 100)
            rbf_errors.append(abs(result['rbf_volume_ml'] - actual) / actual * 100)
        else:
            riemann_errors.append(None)
            rbf_errors.append(None)
    
    # 결과 테이블 출력
    print("\n" + "=" * 100)
    print("전체 결과 비교 (RBF_mesh 버전)")
    print("=" * 100)
    print(f"{'컵 이름':<20} {'실제 부피':<12} {'구분구적법':<15} {'오차(%)':<10} {'RBF 보간':<15} {'오차(%)':<10}")
    print("-" * 100)
    
    for i, (cup_name, actual, riemann, rbf, r_err, b_err) in enumerate(
        zip(cup_names, actual_volumes, riemann_volumes, rbf_volumes, riemann_errors, rbf_errors)
    ):
        actual_str = f"{actual:.1f}" if actual else "N/A"
        r_err_str = f"{r_err:.1f}" if r_err is not None else "N/A"
        b_err_str = f"{b_err:.1f}" if b_err is not None else "N/A"
        print(f"{cup_name:<20} {actual_str:<12} {riemann:<15.2f} {r_err_str:<10} {rbf:<15.2f} {b_err_str:<10}")
    
    print("=" * 100)
    
    # 시각화
    n_cups = len(valid_results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 부피 비교 막대 그래프
    ax1 = axes[0, 0]
    x_pos = np.arange(n_cups)
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, riemann_volumes, width, label='구분구적법', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, rbf_volumes, width, label='RBF 보간', alpha=0.8)
    
    if any(actual_volumes):
        ax1.scatter(x_pos, [v for v in actual_volumes if v], 
                   color='red', marker='*', s=200, label='실제 부피', zorder=5)
    
    ax1.set_xlabel('컵')
    ax1.set_ylabel('부피 (mL)')
    ax1.set_title('측정 방법별 부피 비교 (RBF_mesh)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cup_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for i, (r, b) in enumerate(zip(riemann_volumes, rbf_volumes)):
        ax1.text(i - width/2, r, f'{r:.1f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, b, f'{b:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 오차율 비교
    ax2 = axes[0, 1]
    if any(e is not None for e in riemann_errors):
        bars3 = ax2.bar(x_pos - width/2, [e if e is not None else 0 for e in riemann_errors], 
                       width, label='구분구적법', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, [e if e is not None else 0 for e in rbf_errors], 
                       width, label='RBF 보간', alpha=0.8)
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% 오차')
        ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% 오차')
        ax2.set_ylabel('오차율 (%)')
        ax2.set_title('측정 방법별 오차율 (RBF_mesh)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cup_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 산점도: 실제 vs 측정 (구분구적법)
    ax3 = axes[1, 0]
    if any(actual_volumes):
        actual_valid = [v for v in actual_volumes if v]
        riemann_valid = [riemann_volumes[i] for i, v in enumerate(actual_volumes) if v]
        ax3.scatter(actual_valid, riemann_valid, s=100, alpha=0.7, label='구분구적법')
        # 1:1 선
        max_val = max(max(actual_valid), max(riemann_valid))
        min_val = min(min(actual_valid), min(riemann_valid))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 선')
        ax3.set_xlabel('실제 부피 (mL)')
        ax3.set_ylabel('측정 부피 (mL)')
        ax3.set_title('구분구적법: 실제 vs 측정 (RBF_mesh)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 산점도: 실제 vs 측정 (RBF)
    ax4 = axes[1, 1]
    if any(actual_volumes):
        rbf_valid = [rbf_volumes[i] for i, v in enumerate(actual_volumes) if v]
        ax4.scatter(actual_valid, rbf_valid, s=100, alpha=0.7, label='RBF 보간', color='orange')
        # 1:1 선
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 선')
        ax4.set_xlabel('실제 부피 (mL)')
        ax4.set_ylabel('측정 부피 (mL)')
        ax4.set_title('RBF 보간: 실제 vs 측정 (RBF_mesh)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'volume_comparison_all_cups_rbf_mesh.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n비교 그래프 저장: {output_path}")
    plt.show()


def main():
    """
    메인 함수: 명령줄 인자 또는 기본값으로 실행
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='depth 데이터로부터 부피를 측정합니다. (RBF_mesh 버전)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 단일 파일 처리
  python intergrate_volume_rbf_mesh.py ./data/413_dojagi.npy
  
  # data 폴더의 모든 파일 배치 처리
  python intergrate_volume_rbf_mesh.py --batch
  
  # 배치 처리 (옵션 지정)
  python intergrate_volume_rbf_mesh.py --batch --cup-diameter 70 --shrink-pixels 5
        """
    )
    
    parser.add_argument(
        'depth_file',
        type=str,
        nargs='?',
        default=None,
        help='원본 depth 데이터 파일 경로 (.npy)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='data 폴더의 모든 .npy 파일을 배치 처리'
    )
    
    parser.add_argument(
        '--data-folder',
        type=str,
        default='data',
        help='배치 처리할 데이터 폴더 경로, 기본값: data'
    )
    
    parser.add_argument(
        '--cup-diameter',
        type=float,
        default=87,
        help='컵의 실제 직경 (mm), 기본값: 87'
    )
    
    parser.add_argument(
        '--shrink-pixels',
        type=int,
        default=3,
        help='마스크 축소 픽셀 수, 기본값: 3'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='중간 결과물 저장하지 않음'
    )
    
    args = parser.parse_args()
    
    # 배치 처리 모드
    if args.batch:
        try:
            all_results = process_all_cups_in_data_folder(
                data_folder=args.data_folder,
                cup_diameter_mm=args.cup_diameter,
                shrink_pixels=args.shrink_pixels,
                savedata=not args.no_save
            )
            
            if all_results:
                compare_results(all_results)
                print("\n✓ 배치 처리 완료!")
            return all_results
            
        except Exception as e:
            print(f"\n✗ 배치 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 단일 파일 처리 모드
    if not args.depth_file:
        parser.print_help()
        print("\n오류: 단일 파일 경로를 지정하거나 --batch 옵션을 사용하세요.")
        sys.exit(1)
    
    if not os.path.exists(args.depth_file):
        print(f"오류: 파일을 찾을 수 없습니다: {args.depth_file}")
        sys.exit(1)
    
    try:
        results = measure_volume_from_data(
            depth_file_path=args.depth_file,
            cup_diameter_mm=args.cup_diameter,
            shrink_pixels=args.shrink_pixels,
            savedata=not args.no_save
        )
        
        print("\n✓ 부피 측정 완료!")
        return results
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

