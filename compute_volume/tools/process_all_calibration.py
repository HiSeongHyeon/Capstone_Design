#!/usr/bin/env python3
"""
example 폴더에 있는 cali_00부터 cali_18까지의 파일들에 대해
smooth.py, calibration.py, depthMap.py를 순차적으로 실행하고
그 결과를 result 폴더에 저장하는 스크립트
"""
import os
import numpy as np
from smooth import Smooth
from calibration import Calibration
from depthMap import depthMap
import matplotlib.pyplot as plt

def process_calibration_files():
    # result 폴더 생성
    result_dir = './result'
    example_dir = './example'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")
    
    # cali_00부터 cali_18까지 처리
    for i in range(19):
        cali_name = f'cali_{i:02d}'
        print(f"\n{'='*50}")
        print(f"Processing {cali_name}...")
        print(f"{'='*50}")
        
        # 1단계: smooth.py 실행
        input_file = os.path.join(example_dir, f'{cali_name}.npy')
        smoothed_file = os.path.join(result_dir, f'smoothed_{cali_name}.npy')
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found. Skipping...")
            continue
        
        try:
            print(f"Step 1: Smoothing {input_file}...")
            smoother = Smooth(input_file)
            smoother.save_smoothed(smoothed_file)
            
            # 2단계: calibration.py 실행
            normalized_file = os.path.join(result_dir, f'normalized_{cali_name}.npy')
            print(f"Step 2: Normalizing {smoothed_file}...")
            calib = Calibration(smoothed_file)
            calib.save_normalized(normalized_file)
            
            # 3단계: depthMap.py 실행
            print(f"Step 3: Generating depth map for {normalized_file}...")
            depth_map = depthMap(normalized_file)
            
            # depth map 생성 (출력 경로를 result 폴더로 변경)
            x, y = np.meshgrid(np.arange(depth_map.width), np.arange(depth_map.height))
            depth = depth_map.depth_data
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            vmin = float(np.percentile(depth, 1.0))
            vmax = float(np.percentile(depth, 99.0))
            surf = ax.plot_surface(x, y, depth, cmap='turbo', vmin=vmin, vmax=vmax, 
                                  linewidth=0, antialiased=True)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            ax.set_zlabel('Depth (mm)')
            fig.colorbar(surf, shrink=0.5, aspect=12, label='Depth (mm)')
            ax.set_title(f'Depth Surface - {cali_name}')
            
            depthmap_path = os.path.join(result_dir, f'depthmap_{cali_name}.png')
            fig.savefig(depthmap_path, dpi=1000, bbox_inches='tight')
            plt.close(fig)
            print(f"Depth map saved to {depthmap_path}")
            
            # histogram 생성
            plt.figure(figsize=(8, 6))
            plt.hist(depth[depth > 0].ravel(), bins=255, range=(0, np.max(depth)), 
                    color='blue', alpha=0.7)
            plt.title(f'Depth Data Histogram - {cali_name}')
            plt.xlabel('Depth (mm)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            histogram_path = os.path.join(result_dir, f'depth_histogram_{cali_name}.png')
            plt.savefig(histogram_path, dpi=1000, bbox_inches='tight')
            plt.close()
            print(f"Histogram saved to {histogram_path}")
            
            print(f"Successfully processed {cali_name}!")
            
        except Exception as e:
            print(f"Error processing {cali_name}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Processing complete!")
    print(f"Results saved in: {result_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_calibration_files()

