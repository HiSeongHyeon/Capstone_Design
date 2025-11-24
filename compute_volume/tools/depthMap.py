#클래스 선언부
#npy 파일을 읽어 3D depth map을 그리는 클래스
#예시 사용법은 main 함수를 참고할것
import matplotlib.pyplot as plt
import numpy as np

class depthMap:
    def __init__(self, file_path):
        self.depth_data = np.load(file_path)
        if self.depth_data.ndim != 2:
            raise ValueError(f"2차원 데이터 형식을 요구합니다, 이 데이터는 {self.depth_data.ndim}입니다.")
        self.depth_data = self.depth_data.astype(np.float32)
        self.height, self.width = self.depth_data.shape
    
    def make_depth_Map(self):
        # 원본 데이터 보호를 위해 복사
        depth = self.depth_data.copy()
        
        # [핵심 해결책] 값이 0인 부분(배경)을 NaN으로 변경
        # 이렇게 하면 Matplotlib이 해당 영역을 그리지 않고 무시합니다.
        depth[depth == 0] = np.nan
        
        # X, Y 그리드 생성
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # [수정] NaN이 포함된 데이터에서 min/max를 구하기 위해 nanpercentile 사용
        # 기존 np.percentile은 NaN이 있으면 결과를 NaN으로 반환할 수 있음
        try:
            vmin = float(np.nanpercentile(depth, 1.0))
            vmax = float(np.nanpercentile(depth, 99.0))
        except:
            # 데이터가 모두 NaN이거나 비어있을 경우 예외처리
            vmin, vmax = 0, 1
            
        # 서피스 플롯
        surf = ax.plot_surface(x, y, depth, cmap='turbo_r', vmin=vmin, vmax=vmax, 
                               linewidth=0, antialiased=False, rstride=2, cstride=2)
        
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Depth (mm)')
        
        # Z축 반전 (깊은 곳이 바닥으로)
        ax.invert_zaxis()
        
        # 1:1:1 스케일 (Visual Cube)
        ax.set_box_aspect((1, 1, 1))
        
        # 시점 조정
        ax.view_init(elev=60, azim=-45)

        # 컬러바 생성 (NaN 무시하고 유효 데이터 범위만 표시됨)
        fig.colorbar(surf, shrink=0.5, aspect=12, label='Depth (mm)')
        ax.set_title('3D Depth Surface (Background Removed)')
        
        fig.savefig('depthmap_final.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def make_histogram(self):
        plt.figure(figsize=(8,6))
        plt.hist(self.depth_data[self.depth_data > 0].ravel(), bins=255, range=(0, np.max(self.depth_data)), color='blue', alpha=0.7)
        plt.title('Depth Data Histogram')
        plt.xlabel('Depth (mm)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('depth_histogram.png', dpi=1000, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    example = depthMap('masked_depth.npy')
    # example = depthMap('C:/Users/rngyq/Documents/Capstone_Design/compute_volume/data/413_dojagi.npy')
    example.make_depth_Map()
    example.make_histogram()