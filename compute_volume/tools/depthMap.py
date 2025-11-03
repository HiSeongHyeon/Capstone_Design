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
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        depth = self.depth_data

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        vmin=float(np.percentile(depth, 1.0))
        vmax=float(np.percentile(depth, 99.0))
        surf = ax.plot_surface(x,y,depth, cmap = 'turbo', vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        #실제 데이터가 범위 안에 존재하는지 확인할 필요가 있음
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Depth (mm)')
        
        fig.colorbar(surf, shrink=0.5, aspect=12, label='Depth (mm)')
        ax.set_title('Depth Surface')
        fig.savefig('depthmap.png', dpi=1000, bbox_inches='tight')
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
    example = depthMap('./example/1103/calibration_03.npy')
    example.make_depth_Map()
    example.make_histogram()