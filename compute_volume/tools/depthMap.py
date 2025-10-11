import matplotlib.pyplot as plt
import numpy as np

#클래스 선언부
#npy 파일을 읽어 3D depth map을 그리는 클래스
#예시 사용법은 main 함수를 참고할것

class depthMap:
    def __init__(self, file_path):
        self.depth_data = np.load(file_path)
        if self.depth_data.ndim != 2:
            raise ValueError(f"2차원 데이터 형식을 요구합니다, 이 데이터는 {self.depth_data.ndim}입니다.")
        self.depth_data = self.depth_data.astype(np.float32)
        self.height, self.width = self.depth_data.shape
    
    def make_depth_Map(self):
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        #샘플링 비율 설정은 기본값 1로 설정했음, 실제 데이터와 비교해봐야함
        depth = self.depth_data

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x,y,depth, cmap = 'turbo', vmin=0, vmax=4000, linewidth=0, antialiased=True)
        #실제 데이터가 범위 안에 존재하는지 확인할 필요가 있음
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Depth (mm)')
        fig.colorbar(surf, shrink=0.5, aspect=12, label='Depth (mm)')
        ax.set_title('Depth Surface')
        fig.savefig('depthmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    example = depthMap('example.npy')
    example.make_depth_Map()