import numpy as np
import cv2

# 1. 카메라 좌표 (방금 주신 로그 평균값)
pts_camera = np.float32([
    [0.0876, -0.0636],  # Red
    [-0.0312, -0.1182], # Blue
    [0.0870, -0.1022],  # Green
    [0.0085, -0.1017]   # Yellow
])

# 2. 로봇 좌표 (방금 주신 값)
pts_robot = np.float32([
    [51.17,  201.56],   # Red
    [144.5, 157.4],     # Blue
    [49.15,  165.1],    # Green
    [111.3, 170.3]      # Yellow
])

# 3. 변환 행렬 계산
matrix, _ = cv2.estimateAffine2D(pts_camera, pts_robot)

print("\n" + "="*40)
print("🎯 [결과] 아래 행렬 코드를 복사해서 사용하세요")
print("="*40)
print("TRANSFORM_MATRIX = np.array([")
print(f"    [{matrix[0][0]:.5f}, {matrix[0][1]:.5f}, {matrix[0][2]:.5f}],")
print(f"    [{matrix[1][0]:.5f}, {matrix[1][1]:.5f}, {matrix[1][2]:.5f}]")
print("])")
print("="*40 + "\n")

# 4. 검증 (오차가 10mm 이상이면 큐브 색깔이 서로 바뀐 것일 수 있음)
print("[검증: 각 큐브 위치 오차]")
colors = ['Red', 'Blue', 'Green', 'Yellow']
for i, pt in enumerate(pts_camera):
    input_pt = np.array([pt[0], pt[1], 1.0])
    result = np.dot(matrix, input_pt)
    real = pts_robot[i]
    error = np.sqrt((real[0]-result[0])**2 + (real[1]-result[1])**2)
    print(f"{colors[i]}: 오차 {error:.2f} mm")