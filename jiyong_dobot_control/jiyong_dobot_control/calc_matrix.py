import numpy as np
import cv2

# [중요!] 위 1번 코드로 다시 측정한 값을 여기에 넣어야 합니다!
# (왜곡 보정이 들어갔으므로 좌표값이 미세하게 바뀌었을 것입니다)

# 1. 카메라 좌표 (왜곡 보정된 새 측정값 넣기)
pts_camera = np.float32([
    [0.0764, -0.1373],   # Red (예시값)
    [0.0753, -0.0561],   # Blue (예시값)
    [-0.0034, -0.1372],   # Green (예시값)
    [-0.0051, -0.0570],  # Yellow (예시값)
])

# 2. 로봇 좌표 (기존 측정값 유지)
pts_robot = np.float32([
    [43.7906, 124.6663],    # Red
    [51.4188, 202.5270],    # Blue
    [107.3960, 133.6508],   # Green
    [115.9569, 206.3470],   # Yellow
])

# 3. 변환 행렬 계산 [변경됨: Perspective Transform 사용]
# 점 4개를 사용하여 3x3 원근 변환 행렬을 구합니다.
matrix = cv2.getPerspectiveTransform(pts_camera, pts_robot)

print("\n" + "="*40)
print("🎯 [결과] 아래 행렬 코드를 복사해서 dobot_test3.py에 붙여넣으세요")
print("="*40)
print("TRANSFORM_MATRIX = np.array([")
print(f"    [{matrix[0][0]:.5f}, {matrix[0][1]:.5f}, {matrix[0][2]:.5f}],")
print(f"    [{matrix[1][0]:.5f}, {matrix[1][1]:.5f}, {matrix[1][2]:.5f}],")
print(f"    [{matrix[2][0]:.5f}, {matrix[2][1]:.5f}, {matrix[2][2]:.5f}]")
print("])")
print("="*40 + "\n")

# 4. 검증
print("[검증: 각 큐브 위치 오차]")
colors = ['Red', 'Blue', 'Green', 'Yellow']
for i, pt in enumerate(pts_camera):
    # Perspective 변환을 위한 차원 조작
    input_pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(input_pt, matrix)
    
    real = pts_robot[i]
    calc_x = result[0][0][0]
    calc_y = result[0][0][1]
    
    error = np.sqrt((real[0]-calc_x)**2 + (real[1]-calc_y)**2)
    print(f"{colors[i]}: 오차 {error:.2f} mm")