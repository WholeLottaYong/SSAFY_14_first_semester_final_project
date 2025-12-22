import numpy as np
import cv2

# =========================================================
# ✏️ [입력] 새로 측정한 좌표를 여기에 넣으세요!
# =========================================================

# 1. 카메라 좌표 (RealSense 화면상의 x, y - 단위: 미터 or 픽셀)
pts_camera = np.float32([
    [-0.0155, -0.0369],   # Red
    [0.0417, -0.0423],   # Blue
    [0.0363, -0.0975],   # Green
    [-0.0563, -0.0622],   # Yellow
])

# 2. 로봇 좌표 (Dobot Studio 등에서 확인한 실제 팔 위치 - 단위: mm)
pts_robot = np.float32([
    [221.6697, -22.0051],   # Red
    [216.6557, 25.1184],   # Blue
    [162.4887, 20.8193],   # Green
    [195.4627, -52.5194],   # Yellow
])


# =========================================================
# 🧮 [계산] Perspective Transform (원근 변환)
# =========================================================
matrix = cv2.getPerspectiveTransform(pts_camera, pts_robot)

print("\n" + "="*50)
print("✅ [복사용 코드] 아래 내용을 dobot_vision.py에 덮어쓰세요.")
print("   (참고: 값이 3000~10000 단위로 나오는 건 정상입니다!)")
print("="*50)
print("TRANSFORM_MATRIX = np.array([")
print(f"    [{matrix[0][0]:.5f}, {matrix[0][1]:.5f}, {matrix[0][2]:.5f}],")
print(f"    [{matrix[1][0]:.5f}, {matrix[1][1]:.5f}, {matrix[1][2]:.5f}],")
print(f"    [{matrix[2][0]:.5f}, {matrix[2][1]:.5f}, {matrix[2][2]:.5f}]")
print("])")
print("="*50 + "\n")

# 검증 (당연히 오차 0 나와야 함)
print("[검증] 입력한 좌표들에 대한 변환 오차:")
colors = ['Red', 'Blue', 'Green', 'Yellow']
for i, pt in enumerate(pts_camera):
    input_pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(input_pt, matrix)
    real = pts_robot[i]
    error = np.sqrt((real[0]-result[0][0][0])**2 + (real[1]-result[0][0][1])**2)
    print(f"- {colors[i]}: 오차 {error:.4f} mm")