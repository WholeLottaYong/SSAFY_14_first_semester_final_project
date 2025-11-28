# 다음 주 예정 작업
# 1. cv2.undistortPoints 활용 오차 보정 : 카메라 렌즈의 굴곡(왜곡) 때문에 화면 가장자리에서 좌표가 휘어지는 현상을 수학적으로 펴주는 함수
# 2. 다시 카메라 세팅해서 변환 함수 초기화(카메라 좌표(cube_detector), 로봇 좌표(ros2 topic echo /dobot_TCP) 재측정)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

# Dobot Messages
from dobot_msgs.action import PointToPoint
from dobot_msgs.srv import SuctionCupControl

# Vision Libraries
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading

# =========================================================
# 🎯 [Step 1] 캘리브레이션 변환 행렬 (업데이트됨)
# =========================================================
TRANSFORM_MATRIX = np.array([
    [-792.96669, 14.34424, 108.77482],
    [-58.11570, 977.23931, 261.97363]
])

# [추가] 미세 보정 값 (mm 단위)
# 예: 로봇이 실제 큐브보다 오른쪽(+Y)으로 3mm 가면 -> OFFSET_Y = -3.0
# 예: 로봇이 실제 큐브보다 앞쪽(+X)으로 2mm 가면 -> OFFSET_X = -2.0
# 예: 로봇이 실제 큐브보다 윗쪽(+X)으로 2mm 가면 -> OFFSET_Z = -2.0
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0

# =========================================================
# 📏 [Step 2] Z축 자동 계산 상수 (데이터 기반 산출)
# =========================================================
Z_OFFSET_CONSTANT = 348.8

# [설정] 로봇 이동 안전 높이 (mm)
Z_SAFE = 50.0      

# [설정] 색상별 놓을 위치 (Drop Location) - x, y, z, r
DROP_LOCATIONS = {
    'Red':    [175.0, 0.0, -31, 0.0],
    'Blue':   [175.0, 45.0, -31, 0.0],
    'Green':  [175.0, 90.0, -31, 0.0],
    'Yellow': [175.0, 135.0, -31, 0.0]
}

class ColorSorter(Node):
    def __init__(self):
        super().__init__('dobot_color_sorter')
        
        # 1. Robot Control Setup
        self._action_client = ActionClient(self, PointToPoint, 'PTP_action')
        self._suction_client = self.create_client(SuctionCupControl, '/dobot_suction_cup_service')
        
        # 2. Vision Setup
        self.bridge = CvBridge()
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)

        self.latest_depth_img = None
        self.camera_intrinsics = None
        
        # ROI 및 데이터 관리
        self.roi_start = None; self.roi_end = None
        self.selecting = False; self.roi_selected = False; self.roi_rect = None 
        
        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }
        
        # 데이터 저장소: 좌표뿐만 아니라 Depth 정보도 저장하도록 확장
        # 'depth_m': 저장된 평균 깊이(meter)
        self.scan_data = {color: {'buffer': [], 'fixed': None, 'depth_m': 0.0} for color in self.color_ranges}
        self.is_collecting = False
        self.is_robot_busy = False 

        self.window_name = "Dobot Smart Sorter"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.get_logger().info("✅ Ready! Drag ROI -> 'c'(Capture) -> 'p'(Pick)")

    # ----------------------------------------------------------------
    # [Robot Control Functions]
    # ----------------------------------------------------------------
    def get_smart_z(self, depth_m):
        """ 카메라 깊이(m)를 로봇 높이(mm)로 변환 """
        if depth_m <= 0: return -20.0 # 에러 방지용 안전 높이
        
        # 우리가 구한 공식 적용
        target_z = Z_OFFSET_CONSTANT - (depth_m * 1000.0)
        
        # [안전장치] 바닥을 너무 세게 찍지 않도록 하한선 설정 (-48mm 제한)
        if target_z < -48.0: target_z = -48.0
        
        return target_z

    def send_move_goal(self, x, y, z, r=0.0, mode=1):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Action Server not available!")
            return

        goal_msg = PointToPoint.Goal()
        goal_msg.target_pose = [float(x), float(y), float(z), float(r)]
        goal_msg.motion_type = mode 

        self._action_client.send_goal_async(goal_msg)
        time.sleep(2.5) 

    def set_suction(self, enable: bool):
        if self._suction_client.service_is_ready():
            req = SuctionCupControl.Request()
            req.enable_suction = enable
            self._suction_client.call_async(req)
            time.sleep(0.5)
        else:
            self.get_logger().warn("Suction Service Not Ready")

    def run_sorting_sequence(self):
        self.is_robot_busy = True
        self.get_logger().info("🚀 Start Sorting Sequence...")

        targets = []
        for color, data in self.scan_data.items():
            if data['fixed'] is not None:
                # (Color, X, Y, Depth) 정보를 모두 가져옴
                targets.append((color, data['fixed'][0], data['fixed'][1], data['depth_m']))
        
        if not targets:
            self.get_logger().warn("No targets fixed. Press 'c' first.")
            self.is_robot_busy = False
            return

        for color, tx, ty, tz_m in targets:

            final_x = tx + OFFSET_X
            final_y = ty + OFFSET_Y
            # ★ 핵심: Z값 자동 계산
            smart_z = self.get_smart_z(tz_m)
            final_z = smart_z + OFFSET_Z
            
            self.get_logger().info(f"👉 Picking {color}: ({final_x:.1f}, {final_y:.1f}), Depth={tz_m:.3f}m -> Z={final_z:.1f}mm")
            
            # 1. 접근
            self.send_move_goal(final_x, final_y, Z_SAFE, mode=1)
            # 2. 잡기 (계산된 높이로 이동)
            self.set_suction(True)
            self.send_move_goal(final_x, final_y, final_z, mode=1)
            # 3. 들기
            self.send_move_goal(final_x, final_y, Z_SAFE, mode=1)
            # 4. 이동
            dx, dy, dz, dr = DROP_LOCATIONS[color]
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            # 5. 놓기
            self.send_move_goal(dx, dy, dz, mode=1)
            self.set_suction(False)
            # 6. 복귀
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            
        self.send_move_goal(150.0, 0.0, 100, mode=1) # 호밍 원점
        self.get_logger().info("✨ All Jobs Finished!")
        self.is_robot_busy = False

    # ----------------------------------------------------------------
    # [Vision Functions]
    # ----------------------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True; self.roi_start = (x, y); self.roi_end = (x, y)
            self.roi_selected = False; self.reset_data()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting: self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False; self.roi_end = (x, y); self.roi_selected = True
            x1 = min(self.roi_start[0], self.roi_end[0]); y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0]); y2 = max(self.roi_start[1], self.roi_end[1])
            self.roi_rect = (x1, y1, x2-x1, y2-y1)

    def reset_data(self):
        self.scan_data = {color: {'buffer': [], 'fixed': None, 'depth_m': 0.0} for color in self.color_ranges}
        self.is_collecting = False
        self.get_logger().info("🧹 Data Cleared!")

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def depth_callback(self, msg):
        try: self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except: pass

    def color_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(frame)
        except: pass

    def transform_cam_to_robot(self, cam_x, cam_y):
        input_pt = np.array([cam_x, cam_y, 1.0])
        robot_pos = np.dot(TRANSFORM_MATRIX, input_pt)
        return robot_pos[0], robot_pos[1]

    def process_image(self, frame):
        vis_frame = frame.copy()

        if self.selecting and self.roi_start and self.roi_end:
            cv2.rectangle(vis_frame, self.roi_start, self.roi_end, (0, 0, 255), 1)

        if self.roi_selected and self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            cv2.rectangle(vis_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 2)
            
            y_offset = 0
            for color, data in self.scan_data.items():
                if data['fixed'] is not None:
                    fx, fy = data['fixed']
                    # 화면에 계산될 Z높이도 같이 보여줍니다 (디버깅용)
                    pred_z = self.get_smart_z(data['depth_m'])
                    cv2.putText(vis_frame, f"FIXED [{color}]: ({fx:.0f}, {fy:.0f}) Z:{pred_z:.1f}", 
                                (rx, ry - 20 - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset += 20

            frame_roi = frame[ry:ry+rh, rx:rx+rw]
            hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
            
            for color_name, ranges in self.color_ranges.items():
                try:
                    mask = np.zeros(hsv.shape[:2], dtype="uint8")
                    for (lower, upper) in ranges:
                        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
                    
                    mask = cv2.erode(mask, None, iterations=1)
                    mask = cv2.dilate(mask, None, iterations=2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if 500 < cv2.contourArea(contour) < 20000:
                            bx, by, bw, bh = cv2.boundingRect(contour)
                            cx = int(bx + bw/2) + rx
                            cy = int(by + bh/2) + ry
                            
                            if self.latest_depth_img is not None and self.camera_intrinsics:
                                d_y = min(max(cy, 0), 479); d_x = min(max(cx, 0), 639)
                                depth_mm = self.latest_depth_img[d_y, d_x]
                                
                                if depth_mm > 0:
                                    cam_z = depth_mm / 1000.0
                                    cam_x = (cx - self.camera_intrinsics['cx']) * cam_z / self.camera_intrinsics['fx']
                                    cam_y = (cy - self.camera_intrinsics['cy']) * cam_z / self.camera_intrinsics['fy']
                                    rob_x, rob_y = self.transform_cam_to_robot(cam_x, cam_y)
                                    
                                    # 시각화 및 데이터 수집
                                    if self.scan_data[color_name]['fixed']:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                                        cv2.putText(vis_frame, f"{color_name}", (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                        label = f"{color_name}: {rob_x:.0f},{rob_y:.0f}"
                                        
                                        if self.is_collecting:
                                            # 좌표(x,y)와 깊이(z)를 모두 수집합니다
                                            self.scan_data[color_name]['buffer'].append((rob_x, rob_y, cam_z))
                                            cnt = len(self.scan_data[color_name]['buffer'])
                                            label += f" [{cnt}/60]"
                                            
                                            # 60개 모이면 평균 계산 (X, Y, Depth 모두)
                                            if cnt >= 60:
                                                avg_x = np.mean([p[0] for p in self.scan_data[color_name]['buffer']])
                                                avg_y = np.mean([p[1] for p in self.scan_data[color_name]['buffer']])
                                                avg_z = np.mean([p[2] for p in self.scan_data[color_name]['buffer']])
                                                
                                                self.scan_data[color_name]['fixed'] = (avg_x, avg_y)
                                                self.scan_data[color_name]['depth_m'] = avg_z # 깊이 저장
                                                print(f"✅ FIXED {color_name}: Pos=({avg_x:.1f}, {avg_y:.1f}), Depth={avg_z:.3f}m")

                                        cv2.putText(vis_frame, label, (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except: pass

        if self.is_robot_busy:
            cv2.putText(vis_frame, "Robot is Working...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not self.roi_selected:
            cv2.putText(vis_frame, "Drag mouse to select Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            if self.is_collecting:
                cv2.putText(vis_frame, "Measuring...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(vis_frame, "'c': Capture | 'p': Pick & Place", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.reset_data()
            self.is_collecting = True
        elif key == ord('r'):
            self.reset_data()
            self.roi_selected = False
            self.roi_rect = None
        elif key == ord('p'):
            if not self.is_robot_busy:
                t = threading.Thread(target=self.run_sorting_sequence)
                t.start()
        elif key == 27:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ColorSorter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()