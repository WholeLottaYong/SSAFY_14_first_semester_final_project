import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

# Dobot Messages
from dobot_msgs.action import PointToPoint
# [수정 1] 서비스 이름 변경 (SuctionCup -> SuctionCupControl)
from dobot_msgs.srv import SuctionCupControl

# Vision Libraries
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading

# =========================================================
# 🎯 [User Data] 캘리브레이션 변환 행렬
# =========================================================
TRANSFORM_MATRIX = np.array([
    [-807.89449, 57.06342, 125.21213],
    [-62.02256, 944.67285, 267.03539]
])

# [설정] 로봇 높이 설정 (mm)
Z_SAFE = 50.0      # 이동 시 안전 높이
Z_PICK = -41.0     # 큐브 잡는 높이

# [설정] 색상별 놓을 위치 (Drop Location) - 아까 설정하신 값 유지
DROP_LOCATIONS = {
    'Red':    [175.0, 0.0, Z_PICK + 10, 0.0],
    'Blue':   [175.0, 30.0, Z_PICK + 10, 0.0],
    'Green':  [175.0, 80.0, Z_PICK + 10, 0.0],
    'Yellow': [175.0, 130.0, Z_PICK + 10, 0.0]
}

class ColorSorter(Node):
    def __init__(self):
        super().__init__('dobot_color_sorter')
        
        # ---------------------------------------------------
        # 1. Robot Control Setup (Action & Service)
        # ---------------------------------------------------
        self._action_client = ActionClient(self, PointToPoint, 'PTP_action')
        
        # [수정 2] 클라이언트 생성 시 SuctionCupControl 사용
        # 서비스 이름이 '/dobot_suction_cup_service'가 맞는지 확인 필요 (틀리면 ros2 service list로 확인)
        self._suction_client = self.create_client(SuctionCupControl, '/dobot_suction_cup_service')
        
        # ---------------------------------------------------
        # 2. Vision Setup
        # ---------------------------------------------------
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
        
        # {'Red': {'buffer': [], 'fixed': (x,y)}, ...}
        self.scan_data = {color: {'buffer': [], 'fixed': None} for color in self.color_ranges}
        self.is_collecting = False
        self.is_robot_busy = False 

        self.window_name = "Dobot Color Sorter"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.get_logger().info("✅ System Ready! Flow: Drag -> 'c'(Capture) -> 'p'(Pick)")

    # ----------------------------------------------------------------
    # [Robot Control Functions]
    # ----------------------------------------------------------------
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
        """펌프 제어"""
        if self._suction_client.service_is_ready():
            # [수정 3] Request 객체 생성 시 SuctionCupControl 사용
            req = SuctionCupControl.Request()
            req.enable_suction = enable
            self._suction_client.call_async(req)
            time.sleep(0.5)
        else:
            self.get_logger().warn("Suction Service Not Ready")

    def run_sorting_sequence(self):
        """Pick & Place Sequence"""
        self.is_robot_busy = True
        self.get_logger().info("🚀 Start Sorting Sequence...")

        targets = []
        for color, data in self.scan_data.items():
            if data['fixed'] is not None:
                targets.append((color, data['fixed']))
        
        if not targets:
            self.get_logger().warn("No cubes fixed! Press 'c' to capture first.")
            self.is_robot_busy = False
            return

        for color, (tx, ty) in targets:
            self.get_logger().info(f"👉 Picking {color} at ({tx:.1f}, {ty:.1f})")
            
            # 1. 접근
            self.send_move_goal(tx, ty, Z_SAFE, mode=1)
            # 2. 잡기
            self.set_suction(True)
            self.send_move_goal(tx, ty, Z_PICK, mode=1)
            # 3. 들기
            self.send_move_goal(tx, ty, Z_SAFE, mode=1)
            # 4. 이동
            dx, dy, dz, dr = DROP_LOCATIONS[color]
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            # 5. 놓기
            self.send_move_goal(dx, dy, dz, mode=1)
            self.set_suction(False)
            # 6. 복귀
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            
        self.send_move_goal(150.0, 0.0, Z_SAFE, mode=1)
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
        for color in self.scan_data:
            self.scan_data[color]['buffer'] = []
            self.scan_data[color]['fixed'] = None
        self.is_collecting = False

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
                    cv2.putText(vis_frame, f"FIXED [{color}]: ({fx:.0f}, {fy:.0f})", 
                                (rx, ry - 20 - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

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
                                    
                                    if self.scan_data[color_name]['fixed']:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                                        cv2.putText(vis_frame, f"{color_name}(FIX)", (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                        label = f"{color_name}: {rob_x:.0f},{rob_y:.0f}"
                                        
                                        if self.is_collecting:
                                            self.scan_data[color_name]['buffer'].append((rob_x, rob_y))
                                            cnt = len(self.scan_data[color_name]['buffer'])
                                            label += f" [{cnt}/60]"
                                            if cnt >= 60:
                                                avg_x = np.mean([p[0] for p in self.scan_data[color_name]['buffer']])
                                                avg_y = np.mean([p[1] for p in self.scan_data[color_name]['buffer']])
                                                self.scan_data[color_name]['fixed'] = (avg_x, avg_y)
                                                print(f"✅ FIXED {color_name}: {avg_x:.1f}, {avg_y:.1f}")

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