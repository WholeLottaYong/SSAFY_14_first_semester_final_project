import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from dobot_msgs.action import PointToPoint
from dobot_msgs.srv import SuctionCupControl
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading

# =========================================================
# 🎯 [Step 1] 캘리브레이션 변환 행렬 (3x3 Matrix로 교체해야 함)
# =========================================================
# calc_matrix.py 실행 후 나온 3x3 행렬로 덮어씌우세요!
TRANSFORM_MATRIX = np.array([
    [-26.78282, 801.92427, 263.07654],
    [865.05996, -32.57398, -11.01282],
    [-0.18015, -1.41963, 1.00000]
])

OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0
Z_OFFSET_CONSTANT = 354.72 # 353.0 ~ 357.5 사이 조정
Z_SAFE = 90.0

DROP_LOCATIONS = {
    'Red':    [120.0, 175.0, 3.0, 0.0],
    'Blue':   [120.0, 175.0, 3.0, 0.0],
    'Green':  [53.0, 175.0, 3.0, 0.0],
    'Yellow': [53.0, 175.0, 3.0, 0.0],
}

class ColorSorter(Node):
    def __init__(self):
        super().__init__('dobot_color_sorter')
        self._action_client = ActionClient(self, PointToPoint, 'PTP_action')
        self._suction_client = self.create_client(SuctionCupControl, '/dobot_suction_cup_service')
        self.bridge = CvBridge()
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)

        self.latest_depth_img = None
        self.camera_intrinsics = None
        self.dist_coeffs = None # 왜곡 계수
        
        self.roi_start = None; self.roi_end = None
        self.selecting = False; self.roi_selected = False; self.roi_rect = None 
        
        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }
        self.scan_data = {color: {'buffer': [], 'fixed': None, 'depth_m': 0.0} for color in self.color_ranges}
        self.is_collecting = False; self.is_robot_busy = False 
        self.window_name = "Dobot Smart Sorter"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.get_logger().info("✅ Ready! Drag ROI -> 'c'(Capture) -> 'p'(Pick)")

    def get_smart_z(self, depth_m):
        if depth_m <= 0: return -20.0
        target_z = Z_OFFSET_CONSTANT - (depth_m * 1000.0)
        if target_z < -45: target_z = -45
        return target_z

    def send_move_goal(self, x, y, z, r=0.0, mode=1):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
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

    def run_sorting_sequence(self):
        self.is_robot_busy = True
        targets = []
        for color, data in self.scan_data.items():
            if data['fixed'] is not None:
                targets.append((color, data['fixed'][0], data['fixed'][1], data['depth_m']))
        
        if not targets:
            self.is_robot_busy = False; return

        for color, tx, ty, tz_m in targets:
            final_x = tx + OFFSET_X; final_y = ty + OFFSET_Y
            smart_z = self.get_smart_z(tz_m)
            final_z = smart_z + OFFSET_Z
            self.get_logger().info(f"👉 Picking {color}: ({final_x:.1f}, {final_y:.1f})")
            
            self.send_move_goal(final_x, final_y, Z_SAFE, mode=1)
            self.set_suction(True)
            self.send_move_goal(final_x, final_y, final_z, mode=1)
            self.send_move_goal(final_x, final_y, Z_SAFE, mode=1)
            dx, dy, dz, dr = DROP_LOCATIONS[color]
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            self.send_move_goal(dx, dy, dz, mode=1)
            self.set_suction(False)
            self.send_move_goal(dx, dy, Z_SAFE, mode=1)
            
        self.send_move_goal(150.0, 0.0, 100, mode=1)
        self.is_robot_busy = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True; self.roi_start = (x, y); self.roi_end = (x, y); self.roi_selected = False; self.reset_data()
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

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.dist_coeffs = np.array(msg.d) # 왜곡 계수

    def depth_callback(self, msg):
        try: self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except: pass

    def color_callback(self, msg):
        try: frame = self.bridge.imgmsg_to_cv2(msg, "bgr8"); self.process_image(frame)
        except: pass

    def transform_cam_to_robot(self, cam_x, cam_y):
        # [변경] Perspective Transform 적용
        input_pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
        robot_pos = cv2.perspectiveTransform(input_pt, TRANSFORM_MATRIX)
        return robot_pos[0][0][0], robot_pos[0][0][1]

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
                    cv2.putText(vis_frame, f"FIXED [{color}]: ({fx:.0f}, {fy:.0f})", (rx, ry - 20 - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2); y_offset += 20

            frame_roi = frame[ry:ry+rh, rx:rx+rw]
            hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
            
            for color_name, ranges in self.color_ranges.items():
                try:
                    mask = np.zeros(hsv.shape[:2], dtype="uint8")
                    for (lower, upper) in ranges:
                        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
                    mask = cv2.erode(mask, None, iterations=1); mask = cv2.dilate(mask, None, iterations=2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if 500 < cv2.contourArea(contour) < 20000:
                            bx, by, bw, bh = cv2.boundingRect(contour)
                            cx = int(bx + bw/2) + rx; cy = int(by + bh/2) + ry
                            
                            if self.latest_depth_img is not None and self.camera_intrinsics:
                                d_y = min(max(cy, 0), 479); d_x = min(max(cx, 0), 639)
                                depth_mm = self.latest_depth_img[d_y, d_x]
                                
                                if depth_mm > 0:
                                    cam_z = depth_mm / 1000.0
                                    # [변경] 왜곡 보정 (Undistort) 적용
                                    fx = self.camera_intrinsics['fx']; fy = self.camera_intrinsics['fy']
                                    cx_cam = self.camera_intrinsics['cx']; cy_cam = self.camera_intrinsics['cy']

                                    if self.dist_coeffs is not None:
                                        src_pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                                        camera_matrix = np.array([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]], dtype=np.float32)
                                        dst_pt = cv2.undistortPoints(src_pt, camera_matrix, self.dist_coeffs, P=camera_matrix)
                                        cx_fixed = dst_pt[0][0][0]; cy_fixed = dst_pt[0][0][1]
                                    else:
                                        cx_fixed, cy_fixed = cx, cy

                                    cam_x = (cx_fixed - cx_cam) * cam_z / fx
                                    cam_y = (cy_fixed - cy_cam) * cam_z / fy
                                    rob_x, rob_y = self.transform_cam_to_robot(cam_x, cam_y)
                                    
                                    if self.scan_data[color_name]['fixed']:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                                    else:
                                        cv2.drawMarker(vis_frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                        label = f"{color_name}: {rob_x:.0f},{rob_y:.0f}"
                                        
                                        if self.is_collecting:
                                            self.scan_data[color_name]['buffer'].append((rob_x, rob_y, cam_z))
                                            cnt = len(self.scan_data[color_name]['buffer'])
                                            label += f" [{cnt}/60]"
                                            if cnt >= 60:
                                                avg_x = np.mean([p[0] for p in self.scan_data[color_name]['buffer']])
                                                avg_y = np.mean([p[1] for p in self.scan_data[color_name]['buffer']])
                                                avg_z = np.mean([p[2] for p in self.scan_data[color_name]['buffer']])
                                                self.scan_data[color_name]['fixed'] = (avg_x, avg_y)
                                                self.scan_data[color_name]['depth_m'] = avg_z
                                                print(f"✅ FIXED {color_name}: Pos=({avg_x:.1f}, {avg_y:.1f})")

                                        cv2.putText(vis_frame, label, (cx-40, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except: pass

        if self.is_robot_busy: cv2.putText(vis_frame, "Robot Working...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.is_collecting: cv2.putText(vis_frame, "Measuring...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else: cv2.putText(vis_frame, "'c': Capture | 'p': Pick", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): self.reset_data(); self.is_collecting = True
        elif key == ord('r'): self.reset_data(); self.roi_selected = False; self.roi_rect = None
        elif key == ord('p'): 
            if not self.is_robot_busy: threading.Thread(target=self.run_sorting_sequence).start()
        elif key == 27: rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ColorSorter()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()