import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# =========================================================
# 🎯 [User Data] 변환 행렬 (User Provided)
# =========================================================
TRANSFORM_MATRIX = np.array([
    [-967.27151, -106.58200, 181.96182],
    [-95.73259, 952.66342, 57.90632]
])

class CubeScanner(Node):
    def __init__(self):
        super().__init__('cube_scanner_node')
        
        self.bridge = CvBridge()
        
        # ROS Topics
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)

        self.latest_depth_img = None
        self.camera_intrinsics = None
        
        # --- ROI (Mouse Drag) 관련 변수 ---
        self.roi_start = None
        self.roi_end = None
        self.selecting = False
        self.roi_selected = False
        self.roi_rect = None  # (x, y, w, h)

        # 윈도우 설정
        self.window_name = "Coordinate Scanner"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 색상 범위
        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }
        
        self.get_logger().info("Scanner Started! Drag mouse to set ROI.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.roi_start = (x, y)
            self.roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.roi_end = (x, y)
            self.roi_selected = True
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            self.roi_rect = (x1, y1, x2-x1, y2-y1)

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
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
        # 1. 드래그 중 시각 효과
        if self.selecting and self.roi_start and self.roi_end:
            cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 0, 255), 1)

        # 2. ROI 선택됨
        if self.roi_selected and self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            
            if rw > 0 and rh > 0:
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
                
                frame_roi = frame[ry:ry+rh, rx:rx+rw]
                hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
                
                for color_name, ranges in self.color_ranges.items():
                    mask = np.zeros(hsv.shape[:2], dtype="uint8")
                    for (lower, upper) in ranges:
                        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
                    
                    mask = cv2.erode(mask, None, iterations=1)
                    mask = cv2.dilate(mask, None, iterations=2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if 1000 < cv2.contourArea(contour) < 15000:
                            bx, by, bw, bh = cv2.boundingRect(contour)
                            cx = int(bx + bw/2) + rx
                            cy = int(by + bh/2) + ry
                            
                            if self.latest_depth_img is not None and self.camera_intrinsics:
                                d_y = min(max(cy, 0), 479)
                                d_x = min(max(cx, 0), 639)
                                depth_mm = self.latest_depth_img[d_y, d_x]
                                
                                if depth_mm > 0:
                                    cam_z = depth_mm / 1000.0
                                    cam_x = (cx - self.camera_intrinsics['cx']) * cam_z / self.camera_intrinsics['fx']
                                    cam_y = (cy - self.camera_intrinsics['cy']) * cam_z / self.camera_intrinsics['fy']
                                    
                                    rob_x, rob_y = self.transform_cam_to_robot(cam_x, cam_y)
                                    
                                    # [수정된 부분] 점 대신 십자선(Crosshair) 그리기
                                    # 빨간색 십자선, 크기 20, 두께 2
                                    cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                    
                                    # 텍스트 라벨 개선
                                    label = f"{color_name} Center: ({rob_x:.1f}, {rob_y:.1f})"
                                    cv2.putText(frame, label, (cx - 60, cy - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                    
                                    print(f"[{color_name}] Robot Target: ({rob_x:.1f}, {rob_y:.1f}) mm")

        # 안내 문구
        if not self.roi_selected:
            cv2.putText(frame, "Drag mouse to select Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press 'r' to reset Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.roi_selected = False
            self.roi_rect = None
        elif key == 27:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CubeScanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()