import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# =========================================================
# 🎯 [User Data] 변환 행렬
# =========================================================
TRANSFORM_MATRIX = np.array([
    [-967.27151, -106.58200, 181.96182],
    [-95.73259, 952.66342, 57.90632]
])

class CubeScanner(Node):
    def __init__(self):
        super().__init__('cube_scanner_node')
        
        self.bridge = CvBridge()
        
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)

        self.latest_depth_img = None
        self.camera_intrinsics = None
        
        # --- ROI 변수 ---
        self.roi_start = None; self.roi_end = None
        self.selecting = False; self.roi_selected = False
        self.roi_rect = None 

        # --- 색상별 데이터 관리 ---
        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }
        
        self.scan_data = {color: {'buffer': [], 'fixed': None} for color in self.color_ranges}
        self.is_collecting = False

        self.window_name = "Multi-Cube Scanner"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.get_logger().info("Ready! Drag ROI -> Press 'c' to Capture ALL cubes.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.roi_start = (x, y)
            self.roi_end = (x, y) # 시작할 때 끝점도 초기화
            self.roi_selected = False
            self.reset_data() # 새로 드래그하면 데이터 초기화
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.roi_end = (x, y) # 드래그 중 끝점 업데이트
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.roi_end = (x, y)
            self.roi_selected = True
            
            # 좌표 정규화 (거꾸로 드래그해도 되게)
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
        # [핵심 수정 1] 원본 프레임은 분석용으로 유지하고, 화면 출력용 복사본을 만듭니다.
        # 이렇게 하면 화면에 그리는 박스(ROI Box)가 분석에 영향을 주지 않습니다.
        vis_frame = frame.copy()

        # 1. 드래그 중 시각 효과 (빨간 점선 박스 느낌)
        if self.selecting and self.roi_start and self.roi_end:
            cv2.rectangle(vis_frame, self.roi_start, self.roi_end, (0, 0, 255), 1)

        # 2. ROI가 선택되었을 때
        if self.roi_selected and self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            
            # [핵심 수정 2] ROI 박스 색상을 흰색(255,255,255)으로 변경하여 Blue 인식 방지
            cv2.rectangle(vis_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 2)
            
            # 고정된 값 리스트업 (화면 좌측)
            fixed_y_offset = 0
            for color, data in self.scan_data.items():
                if data['fixed'] is not None:
                    fx, fy = data['fixed']
                    text = f"FIXED [{color}]: X={fx:.1f}, Y={fy:.1f}"
                    cv2.putText(vis_frame, text, (rx, ry - 20 - fixed_y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    fixed_y_offset += 25

            # 분석은 깨끗한 원본 'frame'에서 수행
            frame_roi = frame[ry:ry+rh, rx:rx+rw]
            hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
            
            for color_name, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], dtype="uint8")
                for (lower, upper) in ranges:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
                
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                max_contour = None
                max_area = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 1000 < area < 15000:
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                
                if max_contour is not None:
                    bx, by, bw, bh = cv2.boundingRect(max_contour)
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
                            
                            # --- 시각화는 vis_frame에 그립니다 ---
                            if self.scan_data[color_name]['fixed']:
                                final_x, final_y = self.scan_data[color_name]['fixed']
                                cv2.drawMarker(vis_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                                label = f"{color_name}(FIX): {final_x:.1f},{final_y:.1f}"
                                color_rgb = (0, 255, 0)
                            else:
                                cv2.drawMarker(vis_frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                label = f"{color_name}: {rob_x:.1f},{rob_y:.1f}"
                                color_rgb = (0, 255, 255)
                                
                                # 데이터 수집
                                if self.is_collecting:
                                    self.scan_data[color_name]['buffer'].append((rob_x, rob_y))
                                    cnt = len(self.scan_data[color_name]['buffer'])
                                    label += f" [{cnt}/60]"
                                    
                                    if cnt >= 60:
                                        avg_x = np.mean([p[0] for p in self.scan_data[color_name]['buffer']])
                                        avg_y = np.mean([p[1] for p in self.scan_data[color_name]['buffer']])
                                        self.scan_data[color_name]['fixed'] = (avg_x, avg_y)
                                        print(f"✅ FINAL {color_name}: X={avg_x:.2f}, Y={avg_y:.2f}")

                            cv2.putText(vis_frame, label, (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)

        # 안내 문구
        if not self.roi_selected:
            cv2.putText(vis_frame, "Drag mouse to select Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            if self.is_collecting:
                cv2.putText(vis_frame, "Collecting Data... Please Wait", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(vis_frame, "'c': Capture (2sec) | 'r': Reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.reset_data()
            self.is_collecting = True
            print("\n--- Start Collecting for ALL cubes ---")
        elif key == ord('r'):
            self.reset_data()
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