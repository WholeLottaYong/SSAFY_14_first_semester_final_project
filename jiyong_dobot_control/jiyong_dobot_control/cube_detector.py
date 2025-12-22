import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class CubeDetector(Node):
    def __init__(self):
        super().__init__('cube_detector_node')
        
        self.bridge = CvBridge()
        
        self.color_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.info_topic  = '/camera/camera/color/camera_info'

        self.latest_depth_img = None
        self.camera_intrinsics = None
        self.dist_coeffs = None
        
        self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.color_topic, self.color_callback, qos_profile_sensor_data)
        
        # ROI 관련
        self.roi_start = None; self.roi_end = None; self.selecting = False
        self.roi_selected = False; self.roi_rect = None
        self.window_name = "Z-Value Reader"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }

        # [추가] 데이터 캡처 및 상태 관리 변수
        self.is_capturing = False
        self.measurement_done = False # 측정이 끝나면 True가 되어 로그를 멈춤
        self.capture_count = 0
        self.capture_limit = 60
        self.capture_data = {k: [] for k in self.color_ranges.keys()}

        self.get_logger().info('Ready! Press "c" to capture, "r" to reset.')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4], 
                'cx': msg.k[2], 'cy': msg.k[5]
            }
            self.dist_coeffs = np.array(msg.d)

    def depth_callback(self, msg):
        try: self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError: pass

    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError: pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True; self.roi_start = (x, y); self.roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting: self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False; self.roi_end = (x, y); self.roi_selected = True
            if self.roi_start and self.roi_end:
                x1 = min(self.roi_start[0], self.roi_end[0]); y1 = min(self.roi_start[1], self.roi_end[1])
                x2 = max(self.roi_start[0], self.roi_end[0]); y2 = max(self.roi_start[1], self.roi_end[1])
                self.roi_rect = (x1, y1, x2-x1, y2-y1)

    def process_image(self, frame):
        processing_frame = frame
        output_image = frame.copy()
        
        if self.roi_selected and self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            if w > 0 and h > 0:
                mask_roi = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.rectangle(mask_roi, (x, y), (x+w, y+h), 255, -1)
                processing_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if self.selecting and self.roi_start and self.roi_end:
            cv2.rectangle(output_image, self.roi_start, self.roi_end, (0, 0, 255), 1)

        hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)

        # 현재 프레임에서 발견된 색상들 처리
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for (lower, upper) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8")))

            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if 1000 < cv2.contourArea(contour) < 15000:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
                        
                        if self.latest_depth_img is not None and self.camera_intrinsics:
                            d_y = min(max(cy, 0), self.latest_depth_img.shape[0]-1)
                            d_x = min(max(cx, 0), self.latest_depth_img.shape[1]-1)
                            depth_mm = self.latest_depth_img[d_y, d_x]
                            
                            if depth_mm > 0:
                                cam_z = depth_mm / 1000.0 # meter
                                fx = self.camera_intrinsics['fx']; fy = self.camera_intrinsics['fy']
                                cx_cam = self.camera_intrinsics['cx']; cy_cam = self.camera_intrinsics['cy']

                                if self.dist_coeffs is not None:
                                    src_pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                                    camera_matrix = np.array([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]], dtype=np.float32)
                                    dst_pt = cv2.undistortPoints(src_pt, camera_matrix, self.dist_coeffs, P=camera_matrix)
                                    cx_fixed = dst_pt[0][0][0]
                                    cy_fixed = dst_pt[0][0][1]
                                else:
                                    cx_fixed, cy_fixed = cx, cy

                                cam_x = (cx_fixed - cx_cam) * cam_z / fx
                                cam_y = (cy_fixed - cy_cam) * cam_z / fy
                                
                                text_str = f"{color_name} Z:{cam_z:.3f}m"
                                
                                # 캡처 중도 아니고, 측정 완료 상태도 아닐 때만 로그 출력
                                if not self.is_capturing and not self.measurement_done:
                                    print(f"[{color_name}] Cam: ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f})")
                                
                                # [수정] 캡처 모드일 경우 데이터 저장 (x, y, z 모두 저장)
                                if self.is_capturing:
                                    self.capture_data[color_name].append((cam_x, cam_y, cam_z))

                                cv2.rectangle(output_image, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                                cv2.circle(output_image, (cx, cy), 5, (255, 255, 255), -1)
                                cv2.putText(output_image, text_str, (bx, by - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 캡처 로직 처리
        if self.is_capturing:
            self.capture_count += 1
            status_text = f"Capturing... {self.capture_count}/{self.capture_limit}"
            cv2.putText(output_image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if self.capture_count >= self.capture_limit:
                self.finalize_capture()
                self.is_capturing = False
        
        # 측정 완료 상태 표시
        if self.measurement_done:
             cv2.putText(output_image, "Result Fixed. Press 'r' to reset", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not self.roi_selected:
            cv2.putText(output_image, "Drag mouse to select Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, output_image)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            rclpy.shutdown()
        elif key == ord('c'): # Capture
            if not self.is_capturing and not self.measurement_done:
                self.start_capture()
        elif key == ord('r'): # Reset
            self.reset_measurement()

    def start_capture(self):
        self.is_capturing = True
        self.capture_count = 0
        for k in self.capture_data:
            self.capture_data[k] = []
        self.get_logger().info("Started capturing 60 frames...")

    def reset_measurement(self):
        self.measurement_done = False
        self.is_capturing = False
        self.capture_count = 0
        self.get_logger().info("Reset! Logs resumed. Press 'c' to capture again.")

    def finalize_capture(self):
        target_order = ['Red', 'Blue', 'Green', 'Yellow']
        
        print("\n" + "="*40)
        print("CAPTURE COMPLETE (Avg of 60 frames)")
        print("="*40)
        
        output_str = "solution)\n# 1. 카메라 좌표\npts_camera = np.float32([\n"
        
        z_values = [] # Z값 평균을 저장할 리스트
        
        for color in target_order:
            points = self.capture_data.get(color, [])
            if len(points) > 0:
                avg_x = np.mean([p[0] for p in points])
                avg_y = np.mean([p[1] for p in points])
                avg_z = np.mean([p[2] for p in points]) # Z값 평균 계산
                
                output_str += f"    [{avg_x:.4f}, {avg_y:.4f}],   # {color}\n"
                z_values.append(f"{avg_z:.3f}")
            else:
                output_str += f"    [0.0000, 0.0000],   # {color} (Not Detected)\n"
                z_values.append("0.000")
        
        output_str += "])\n\n"
        
        # [추가] 2. 카메라 좌표 평균 z 값 출력 섹션
        output_str += "# 2. 카메라 좌표 평균 z 값 :\n"
        output_str += f"[{', '.join(z_values)}] # Red, Blue, Green, Yellow"

        print(output_str)
        print("="*40)
        print("Log stopped. Press 'r' to restart.")
        
        # 측정 완료 플래그 활성화 -> 로그 중단
        self.measurement_done = True

def main(args=None):
    rclpy.init(args=args)
    node = CubeDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()