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
        
        # 토픽 설정 (환경에 따라 다를 수 있음)
        self.color_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.info_topic  = '/camera/camera/color/camera_info'

        self.latest_depth_img = None
        self.camera_intrinsics = None
        
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
        self.get_logger().info('Ready! Check the Z value in terminal.')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def depth_callback(self, msg):
        try: self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError as e: pass

    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError as e: pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True; self.roi_start = (x, y); self.roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting: self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False; self.roi_end = (x, y); self.roi_selected = True
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

        if self.selecting:
            cv2.rectangle(output_image, self.roi_start, self.roi_end, (0, 0, 255), 1)

        hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for (lower, upper) in ranges:
                # [수정된 부분] 리스트를 넘파이 배열로 변환
                lower_np = np.array(lower, dtype="uint8")
                upper_np = np.array(upper, dtype="uint8")
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if 1000 < cv2.contourArea(contour) < 15000:
                    x, y, w, h = cv2.boundingRect(contour)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
                        
                        if self.latest_depth_img is not None and self.camera_intrinsics:
                            # Depth 좌표 클리핑 (IndexError 방지)
                            d_y = min(max(cy, 0), self.latest_depth_img.shape[0]-1)
                            d_x = min(max(cx, 0), self.latest_depth_img.shape[1]-1)
                            
                            depth_mm = self.latest_depth_img[d_y, d_x]
                            
                            if depth_mm > 0:
                                cam_z = depth_mm / 1000.0 # meter
                                fx = self.camera_intrinsics['fx']; fy = self.camera_intrinsics['fy']
                                cx_cam = self.camera_intrinsics['cx']; cy_cam = self.camera_intrinsics['cy']

                                cam_x = (cx - cx_cam) * cam_z / fx
                                cam_y = (cy - cy_cam) * cam_z / fy
                                
                                # 화면 출력 텍스트
                                text_str = f"{color_name} Z:{cam_z:.3f}m"
                                
                                # 터미널 로그 출력
                                print(f"[{color_name}] Cam: ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f})")

                                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.circle(output_image, (cx, cy), 5, (255, 255, 255), -1)
                                cv2.putText(output_image, text_str, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not self.roi_selected:
            cv2.putText(output_image, "Drag mouse to select Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, output_image)
        if cv2.waitKey(1) & 0xFF == 27: rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CubeDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()