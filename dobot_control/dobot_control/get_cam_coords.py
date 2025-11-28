import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class RawCameraReader(Node):
    def __init__(self):
        super().__init__('raw_cam_reader')
        self.bridge = CvBridge()
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.latest_depth_img = None
        self.camera_intrinsics = None
        self.color_ranges = {
            'Red':   [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'Blue':  [([100, 150, 0], [140, 255, 255])],
            'Green': [([40, 70, 70], [80, 255, 255])],
            'Yellow':[([20, 100, 100], [30, 255, 255])]
        }
        self.get_logger().info("📷 Show me the Cubes! Printing RAW Camera Coords...")

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

    def process_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 간단하게 가장 큰 큐브 하나씩만 찾아서 좌표 출력
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for (lower, upper) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
            
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if 1000 < cv2.contourArea(contour) < 20000:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    cx = int(bx + bw/2)
                    cy = int(by + bh/2)
                    
                    if self.latest_depth_img is not None and self.camera_intrinsics:
                        d_y = min(max(cy, 0), 479); d_x = min(max(cx, 0), 639)
                        depth_mm = self.latest_depth_img[d_y, d_x]
                        
                        if depth_mm > 0:
                            cam_z = depth_mm / 1000.0
                            cam_x = (cx - self.camera_intrinsics['cx']) * cam_z / self.camera_intrinsics['fx']
                            cam_y = (cy - self.camera_intrinsics['cy']) * cam_z / self.camera_intrinsics['fy']
                            
                            # 화면 표시
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                            cv2.putText(frame, f"{color_name}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # ★ 핵심: 터미널에 Raw 좌표 출력
                            print(f"[{color_name}] RAW Camera: {cam_x:.4f}, {cam_y:.4f}")

        cv2.imshow("Raw Camera Reader", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RawCameraReader()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()