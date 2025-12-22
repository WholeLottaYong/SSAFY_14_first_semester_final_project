import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import time
import threading

# ==========================================
# [설정] 라즈베리파이 주소
# ==========================================
RPI_IP = '192.168.110.150'
RPI_PORT = 65432
# ==========================================

class ConveyorDriver(Node):

    def __init__(self):
        super().__init__('conveyor_driver')
        
        # 1. 터틀봇 명령 수신 (Sub)
        self.subscription = self.create_subscription(
            String,
            '/gripper_cmd',
            self.listener_callback,
            10)
            
        # 2. [추가] 두봇에게 상태 알림 (Pub)
        self.status_publisher = self.create_publisher(String, '/conveyor_status', 10)
        
        self.is_moving = False
        self.get_logger().info('✅ Conveyor Driver Started.')

    def listener_callback(self, msg):
        command = msg.data.strip()
        if command == 'move':
            if not self.is_moving:
                self.get_logger().info('✨ "move" command received. Triggering conveyor sequence...')
                threading.Thread(target=self.trigger_conveyor).start()
            else:
                self.get_logger().warning('⚠️ Conveyor is busy!')
        
    def trigger_conveyor(self):
        self.is_moving = True
        try:
            self.get_logger().info('⏳ Waiting 2.0s before signal...')
            time.sleep(2.0)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10.0) 
                s.connect((RPI_IP, RPI_PORT))
                s.sendall(b'move')
                self.get_logger().info(f'🚀 Signal sent to RPi.')

                # RPi로부터 'Done' 대기
                data = s.recv(1024)
                response = data.decode().strip()
                
                if response == "Done":
                    self.get_logger().info('✅ Conveyor Stopped. Publishing status...')
                    # [추가] 두봇에게 "멈췄다"고 알림!
                    msg = String()
                    msg.data = "stopped"
                    self.status_publisher.publish(msg)
                else:
                    self.get_logger().error(f'❌ RPi Error: {response}')
                    
        except Exception as e:
            self.get_logger().error(f'❌ Connection Error: {e}')
        finally:
            self.is_moving = False

def main(args=None):
    rclpy.init(args=args)
    node = ConveyorDriver()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()