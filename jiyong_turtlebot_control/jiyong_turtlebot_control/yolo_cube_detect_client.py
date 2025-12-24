import socket
import struct
import cv2
import numpy as np
import json
import os

# --- 설정 ---
TURTLEBOT_IP = '192.168.110.172'  # 터틀봇 IP 확인 필수!
PORT = 8000

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        print(f"{TURTLEBOT_IP}에 연결 중...")
        client_socket.connect((TURTLEBOT_IP, PORT))
        print("연결 성공! 로그를 출력합니다.\n")

        while True:
            # 1. 헤더 수신
            header = recv_all(client_socket, 8)
            if not header: break
            img_size, json_size = struct.unpack(">LL", header)

            # 2. 데이터 수신
            img_data = recv_all(client_socket, img_size)
            json_data = recv_all(client_socket, json_size)
            if not img_data or not json_data: break

            # 3. 데이터 파싱
            detections = json.loads(json_data.decode('utf-8'))

            # ==========================================
            # [요청하신 로그 출력 부분]
            # ==========================================
            if detections:
                print(f"--- 감지된 객체 {len(detections)}개 ---")
                for item in detections:
                    # 예: [Red, (320, 240)]
                    print(f"[{item['label']}, {tuple(item['coord'])}]")
                print("") # 줄바꿈으로 가독성 확보
            
            # (옵션) 화면은 그냥 원본만 띄워둡니다 (상황 파악용)
            frame = np.frombuffer(img_data, dtype="uint8")
            decoded_frame = cv2.imdecode(frame, 1)
            cv2.imshow("Original View", decoded_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"에러: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()