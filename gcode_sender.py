import socket
import time

def send_gcode(ip, port, gcode):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect((ip, port))

        time.sleep(1)

        for line in gcode.splitlines():
            line = line.strip()
            if not line:
                continue

            s.sendall((line + "\n").encode("utf-8"))
            time.sleep(0.03)