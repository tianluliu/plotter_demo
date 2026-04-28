import socket
import time

active_socket = None


def send_gcode(ip, port, gcode, stop_event=None):
    global active_socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            active_socket = s
            s.settimeout(0.2)
            s.connect((ip, port))
            time.sleep(1)

            for line in gcode.splitlines():
                if stop_event is not None and stop_event.is_set():
                    return False

                line = line.strip()
                if not line or line in ["M2", "M30"]:
                    continue

                print("SEND:", line)
                s.sendall((line + "\n").encode("utf-8"))

                # small delay only, do not wait forever for ok
                time.sleep(0.03)

            return True

    except OSError:
        return False

    finally:
        active_socket = None


def emergency_stop_socket():
    global active_socket

    if active_socket is not None:
        try:
            active_socket.sendall(b"\x18")  # GRBL soft reset: clears buffer
            time.sleep(0.1)
        except Exception:
            pass

        try:
            active_socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

        try:
            active_socket.close()
        except Exception:
            pass

def return_to_start(ip="192.168.4.1", port=23):
    cmds = [
        "\r\n\r\n",
        "$X",
        "G90",
        "G0 Z5",
        "G0 X0 Y0",
    ]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        s.connect((ip, port))

        for cmd in cmds:
            s.sendall((cmd + "\n").encode("utf-8"))
            time.sleep(0.25)
