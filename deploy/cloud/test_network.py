import requests
import socket

def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

# 测试本地
print("Testing localhost...")
local_ok = check_port("127.0.0.1", 8501)
print(f"Local 8501: {'OK' if local_ok else 'FAIL'}")

# 测试外部IP
print("\nTesting external IP 121.48.164.7...")
external_ok = check_port("121.48.164.7", 8501)
print(f"External 8501: {'OK' if external_ok else 'FAIL'}")

external_8000 = check_port("121.48.164.7", 8000)
print(f"External 8000: {'OK' if external_8000 else 'FAIL'}")