import requests

try:
    resp = requests.get('http://localhost:8501', timeout=5)
    print(f'Frontend status: {resp.status_code}')
except Exception as e:
    print(f'Frontend error: {e}')

try:
    resp = requests.get('http://localhost:8000/health', timeout=5)
    print(f'Backend health: {resp.json()}')
except Exception as e:
    print(f'Backend error: {e}')