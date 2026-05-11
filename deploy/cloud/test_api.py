import requests
try:
    resp = requests.get('http://localhost:8000/health', timeout=5)
    print('Health check:', resp.json())
except Exception as e:
    print('Error:', e)

try:
    resp = requests.get('http://localhost:8000/demo_results', timeout=5)
    print('Demo results:', resp.json())
except Exception as e:
    print('Error:', e)

try:
    resp = requests.get('http://localhost:8000/model_status', timeout=5)
    print('Model status:', resp.json())
except Exception as e:
    print('Error:', e)