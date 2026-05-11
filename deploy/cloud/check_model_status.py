import requests
try:
    resp = requests.get('http://localhost:8000/model_status', timeout=5)
    print('Model status:', resp.json())
except Exception as e:
    print('Error:', e)