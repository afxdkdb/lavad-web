import requests

r = requests.get('http://localhost:8501', timeout=10)
print(f'Status: {r.status_code}')
print(f'Content-Type: {r.headers.get("content-type")}')
print(f'Content length: {len(r.text)}')
print(f'First 2000 chars:')
print(r.text[:2000])