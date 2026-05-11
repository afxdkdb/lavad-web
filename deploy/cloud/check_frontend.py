import requests

try:
    r = requests.get('http://localhost:8501', timeout=10)
    print(f'Status: {r.status_code}')
    print(f'Length: {len(r.text)}')
    print(f'Contains streamlit: {"streamlit" in r.text.lower()}')
    print(f'Contains script: {"script" in r.text.lower()}')
except Exception as e:
    print(f'Error: {e}')