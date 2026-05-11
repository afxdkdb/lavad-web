import requests
r = requests.get("http://localhost:8000/model_status")
print(r.json())