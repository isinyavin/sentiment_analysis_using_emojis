import requests

url = "http://127.0.0.1:8002/predict_emojis"
data = {"text": "hey"}
response = requests.post(url, json=data)
print(response.json())
