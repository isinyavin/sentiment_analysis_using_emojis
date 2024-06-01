import requests

url = "http://127.0.0.1:8017/predict_emojis"
data = {"text": "the british are coming"}
response = requests.post(url, json=data)
print(response.json())
