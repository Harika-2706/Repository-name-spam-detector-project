import requests

url = "http://127.0.0.1:5000/predict"
data = {"message": "Claim your prize now"}

response = requests.post(url, json=data)
print(response.json())