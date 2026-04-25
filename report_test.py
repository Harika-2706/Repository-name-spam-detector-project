import requests

url = "http://127.0.0.1:5000/report"
data = {"message": "You won free money!!!"}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response Text:", response.text)