import requests

url = 'http://localhost:5002/api'
r = requests.post(url, json={'exp': 1.8, })
print(r.json())
