# request.py

# This is a test document
import requests

# request api from port 5002
payload = {
	'exp':5.8
}
url = 'http://localhost:5002/api'
r = requests.post(url, json=payload)
print(r.json())
