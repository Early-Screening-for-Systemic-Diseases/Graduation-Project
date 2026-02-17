import urllib.request
import json

url = 'http://127.0.0.1:8000/predict'
data = {"data": {"GenHlth": 3, "BMI": 31, "HighBP": 1, "Age": 9, "HighChol": 1, "PhysHlth": 5, "DiffWalk": 0, "PhysActivity": 0}}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=30) as resp:
    print('Status:', resp.status)
    body = resp.read().decode('utf-8')
    print('Body:', body)
    try:
        print('JSON:', json.loads(body))
    except Exception as e:
        print('JSON parse error:', e)
