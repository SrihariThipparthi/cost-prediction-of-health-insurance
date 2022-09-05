import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':100, 'sex':2, 'bmi':100, 'children':5, 'smoker':2,'region':4})

print(r.json())