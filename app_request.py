import json
import requests

url = 'https://fastapi-model-ncoa.onrender.com/predict'

sample = {'age': 22,
        'workclass': ' Private',
        'fnlgt': 201490,
        'education': ' HS-grad',
        'education-num': 9,
        'marital-status': ' Never-married',
        'occupation': ' Adm-clerical',
        'relationship': ' Own-child',
        'race': ' White',
        'sex': ' Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 20,
        'native-country': ' United-States',
        }

headers = {'Content-type': "application/json"}

response = requests.post(url,data=json.dumps(sample), headers=headers)
print('Status Code: ',response.status_code)
print('Prediction: ',response.text)