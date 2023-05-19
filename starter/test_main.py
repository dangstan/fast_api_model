'''
This file includes tests for the main FastAPI application. 
There are three tests in total. One is for the GET request, 
and the other two are for the POST request. 
The tests for the POST request send example data and check 
the output of the model to see if it matches the expected output.
'''


from fastapi.testclient import TestClient
import os
from main import app

client = TestClient(app)


def test_get_method():

    response = client.get('/')

    assert response.status_code == 200, 'response failed'
    assert response.json() == {"greeting": "Welcome to my API!"}


def test_post_inference_example_one():

    input_data = {'age': 52,
                  'workclass': ' Self-emp-inc',
                  'fnlgt': 287927,
                  'education': ' HS-grad',
                  'education-num': 9,
                  'marital-status': ' Married-civ-spouse',
                  'occupation': ' Exec-managerial',
                  'relationship': ' Wife',
                  'race': ' White',
                  'sex': ' Female',
                  'capital-gain': 15024,
                  'capital-loss': 0,
                  'hours-per-week': 40,
                  'native-country': ' United-States'
                  }

    response = client.post('/predict', json=input_data)

    assert response.status_code == 200, 'response failed'
    assert response.json() == {
        "prediction": ">50K"}, "wrong prediction: expected >50K, but the result was <=50K"


def test_post_inference_example_two():

    input_data = {'age': 22,
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

    response = client.post('/predict', json=input_data)

    assert response.status_code == 200, 'response failed'
    assert response.json() == {
        "prediction": "<=50K"}, "wrong prediction: expected <=50K, but the result was >50K"
