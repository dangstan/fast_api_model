from fastapi.testclient import TestClient
import os
from main import app

client = TestClient(app)

def test_get_method():
    
    response = client.get('/')

    assert response.status_code == 200, 'response failed'
    assert response.json() == {"greeting":"Welcome to my API!"} 


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
    assert response.json() == {"prediction":">50K"}, "wrong prediction: expected >50K, but the result was <=50K"

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
    assert response.json() == {"prediction":"<=50K"}, "wrong prediction: expected <=50K, but the result was >50K"

def test_if_file_in_subfolder():
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk():
        for filename in files:
            # Join the two strings to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    print(file_paths)

    assert set([x for x in file_paths if x.endswith('census.csv')]) == {True,False}