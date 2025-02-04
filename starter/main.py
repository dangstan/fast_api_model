'''
This is the main FastAPI application. It has two endpoints. 
One is a simple GET request that returns a greeting. 
The second is a POST request where you send in data about a person, 
and the app returns whether they make over 50K or less. 
It loads a pre-trained XGBoost model from a pickle file to make the prediction.
'''



import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Union
import pickle
import json

try:
    from ml.data import process_data
except:
    from starter.ml.data import process_data

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my API!"}


# Loading in model from serialized .pkl file

try:
    pkl_filename = "starter/model/gridxgb_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
except BaseException:
    pkl_filename = "model/gridxgb_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                'age': 58,
                'workclass': ' Private',
                'fnlgt': 151910,
                'education': ' HS-grad',
                'education-num': 9,
                'marital-status': ' Widowed',
                'occupation': ' Adm-clerical',
                'relationship': ' Unmarried',
                'race': ' White',
                'sex': ' Female',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': ' United-States',
                'salary': ' <=50K'
            }
        }


# Defining the prediction endpoint with data validation
@app.post('/predict')
async def predict(data: InputData):

    dictr = data.dict(by_alias=True)
    # Converting input data into Pandas DataFrame
    df = pd.DataFrame(dictr, index=[0])

    # Processing the data
    X = process_data(df)

    # Getting the prediction from the XGBoost Classification model
    pred = ['>50K' if model.predict(X.values).tolist()[0] == 1 else '<=50K'][0]
    return {"prediction": pred}
