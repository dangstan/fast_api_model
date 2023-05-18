import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Union
import pickle
import json
from ml.data import process_data

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
except:
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


# Defining the prediction endpoint with data validation
@app.post('/predict')
async def predict(data: InputData):

    dictr = data.dict(by_alias=True)
    # Converting input data into Pandas DataFrame
    df = pd.DataFrame(dictr,index=[0])
    
    # Processing the data
    X = process_data(df)

    # Getting the prediction from the XGBoost Classification model
    pred = ['>50K' if model.predict(X.values).tolist()[0]==1 else '<=50K'][0]
    return {"prediction":pred}