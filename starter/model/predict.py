"""
predict.py

This script loads a pre-trained XGBoost model and defines an endpoint for making predictions with the model.

The script uses the Pydantic BaseModel for input data validation, and FastAPI's app for defining the POST request endpoint.

The input data for the '/predict' endpoint should be provided in the following format:
{
    "data": {
        "feature1": value1,
        "feature2": value2,
        ...
    }
}

Once the input data is validated and processed, it is fed into the loaded model for predictions. The predicted value is returned in the response.

Script Steps:
1. Load the serialized XGBoost model from 'gridxgb_model.pkl'.
2. Define a Pydantic model 'InputData' for input data validation.
3. Define a POST endpoint '/predict' that:
    - Validates and processes the input data.
    - Makes a prediction using the pre-loaded model.
    - Returns the prediction.

Usage: 
This script is designed to be run as a FastAPI application with Uvicorn or another ASGI server.
"""

from pydantic import BaseModel
from typing import Dict, Union
from main import app
import pandas as pd
import pickle
import json
from starter.ml.data import process_data


# Loading in model from serialized .pkl file
pkl_filename = "model/gridxgb_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)


class InputData(BaseModel):
    data: Dict[str, Union[str, int]]


# Defining the prediction endpoint with data validation
@app.post('/predict')
async def predict(data: InputData):

    dictr = data.dict()['data']
    # Converting input data into Pandas DataFrame
    df = pd.DataFrame({k: (eval(v) if v.isnumeric() else v)
                      for k, v in dictr.items()}, index=[0])
    X = process_data(df)

    # Getting the prediction from the XGBoost Regression model
    pred = model.predict(X).tolist()
    return {"prediction": pred[0]}
