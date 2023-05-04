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
    data: Dict[str,Union[str,int]]


# Defining the prediction endpoint with data validation
@app.post('/predict')
async def predict(data: InputData):

    dictr = data.dict()['data']
    # Converting input data into Pandas DataFrame
    df = pd.DataFrame({k:(eval(v) if v.isnumeric() else v) for k,v in dictr.items()},index=[0])
    X = process_data(df)

    # Getting the prediction from the XGBoost Regression model
    pred = model.predict(X).tolist()
    return {"prediction":pred[0]}