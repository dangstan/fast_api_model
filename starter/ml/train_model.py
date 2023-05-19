'''
train_model.py

This script is used to train a machine learning model using data from the 'census.csv' file. It performs the following steps:
1. Load the data from 'census.csv'.
2. Preprocess the data using the 'process_data' function from the 'data' module.
3. Save the preprocessed data to 'pre_train_data.csv'.
4. Split the data into training and test sets.
5. Train the model using Grid Search Cross Validation with the 'train_model' function from the 'model' module.
6. Compute the model's metrics (precision, recall, f1) using 'compute_model_metrics' function from the 'model' module.
7. Save the trained model to 'gridxgb_model.pkl'.
8. Compute slice performance and store it in 'slice_output.txt' file.

Usage: 
Run this script from the command line using 'python ml/train_model.py'.
'''

from sklearn.model_selection import KFold
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from model import *
from data import process_data

df = pd.read_csv('data/census.csv')

data, X_ref, y = process_data(df, training=True)

data.to_csv('data/pre_train_data.csv', index=False)

kf = KFold(3, shuffle=True)

model = [None, [0.0]]

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.25, random_state=11)

gridcv = train_model(X_train, y_train)
score = compute_model_metrics(y_test, gridcv.predict(X_test))

with open('model/gridxgb_model.pkl', 'wb') as f:
    pickle.dump(gridcv, f)


score = compute_model_metrics(y_test, gridcv.predict(X_test))

fixes = {}
fixes['main'] = dict(zip(['precision', 'recall', 'f1'], list(score)))
fixes = slices_performance(data, X_ref, y, fixes, gridcv)

with open('data/slice_output.txt', 'w') as file:
    file.write(json.dumps(fixes))
