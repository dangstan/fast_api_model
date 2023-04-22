# Script to train machine learning model.

from sklearn.model_selection import KFold
from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
from ml.model import *
from ml.data import process_data

data = pd.read_csv('../data/census.csv')

y = data.pop('salary')

y= y.map({' >50K':1,' <=50K':0})

data = process_data(data, training=True)

data.to_csv('data/pre_train_data.csv', index=False)

kf = KFold(5,shuffle=True)

model = [None,0.0]

for train_idx, test_idx in kf.split(data):

    X_train, X_test = data[train_idx],data[test_idx]
    y_train, y_test = y[train_idx],y[test_idx]

    oversample = BorderlineSMOTE()
    X_train,y_train = oversample.fit_resample(X_train,y_train)

    xgboost = train_model(X_train,y_train)
    score = compute_model_metrics(y_test,inference(X_test))[-1]

    if score>model[1]:
        model = [xgboost,score]


model[0].save_model("data/xgb_model.pkl")
