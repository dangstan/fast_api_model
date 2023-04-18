# Script to train machine learning model.

from sklearn.model_selection import KFold
from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
from ml.model import *

data = pd.read_csv('../data/census.csv')

y = data.pop('salary')

y= y.map({' >50K':1,' <=50K':0})

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
