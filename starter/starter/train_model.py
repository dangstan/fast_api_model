# Script to train machine learning model.

from sklearn.model_selection import KFold
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import *
from ml.data import process_data

data = pd.read_csv('../data/census.csv')

data, y = process_data(data, training=True)

data.to_csv('../data/pre_train_data.csv', index=False)

kf = KFold(3,shuffle=True)

model = [None,[0.0]]

X_train, X_test, y_train, y_test  = train_test_split(data,y, test_size=0.25, random_state=11)

for train_idx, test_idx in kf.split(X_train):

    X_train_2 = data[data.index.isin(train_idx)]
    y_train_2 = y[y.index.isin(train_idx)]

    oversample = BorderlineSMOTE()
    X_train_2,y_train_2 = oversample.fit_resample(X_train_2,y_train_2)

    xgboost = train_model(X_train_2,y_train_2)
    score = compute_model_metrics(y_test,xgboost.predict(X_test))
    print(score)

    if score[-1]>model[1][-1]:
        model = [xgboost,score]

fixes = {}
fixes['main'] = dict(zip(['precision','recall','f1'],list(model[1])))
fixes = slices_performance(data, y, fixes, model[0])

with open('../data/slice_output.txt', 'w') as file:
    file.write(json.dumps(fixes)) 

model[0].save_model("../data/xgb_model.pkl")