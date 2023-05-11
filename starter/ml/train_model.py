# Script to train machine learning model.

from sklearn.model_selection import KFold
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from model import *
from data import process_data

df = pd.read_csv('data/census.csv')

data,X_ref, y = process_data(df, training=True)

data.to_csv('data/pre_train_data.csv', index=False)

kf = KFold(3,shuffle=True)

model = [None,[0.0]]

X_train, X_test, y_train, y_test  = train_test_split(data,y, test_size=0.25, random_state=11)

gridcv = train_model(X_train,y_train)
score = compute_model_metrics(y_test,gridcv.predict(X_test))

with open('model/gridxgb_model.pkl', 'wb') as f:
    pickle.dump(gridcv, f)


score = compute_model_metrics(y_test,gridcv.predict(X_test))

fixes = {}
fixes['main'] = dict(zip(['precision','recall','f1'],list(score)))
fixes = slices_performance(data, X_ref, y, fixes, gridcv)

with open('data/slice_output.txt', 'w') as file:
    file.write(json.dumps(fixes)) 