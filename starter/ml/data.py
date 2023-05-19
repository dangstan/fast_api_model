'''
This file contains a function process_data that preprocesses 
the data before it can be used for training or prediction. 
It takes care of encoding categorical variables, scaling 
numerical variables, and other data cleaning tasks.
'''

import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from joblib import dump, load
import json
import os


def process_data(
    X, training=False
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    """

    X.columns = X.columns.str.strip()
    df_obj = X.select_dtypes(['object'])
    X[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    X['sex'] = X['sex'].map({'Male': 1, 'Female': 0})

    X_obj = X.select_dtypes(['object'])
    X[X_obj.columns] = X_obj.apply(lambda x: x.str.strip())

    X = X[~((X['workclass'] == '?') | (X['occupation'] == '?')
            | (X['native-country'] == '?'))]

    if training is True:

        num_ft = X.dtypes[X.dtypes == 'int64'].index.tolist()
        cat_ft = X.dtypes[X.dtypes != 'int64'].index.tolist()[:-1]

        with open('data/cat_ft.json', 'w') as f:
            json.dump(cat_ft, f)
        with open('data/num_ft.json', 'w') as f:
            json.dump(num_ft, f)

        if not os.path.exists("data/std_scalers"):
            os.makedirs("data/std_scalers")

        scaler = RobustScaler()
        for col in num_ft:
            X[col] = scaler.fit_transform(X[[col]])
            dump(
                scaler,
                'data/std_scalers/' +
                col +
                '_std_scaler.bin',
                compress=True)

        X = X[(X['capital-gain'] < 28000) & (X['capital-loss'] < 2800)]

        y = X.pop('salary')
        y = y.map({'>50K': 1, '<=50K': 0})

        dummies = pd.get_dummies(X.loc[:, cat_ft])
        df_ref = X.copy()
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=cat_ft)
        with open('data/dummies.json', 'w') as f:
            json.dump(dummies.columns.tolist(), f)

        return X, df_ref, y

    else:

        cat_ft = json.load(open('starter/data/cat_ft.json'))
        num_ft = json.load(open('starter/data/num_ft.json'))

        X = X[X.columns[X.columns.isin(cat_ft + num_ft)]]

        for col in num_ft:
            scaler = load(
                'starter/data/std_scalers/' +
                col +
                '_std_scaler.bin')
            X[col] = scaler.transform(X[[col]])

        default_dumm = json.load(open('starter/data/dummies.json'))
        dummies = pd.get_dummies(X.loc[:, cat_ft])
        X = pd.concat([X, dummies], axis=1)
        X[[x for x in default_dumm if x not in dummies.columns]] = 0
        X = X.drop(columns=cat_ft)

        print(X)
        print(X.columns.tolist())

        return X
