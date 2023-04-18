import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import json


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

    cat_ft = json.load(open('cat_ft.json'))
    num_ft = json.load(open('num_ft.json'))

    X['sex']=X['sex'].map({' Male':1,' Female':0})

    if training is True:

        scaler = StandardScaler()
        for col in num_ft:
            X[col] = scaler.fit_transform(X[[col]])
            dump(scaler, 'std_scalers/'+col+'_std_scaler.bin', compress=True)

        X = X[(X['capital-gain']<28000) & (X['capital-loss']<2800)]
        X = X[~((X['workclass']==' ?') | (X['occupation']==' ?') | (X['native-country']==' ?'))]
        
        dummies = pd.get_dummies(X.loc[:,cat_ft])
        X = pd.concat([X,dummies],axis=1)
        X = X.drop(columns=cat_ft)
        
        with open('dummies.json', 'w') as f:
            json.dump(dummies.columns.tolist(), f)

    else:

        for col in num_ft:
            load(scaler, 'std_scalers/'+col+'_std_scaler.bin')
            X[col] = scaler.transform(X[[col]])

        dumm_cols = json.load(open('dummies.json'))
        new_d = pd.get_dummies(X.loc[:,cat_ft])
        new_d = new_d[new_d.columns[new_d.columns.isin(dumm_cols)]]
        X = pd.concat([X,new_d],axis=1)
        X[X.columns[~X.columns.isin(dumm_cols)]] = 0


    return X
