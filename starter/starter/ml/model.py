from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
from ml.data import process_data
import json


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    imba_pipeline = make_pipeline(BorderlineSMOTE(random_state=42), 
                              XGBClassifier(random_state=42))

    params = {
        'min_child_weight': [1, 10, 50],
        'gamma': [0.5, 1, 2, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [10, 100, 500],
        'n_estimators': [100,400,800]
        }

    new_params = {'xgbclassifier__' + key: params[key] for key in params}

    grid_imba = GridSearchCV(imba_pipeline, param_grid = new_params, scoring='f1', cv=kf)

    grid_imba.fit(X_train, y_train)

    return grid_imba



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : XGBClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    X = process_data(X)
    return model.predict(X)


def slices_performance(X, y, fixes, model):

    cat_ft = json.load(open('../data/cat_ft.json'))

    for col in cat_ft:
        fixes[col] = {}
        for slice in X[col].uniques():

            fixes[col][slice] = {}

            temp = X[X[col]==slice]
            y_t = y[temp.index]

            y_pred = model.predict(temp)

            metrics = compute_model_metrics(y_t,y_pred)

            fixes[col][slice]['precision'] = metrics[0]
            fixes[col][slice]['recall'] = metrics[1]
            fixes[col][slice]['f1'] = metrics[2]

    return fixes
    
