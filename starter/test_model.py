import pandas as pd
import numpy as np
import scipy.stats
from ml.model import compute_model_metrics,inference
from joblib import load
import json

def test_column_names(data):

    cat_ft = json.load(open('starter/data/cat_ft.json'))
    num_ft = json.load(open('starter/data/num_ft.json'))

    expected_columns = cat_ft + num_ft

    actual_cols = [x[:x.find('_')] if '_' in x else x for x in data.columns]

    # This also enforces the same order
    assert set(expected_columns) == set(actual_cols)


def test_compute_model_metrics():

    metrics =  compute_model_metrics([0,1,1,0,1,0,0,0,1],[1,1,1,0,0,1,0,0,1])

    assert max(metrics)<1 and min(metrics)>0.5


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper capital-gain and capital-loss boundaries
    """

    for col in ['capital-gain','capital-loss']:
        scaler = load('starter/data/std_scalers/'+col+'_std_scaler.bin')
        data[col] = scaler.inverse_transform(data[[col]])
        #data[col] = (data[col] * scaler.scale_) + scaler.mean_

    idx = data['capital-gain'].between(0, 29000) & data['capital-loss'].between(0, 2900)

    assert np.sum(~idx) == 0


def test_row_count(data):
    assert 15000 < data.shape[0] < 40000


def test_inference(init_data,model):
    print(init_data.sample(frac=.2))
    inf = inference(model,init_data.sample(frac=.2).drop(columns=' salary'))
    assert set(list(inf))=={0,1}

