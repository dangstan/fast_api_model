import pandas as pd
import numpy as np
import scipy.stats
from model import compute_model_metrics,inference


def test_column_names(data):

    expected_colums = [
        'workclass',
        'capital-loss',
        'capital-gain',
        'race',
        'occupation',
        'native-country',
        'fnlgt',
        'hours-per-week',
        'education',
        'relationship',
        'age',
        'education-num',
        'marital-status'
    ]

    # This also enforces the same order
    assert expected_colums == data.columns.tolist()


def test_compute_model_metrics():

    metrics =  compute_model_metrics([0,1,1,0,1,0,0,0,1],[1,1,1,0,0,1,0,0,1])

    assert max(metrics)<1 and min(metrics)>0.6


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper capital-gain and capital-loss boundaries
    """
    idx = data['capital-gain'].between(0, 28000) & data['capital-loss'].between(0, 2800)

    assert np.sum(~idx) == 0


def test_row_count(data):
    assert 15000 < data.shape[0] < 40000


def test_inference(data):
    assert set(list(inference(data.iloc[:,200])))=={0,1}

