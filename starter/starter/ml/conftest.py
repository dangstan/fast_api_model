import pytest
import pandas as pd
import json


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--model", action="store")
    parser.addoption("--columns", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):

    df = pd.read_csv('data/pre_trained_data.csv')

    return df

@pytest.fixture(scope='session')
def model(request):

    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model("data/xgb_model.pkl")

    return model

@pytest.fixture(scope='session')
def columns(request):
    
    cat_ft = json.load(open('cat_ft.json'))
    num_ft = json.load(open('num_ft.json'))

    return cat_ft + num_ft

@pytest.fixture(scope='session')
def max_gain(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_loss(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
