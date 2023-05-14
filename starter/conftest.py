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

    df = pd.read_csv('starter/data/pre_train_data.csv')

    return df

@pytest.fixture(scope='session')
def init_data(request):

    df = pd.read_csv('starter/data/census.csv')

    return df

@pytest.fixture(scope='session')
def model(request):

    import pickle

    filename = "starter/model/gridxgb_model.pkl"

    model = pickle.load(open(filename, 'rb'))

    return model

@pytest.fixture(scope='session')
def columns(request):
    
    cat_ft = json.load(open('starter/data/cat_ft.json'))
    num_ft = json.load(open('starter/data/num_ft.json'))

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
