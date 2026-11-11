import pytest

from cs_outbound_coli_prediction.config import input_path
from cs_outbound_coli_prediction.data import load_deliveries
from cs_outbound_coli_prediction.pipeline import train_models


@pytest.fixture(scope="session")
def deliveries():
    """The first 20 stores of the input data: enough for the tests, fast to train on."""
    data = load_deliveries(input_path)
    return data[data["store"].isin(sorted(data["store"].unique())[:20])]


@pytest.fixture(scope="session")
def models(deliveries):
    return train_models(deliveries)
