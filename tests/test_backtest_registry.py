import json

import pandas as pd
import pytest

from cs_outbound_coli_prediction.backtest import (
    backtest,
    bias,
    split_train_holdout,
    usable_rows,
    wape,
)
from cs_outbound_coli_prediction.config import holdout_weeks, input_path, max_horizon
from cs_outbound_coli_prediction.features import build_features
from cs_outbound_coli_prediction.registry import load_current, promote, save_version


def test_wape_and_bias():
    actual = pd.Series([100.0, 100.0])
    predicted = pd.Series([110.0, 80.0])
    assert wape(actual, predicted) == pytest.approx(0.15)
    assert bias(actual, predicted) == pytest.approx(-0.05)


def test_holdout_split_keeps_the_gap(deliveries):
    train, holdout = split_train_holdout(usable_rows(build_features(deliveries)))
    assert holdout["Monday_of_week"].nunique() == holdout_weeks
    assert train["Monday_of_week"].max() == holdout["Monday_of_week"].min() - pd.Timedelta(weeks=max_horizon)


@pytest.fixture(scope="module")
def backtest_result(deliveries):
    return backtest(deliveries)


def test_backtest_reports_both_models_and_every_horizon(backtest_result):
    metrics, _ = backtest_result
    assert {"holdout_wape_main", "holdout_wape_fallback"} <= metrics.keys()
    assert sum(key.startswith("holdout_wape_main_horizon_") for key in metrics) == max_horizon
    assert {"holdout_wape_store_date", "holdout_wape_dc_date"} <= metrics.keys()


def test_breakdown_tables_add_up_to_the_pooled_metrics(backtest_result, deliveries):
    metrics, tables = backtest_result
    assert set(tables) == {"store", "dc", "weekday"}
    assert len(tables["store"]) == deliveries["store"].nunique()
    assert {"main_wape", "main_bias", "fallback_wape", "fallback_bias", "actual_colli"} <= set(tables["store"].columns)

    def pooled(table, column):    # the quantity-weighted mean of the per-row WAPE is the pooled WAPE
        return (table[column] * table["actual_colli"]).sum() / table["actual_colli"].sum()

    assert pooled(tables["store"], "main_wape") == pytest.approx(metrics[f"holdout_wape_main_horizon_{max_horizon}"])
    assert pooled(tables["dc"], "main_wape") == pytest.approx(metrics["holdout_wape_dc_date"])
    assert pooled(tables["weekday"], "main_wape") == pytest.approx(metrics["holdout_wape_dc_date"])
    assert pooled(tables["dc"], "main_bias") == pytest.approx(metrics["holdout_bias_dc_date"])
    assert tables["store"]["main_wape"].is_monotonic_decreasing                 # worst first


def test_saved_version_can_be_loaded_and_rolled_back(deliveries, models, backtest_result, tmp_path):
    first = save_version(models, input_path, deliveries, {"holdout_wape_main": 0.1}, tmp_path, backtest_result[1])
    promote(first, tmp_path)
    loaded, metadata = load_current(tmp_path)

    assert set(loaded) == {"main", "fallback"}
    assert metadata["version"] == first
    assert {"git_commit", "data", "features", "metrics", "libraries"} <= metadata.keys()
    assert len(metadata["data"]["sha256"]) == 64
    assert json.loads((tmp_path / first / "metadata.json").read_text())["version"] == first
    assert sorted(path.name for path in (tmp_path / first / "backtest").iterdir()) == ["by_dc.csv", "by_store.csv", "by_weekday.csv"]


def test_models_are_not_loaded_with_other_library_versions(deliveries, models, tmp_path):
    version = save_version(models, input_path, deliveries, {}, tmp_path)
    promote(version, tmp_path)
    load_current(tmp_path)                                      # same libraries: loads

    metadata_file = tmp_path / version / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["libraries"]["lightgbm"] = "0.0.1"
    metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="lightgbm: trained with 0.0.1"):
        load_current(tmp_path)
