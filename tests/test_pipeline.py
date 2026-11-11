import numpy as np
import pandas as pd
import pytest

from cs_outbound_coli_prediction.config import feature_columns, max_horizon
from cs_outbound_coli_prediction.data import assumed_schedule, monday_of
from cs_outbound_coli_prediction.features import build_features
from cs_outbound_coli_prediction.pipeline import (
    dc_totals,
    forecast_daily,
    forecast_weekly,
)

# lags, rolling statistics and sales: built from past weeks only (the schedule and calendar are known ahead)
history_columns = [c for c in feature_columns if c.startswith(("qty_", "mean_", "std_", "sales_"))]


def test_features_only_use_weeks_known_at_the_forecast_origin(deliveries):
    """Changing everything after an origin must not change the features of rows forecast from that origin."""
    origin = deliveries["Monday_of_week"].drop_duplicates().sort_values().iloc[70]
    changed = deliveries.copy()
    later = changed["Monday_of_week"] > origin
    changed.loc[later, ["delivery_quantity", "sales_index"]] *= 1000

    key = ["store", "Monday_of_week", "horizon"]
    both = build_features(deliveries).merge(build_features(changed), on=key, suffixes=("_a", "_b"))
    for horizon in range(1, max_horizon + 1):
        known = both[(both["horizon"] == horizon)
                     & (both["Monday_of_week"] <= origin + pd.Timedelta(weeks=horizon))]
        for column in history_columns:
            pd.testing.assert_series_equal(known[f"{column}_a"], known[f"{column}_b"], check_names=False)


def test_forecast_has_one_row_per_store_and_week(deliveries, models):
    schedule = assumed_schedule(deliveries)
    weekly, daily = forecast_daily(deliveries, schedule, models)

    assert not weekly.duplicated(["store", "Monday_of_week"]).any()
    assert weekly.groupby("store").size().eq(max_horizon).all()
    assert (weekly["forecast"] >= 0).all()
    # the daily quantities of a store and week add up to its weekly forecast
    daily["Monday_of_week"] = monday_of(daily["delivery_date"])
    totals = daily.groupby(["store", "Monday_of_week"])["quantity"].sum().rename("daily")
    check = weekly.set_index(["store", "Monday_of_week"]).join(totals, how="inner")
    np.testing.assert_allclose(check["forecast"], check["daily"])
    # the DC totals add up to the same quantity
    assert dc_totals(daily)["quantity"].sum() == pytest.approx(daily["quantity"].sum())


def test_fallback_model_is_used_when_the_chosen_model_fails(deliveries, models):
    class Broken:
        def predict(self, table):
            raise RuntimeError("model unavailable")

    weekly = forecast_weekly(deliveries, assumed_schedule(deliveries), {**models, "main": Broken()})
    assert (weekly["model"] == "fallback").all()
    assert len(weekly) > 0
