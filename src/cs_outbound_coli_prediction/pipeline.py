"""Training and forecasting. The chosen model is used first, the fallback model where it fails."""
import logging

import numpy as np
import pandas as pd

from .backtest import usable_rows
from .config import active_weeks
from .data import monday_of
from .features import build_features
from .models import FallbackForecaster, Forecaster
from .split import split_by_dc, split_by_delivery_date, weekday_shares

logger = logging.getLogger(__name__)


def train_models(deliveries):
    """Fit the chosen model on all history. No gap is needed: every target up to now is known."""
    train = usable_rows(build_features(deliveries))
    return {"main": Forecaster().fit(train), "fallback": FallbackForecaster()}


def forecast_weekly(deliveries, schedule, models):
    """Forecast per store for the weeks of `schedule` (up to max_horizon weeks after the last week of `deliveries`).

    `deliveries` must end with a complete week.
    """
    last_week = deliveries["Monday_of_week"].max()
    recent = deliveries["Monday_of_week"] > last_week - pd.Timedelta(weeks=active_weeks)
    stores = deliveries.loc[recent, ["store", "country_store"]].drop_duplicates()

    features = build_features(pd.concat([deliveries, _future_rows(schedule, stores)], ignore_index=True))
    features = features.merge(stores[["store"]], on="store")

    # Week last_week + k is forecast with its horizon-k row, whose features only use weeks up to last_week.
    weeks_ahead = (features["Monday_of_week"] - last_week) // pd.Timedelta(weeks=1)
    rows = features[features["horizon"] == weeks_ahead].copy()
    rows["forecast"], rows["model"] = _predict(rows, models)

    missing = rows["forecast"].isna()
    if missing.any():
        skipped = sorted(rows.loc[missing, "store"].unique())
        logger.warning("no forecast for %d stores with too little history: %s", len(skipped), skipped[:10])
    return rows.loc[~missing, ["store", "Monday_of_week", "forecast", "model"]].reset_index(drop=True)


def forecast_daily(deliveries, schedule, models):
    """The weekly forecast per store, and its split over the delivery dates and the DCs of `schedule`."""
    weekly_forecast = forecast_weekly(deliveries, schedule, models)
    until = deliveries["Monday_of_week"].max() + pd.Timedelta(weeks=1)     # the history known at the forecast origin
    store_daily = split_by_delivery_date(weekly_forecast, schedule, weekday_shares(deliveries, until))
    daily_forecast = split_by_dc(store_daily, schedule, deliveries, until)
    return weekly_forecast, daily_forecast


def dc_totals(daily_forecast):
    """Quantity per DC and delivery date: the sum over the stores it serves."""
    return daily_forecast.groupby(["dc", "delivery_date"], as_index=False)["quantity"].sum()


def _future_rows(schedule, stores):
    """One placeholder row per planned delivery date, so the number of delivery days of the future weeks is known."""
    future = schedule[["store", "delivery_date"]].drop_duplicates().copy()
    future["Monday_of_week"] = monday_of(future["delivery_date"])
    return future.merge(stores, on="store").assign(delivery_quantity=0.0)


def _predict(rows, models):
    """Chosen model first; the fallback model for rows where it fails or gives no valid number."""
    try:
        forecast = pd.Series(models["main"].predict(rows), index=rows.index)
    except Exception:  # any failure of the chosen model must not stop the forecast
        logger.exception("chosen model failed, using the fallback model for all rows")
        forecast = pd.Series(np.nan, index=rows.index)
    model = pd.Series("main", index=rows.index)

    invalid = ~np.isfinite(forecast)
    if invalid.any():
        forecast[invalid] = models["fallback"].predict(rows[invalid])
        model[invalid] = "fallback"
    return forecast, model


def unscheduled_weeks(weekly_forecast, schedule):
    """Number of weekly forecasts without any scheduled delivery date: they have nothing to be split over and are lost."""
    scheduled = schedule.assign(Monday_of_week=monday_of(schedule["delivery_date"]))[["store", "Monday_of_week"]]
    checked = weekly_forecast.merge(scheduled.drop_duplicates(), how="left", indicator=True)
    return int((checked["_merge"] == "left_only").sum())
