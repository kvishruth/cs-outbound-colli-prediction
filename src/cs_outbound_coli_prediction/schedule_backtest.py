"""Backtest without knowing the schedule: the DC x date accuracy when the delivery lines of the last week are repeated.

The normal backtest splits the forecast over the actual delivery lines of the holdout weeks, as if the schedule were
known perfectly and never changed. Here every holdout week is forecast like a production run without a schedule file:
from the data up to 8 weeks before it, with the lines (store, DC, date) of the last known week repeated.
The accuracy with the real planned schedule lies between this and the normal backtest.
"""
import pandas as pd

from .backtest import bias, compare_quantities, split_train_holdout, usable_rows, wape
from .config import max_horizon
from .data import assumed_schedule
from .features import build_features
from .models import FallbackForecaster, Forecaster
from .pipeline import dc_totals, forecast_daily


def repeated_schedule_metrics(deliveries):
    """WAPE and bias per DC and delivery date of the chosen and the fallback model, over the holdout weeks."""
    train, holdout = split_train_holdout(usable_rows(build_features(deliveries)))
    models = {"main": Forecaster().fit(train), "fallback": FallbackForecaster()}
    fallback_only = {"main": FallbackForecaster(), "fallback": FallbackForecaster()}

    forecasts = {"main": [], "fallback": []}
    for week in sorted(holdout["Monday_of_week"].unique()):
        known = deliveries[deliveries["Monday_of_week"] <= week - pd.Timedelta(weeks=max_horizon)]
        schedule = assumed_schedule(known)
        for name, used in (("main", models), ("fallback", fallback_only)):
            _, daily = forecast_daily(known, schedule, used)
            in_week = (daily["delivery_date"] >= week) & (daily["delivery_date"] < week + pd.Timedelta(weeks=1))
            forecasts[name].append(dc_totals(daily[in_week]))

    actual = deliveries[deliveries["Monday_of_week"].isin(holdout["Monday_of_week"])]
    metrics = {}
    for name, parts in forecasts.items():
        both = compare_quantities(actual, pd.concat(parts), ["dc", "delivery_date"])
        metrics[f"holdout_wape_dc_date_repeated_schedule_{name}"] = wape(both["actual"], both["forecast"])
        metrics[f"holdout_bias_dc_date_repeated_schedule_{name}"] = bias(both["actual"], both["forecast"])
    return metrics
