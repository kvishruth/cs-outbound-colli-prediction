"""Backtest: train on the past, score the last weeks. The same numbers as the notebook's holdout."""
import numpy as np
import pandas as pd

from .config import holdout_weeks, max_horizon, target
from .features import build_features
from .models import FallbackForecaster, Forecaster
from .split import split_by_dc, split_by_delivery_date, weekday_shares


def wape(actual, predicted):
    """Sum of absolute errors divided by the sum of the actuals."""
    return float(np.abs(actual - predicted).sum() / actual.sum())


def bias(actual, predicted):
    """Positive means over-forecast."""
    return float((predicted.sum() - actual.sum()) / actual.sum())


def usable_rows(features):
    """Rows with a known target and enough history."""
    return features.dropna(subset=[target, "qty_last_1", "mean_13"])


def split_train_holdout(rows):
    """Holdout = the last weeks. Training stops max_horizon weeks before them, so no training
    target is newer than the first forecast origin of the holdout."""
    weeks = rows["Monday_of_week"].drop_duplicates().sort_values()
    holdout_start = weeks.iloc[-holdout_weeks]
    train = rows[rows["Monday_of_week"] <= holdout_start - pd.Timedelta(weeks=max_horizon)]
    holdout = rows[rows["Monday_of_week"] >= holdout_start]
    return train, holdout


def backtest(deliveries):
    """Train on the past and score the last weeks.

    Returns the metrics (a flat dict: WAPE and bias of the chosen and the fallback model, WAPE per horizon, WAPE and
    bias per store and date and per DC and date) and tables with WAPE and bias per store, per DC and per weekday.
    """
    train, holdout = split_train_holdout(usable_rows(build_features(deliveries)))
    predictions = {"main": Forecaster().fit(train).predict(holdout),
                   "fallback": FallbackForecaster().predict(holdout)}
    actual = holdout[target]

    metrics = {}
    for name, predicted in predictions.items():
        metrics[f"holdout_wape_{name}"] = wape(actual, predicted)
        metrics[f"holdout_bias_{name}"] = bias(actual, predicted)
    for horizon in range(1, max_horizon + 1):
        in_horizon = (holdout["horizon"] == horizon).to_numpy()
        metrics[f"holdout_wape_main_horizon_{horizon}"] = wape(actual[in_horizon], predictions["main"][in_horizon])

    until = train["Monday_of_week"].max() + pd.Timedelta(weeks=1)     # the history known at the first forecast origin
    metrics |= daily_metrics(deliveries, holdout, predictions["main"], until)
    return metrics, breakdown_tables(deliveries, holdout, predictions, until)


def holdout_daily_forecasts(deliveries, holdout, predicted, until):
    """Forecasts per store and date, and per store, DC and date, for the furthest horizon of the holdout weeks.

    The weeks are split as in the forecast: over the schedule of those weeks (known in advance) with only the
    history before `until`. Returns both tables and the actual deliveries of the holdout weeks.
    """
    furthest = (holdout["horizon"] == max_horizon).to_numpy()
    weekly = holdout.loc[furthest, ["store", "Monday_of_week"]].assign(forecast=predicted[furthest])
    in_holdout = deliveries["Monday_of_week"].between(holdout["Monday_of_week"].min(), holdout["Monday_of_week"].max())
    actual = deliveries[in_holdout]
    schedule = actual[["store", "dc", "delivery_date"]].drop_duplicates()

    store_daily = split_by_delivery_date(weekly, schedule, weekday_shares(deliveries, until))
    return store_daily, split_by_dc(store_daily, schedule, deliveries, until), actual


def compare_quantities(actual, forecast, columns):
    """Actual and forecast quantity per `columns` (for example dc and delivery_date); 0 where one of them is missing."""
    both = pd.concat([actual.groupby(columns)["delivery_quantity"].sum().rename("actual"),
                      forecast.groupby(columns)["quantity"].sum().rename("forecast")], axis=1)
    return both.fillna(0).reset_index()


def daily_metrics(deliveries, holdout, predicted, until):
    """WAPE and bias per store and delivery date, and per DC and delivery date, for the furthest horizon."""
    store_daily, dc_daily, actual = holdout_daily_forecasts(deliveries, holdout, predicted, until)
    metrics = {}
    for name, forecast, columns in [("store_date", store_daily, ["store", "delivery_date"]),
                                    ("dc_date", dc_daily, ["dc", "delivery_date"])]:
        both = compare_quantities(actual, forecast, columns)
        metrics[f"holdout_wape_{name}"] = wape(both["actual"], both["forecast"])
        metrics[f"holdout_bias_{name}"] = bias(both["actual"], both["forecast"])
    return metrics


def breakdown_tables(deliveries, holdout, predictions, until):
    """WAPE and bias of every model per store (store and week), per DC and per weekday (DC and delivery date).

    Furthest horizon. The worst stores and DCs come first. `actual_colli` shows how much quantity a row stands for.
    """
    furthest = (holdout["horizon"] == max_horizon).to_numpy()
    store_weeks, dc_dates = {}, {}
    for name, predicted in predictions.items():
        store_weeks[name] = holdout.loc[furthest, ["store"]].assign(
            actual=holdout.loc[furthest, target], forecast=predicted[furthest])
        _, dc_daily, actual = holdout_daily_forecasts(deliveries, holdout, predicted, until)
        dc_dates[name] = compare_quantities(actual, dc_daily, ["dc", "delivery_date"])
        dc_dates[name]["weekday"] = dc_dates[name]["delivery_date"].dt.day_name()

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_weekday = _scores_by(dc_dates, "weekday")
    return {
        "store": _scores_by(store_weeks, "store").sort_values("main_wape", ascending=False),
        "dc": _scores_by(dc_dates, "dc").sort_values("main_wape", ascending=False),
        "weekday": by_weekday.reindex([day for day in weekdays if day in by_weekday.index]),
    }


def _scores_by(frames, by):
    """One row per value of `by`: WAPE and bias of each model (frames hold actual and forecast), and the actual quantity."""
    scores = []
    for name, frame in frames.items():
        total = frame.assign(error=(frame["actual"] - frame["forecast"]).abs()).groupby(by)[["actual", "forecast", "error"]].sum()
        scores.append(pd.DataFrame({f"{name}_wape": total["error"] / total["actual"],
                                    f"{name}_bias": (total["forecast"] - total["actual"]) / total["actual"]}))
    return pd.concat(scores, axis=1).assign(actual_colli=total["actual"])
