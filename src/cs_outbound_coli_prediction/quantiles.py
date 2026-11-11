"""Quantile forecasts per DC and delivery date: the point forecast times a factor learnt from past forecast errors.

Quantiles of a sum are not the sum of the store quantiles, so the factors are fitted on the DC totals themselves.
"""
import numpy as np
import pandas as pd

from .backtest import compare_quantities, holdout_daily_forecasts, usable_rows
from .config import holdout_weeks, max_horizon, quantile_levels
from .data import monday_of
from .features import build_features, in_christmas_period
from .models import FallbackForecaster, Forecaster


def rolling_dc_forecasts(deliveries, fold_count=3):
    """Out-of-sample forecasts per DC and delivery date (furthest horizon) of consecutive holdouts, newest first.

    Fold 0 is the holdout of the backtest, fold 1 the holdout_weeks weeks before it, and so on.
    Columns: fold, dc, delivery_date, actual, forecast (chosen model) and fallback (fallback model).
    """
    rows = usable_rows(build_features(deliveries))
    weeks = rows["Monday_of_week"].drop_duplicates().sort_values().reset_index(drop=True)

    folds = []
    for fold in range(fold_count):
        end = len(weeks) - fold * holdout_weeks
        holdout = rows[rows["Monday_of_week"].isin(weeks[end - holdout_weeks:end])]
        train = rows[rows["Monday_of_week"] <= holdout["Monday_of_week"].min() - pd.Timedelta(weeks=max_horizon)]
        until = train["Monday_of_week"].max() + pd.Timedelta(weeks=1)

        main_model, fallback_model = Forecaster().fit(train), FallbackForecaster()
        _, main_daily, actual = holdout_daily_forecasts(deliveries, holdout, main_model.predict(holdout), until)
        _, fallback_daily, _ = holdout_daily_forecasts(deliveries, holdout, fallback_model.predict(holdout), until)

        keys = ["dc", "delivery_date"]
        main = compare_quantities(actual, main_daily, keys)
        fallback = compare_quantities(actual, fallback_daily, keys).rename(columns={"forecast": "fallback"})
        both = main.merge(fallback[keys + ["fallback"]], on=keys, how="outer").fillna(0)   # every actual DC-day is in both
        folds.append(both.assign(fold=fold))
    return pd.concat(folds, ignore_index=True)


def fit_factors(errors, min_days=30):
    """Per DC: the quantiles of actual / forecast over its past DC-days. DCs with few days use the 'all' row."""
    errors = errors[errors["forecast"] > 0]
    ratio = errors["actual"] / errors["forecast"]
    factors = ratio.groupby(errors["dc"]).quantile(list(quantile_levels)).unstack()
    factors = factors[ratio.groupby(errors["dc"]).size() >= min_days]
    factors.loc["all"] = ratio.quantile(list(quantile_levels))
    factors.columns = [f"p{round(level * 100)}" for level in quantile_levels]
    return factors


def add_quantiles(dc_forecast, factors):
    """Add a column per quantile (quantity_p80, quantity_p90, ...) to a table with dc and quantity."""
    per_dc = factors.reindex(dc_forecast["dc"]).fillna(factors.loc["all"]).set_axis(dc_forecast.index)
    for column in factors.columns:
        dc_forecast[f"quantity_{column}"] = dc_forecast["quantity"] * per_dc[column]
    return dc_forecast


def pinball_loss(actual, forecast, level):
    """Pinball loss of a quantile forecast, relative to the total actual quantity."""
    error = actual - forecast
    return float(np.maximum(level * error, (level - 1) * error).sum() / actual.sum())


def _newest_with_quantiles(calibration):
    """The newest fold with the quantile columns, from factors fitted on the older folds."""
    newest = calibration[calibration["fold"] == 0]
    factors = fit_factors(calibration[calibration["fold"] > 0])
    return add_quantiles(newest.rename(columns={"forecast": "quantity"}), factors)


def quantile_metrics(calibration):
    """Fit the factors on the older folds and score the newest fold: coverage and pinball loss, against the point forecast."""
    scored = _newest_with_quantiles(calibration)

    metrics = {}
    for level in quantile_levels:
        name = f"p{round(level * 100)}"
        metrics[f"holdout_coverage_{name}"] = float((scored["actual"] <= scored[f"quantity_{name}"]).mean())
        metrics[f"holdout_pinball_{name}"] = pinball_loss(scored["actual"], scored[f"quantity_{name}"], level)
        metrics[f"holdout_pinball_{name}_point_forecast"] = pinball_loss(scored["actual"], scored["quantity"], level)
    return metrics


def dc_quantile_table(calibration):
    """Coverage and pinball loss of the quantiles per DC, on the newest fold (factors fitted on the older folds)."""
    scored = _newest_with_quantiles(calibration)
    table = {}
    for dc, rows in scored.groupby("dc"):
        table[dc] = {}
        for level in quantile_levels:
            name = f"p{round(level * 100)}"
            table[dc][f"{name}_coverage"] = (rows["actual"] <= rows[f"quantity_{name}"]).mean()
            table[dc][f"{name}_pinball"] = pinball_loss(rows["actual"], rows[f"quantity_{name}"], level)
    return pd.DataFrame(table).T.rename_axis("dc")


def day_type_table(calibration):
    """WAPE, bias and quantile coverage for all days, peak days and holiday weeks, over all folds.

    Peak days are the 10% of DC-days with the most colli of each DC and fold. They are chosen by the actual quantity,
    so a point forecast is expected to be too low there. The factors of each fold come from the other folds.
    """
    rows = calibration.copy()
    for fold in rows["fold"].unique():
        factors = fit_factors(rows[rows["fold"] != fold])
        in_fold = rows["fold"] == fold
        per_dc = factors.reindex(rows.loc[in_fold, "dc"]).fillna(factors.loc["all"])
        for column in factors.columns:
            rows.loc[in_fold, column] = rows.loc[in_fold, "forecast"].to_numpy() * per_dc[column].to_numpy()

    peak = rows["actual"] >= rows.groupby(["fold", "dc"])["actual"].transform(lambda actual: actual.quantile(0.9))
    holiday = in_christmas_period(monday_of(rows["delivery_date"]))
    day_types = {"all days": peak | ~peak, "peak days (busiest 10% by actual)": peak, "other days": ~peak,
                 "holiday weeks (Christmas + 4)": holiday, "other weeks": ~holiday}

    table = {}
    for name, selected in day_types.items():
        part = rows[selected]
        table[name] = {"dc_days": len(part), "actual_colli": part["actual"].sum()}
        for model, column in (("main", "forecast"), ("fallback", "fallback")):
            error = (part["actual"] - part[column]).abs().sum()
            table[name][f"{model}_wape"] = error / part["actual"].sum()
            table[name][f"{model}_bias"] = (part[column].sum() - part["actual"].sum()) / part["actual"].sum()
        for column in ("p80", "p90"):
            table[name][f"main_{column}_coverage"] = (part["actual"] <= part[column]).mean()
    return pd.DataFrame(table).T.rename_axis("day_type")
