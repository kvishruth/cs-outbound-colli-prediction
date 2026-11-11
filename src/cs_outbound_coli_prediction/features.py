"""Weekly store features, one copy of every store-week for each forecast horizon.

A row with horizon h forecasts week t from the data up to week t - h, so every lag and rolling
feature is shifted by h. Only the number of delivery days of week t is used unshifted, because
the delivery schedule is planned ahead.
"""
import numpy as np
import pandas as pd

from .config import christmas_weeks_after, max_horizon, recent_weeks_count, target

store_week = ["store", "Monday_of_week"]


def build_features(deliveries):
    weekly = _add_calendar(_weekly_table(deliveries))
    return pd.concat([_add_horizon_features(weekly, horizon) for horizon in range(1, max_horizon + 1)],
                     ignore_index=True)


def _weekly_table(deliveries):
    """One row per store and week. A week without deliveries counts as zero from the store's first delivery on."""
    weekly = deliveries.groupby(store_week, as_index=False).agg(
        **{target: ("delivery_quantity", "sum"), "sales_index": ("sales_index", "mean"),
           "n_days": ("delivery_date", "nunique")})

    all_weeks = pd.date_range(weekly["Monday_of_week"].min(), weekly["Monday_of_week"].max(), freq="7D")
    grid = pd.MultiIndex.from_product([weekly["store"].unique(), all_weeks], names=store_week)
    table = weekly.set_index(store_week).reindex(grid).reset_index().sort_values(store_week, ignore_index=True)

    first_week = table["store"].map(weekly.groupby("store")["Monday_of_week"].min())
    started = table["Monday_of_week"] >= first_week
    table.loc[started, [target, "n_days"]] = table.loc[started, [target, "n_days"]].fillna(0.0)

    countries = deliveries[["store", "country_store"]].drop_duplicates()
    if countries["store"].duplicated().any():
        raise ValueError("a store has more than one country_store")
    return table.merge(countries, on="store")


def _add_calendar(table):
    """ISO week features, since weeks start on Monday, and the weeks before Christmas."""
    iso = table["Monday_of_week"].dt.isocalendar()
    iso_year, iso_week = iso["year"].astype(int), iso["week"].astype(int)
    christmas_week = (pd.to_datetime({"year": iso_year, "month": 12, "day": 25})
                      .dt.isocalendar()["week"].astype(int))

    table["week_of_year"] = iso_week
    table["month"] = table["Monday_of_week"].dt.month
    table["week_sin"] = np.sin(2 * np.pi * iso_week / 52)
    table["week_cos"] = np.cos(2 * np.pi * iso_week / 52)
    table["xmas_week"] = (iso_week == christmas_week).astype(int)
    table["xmas_1w_before"] = (iso_week == christmas_week - 1).astype(int)
    table["xmas_2w_before"] = (iso_week == christmas_week - 2).astype(int)
    return table


def in_christmas_period(mondays):
    """True for the Christmas week (the ISO week with 25 December) and the weeks after it."""
    naive = mondays.dt.tz_localize(None)
    in_period = pd.Series(False, index=mondays.index)
    for years_back in (0, 1):                 # the period of this year's Christmas or of last year's
        christmas = pd.to_datetime({"year": naive.dt.year - years_back, "month": 12, "day": 25}).set_axis(mondays.index)
        christmas_monday = christmas - pd.to_timedelta(christmas.dt.day_of_week, unit="D")
        weeks_after = (naive - christmas_monday).dt.days // 7
        in_period |= weeks_after.between(0, christmas_weeks_after)
    return in_period


def _rolling(values, stores, window):
    """Rolling window per store, needing at least half of the window."""
    return values.groupby(stores).rolling(window, min_periods=max(2, window // 2))


def _add_horizon_features(weekly, horizon):
    table = weekly.copy()
    table["horizon"] = horizon
    by_store = table.groupby("store")

    # the most recent known weeks: week t - horizon, the week before, ...
    for lag in range(recent_weeks_count):
        table[f"qty_last_{lag + 1}"] = by_store[target].shift(horizon + lag)
        table[f"sales_last_{lag + 1}"] = by_store["sales_index"].shift(horizon + lag)
    table["qty_lag_52"] = by_store[target].shift(52)

    known_quantity = by_store[target].shift(horizon)
    for window in (4, 13, 52):
        rolling = _rolling(known_quantity, table["store"], window)
        table[f"mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        table[f"std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    known_sales = by_store["sales_index"].shift(horizon)
    table["sales_mean_4"] = _rolling(known_sales, table["store"], 4).mean().reset_index(level=0, drop=True)

    # delivery days of week t against the store's usual number of delivery days
    known_days = by_store["n_days"].shift(horizon)
    usual_days = _rolling(known_days, table["store"], 13).mean().reset_index(level=0, drop=True)
    table["days_ratio"] = (table["n_days"] / usual_days).where(usual_days > 0)
    return table
