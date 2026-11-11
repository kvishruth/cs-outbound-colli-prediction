"""Reading the delivery data and the delivery schedule."""
import pandas as pd

from .config import max_horizon


def load_deliveries(path):
    """Delivery rows with parsed dates (columns as in data/input/store_deliveries.csv)."""
    deliveries = pd.read_csv(path)
    deliveries["Monday_of_week"] = pd.to_datetime(deliveries["Monday_of_week"], utc=True)
    deliveries["delivery_date"] = pd.to_datetime(deliveries["delivery_date"], utc=True)
    return deliveries


def load_schedule(path):
    """Planned deliveries of the coming weeks (columns: store, dc, delivery_date)."""
    schedule = pd.read_csv(path)
    missing = {"store", "dc", "delivery_date"} - set(schedule.columns)
    if missing:
        raise ValueError(f"the schedule needs the columns store, dc and delivery_date, missing: {sorted(missing)}")
    schedule["delivery_date"] = pd.to_datetime(schedule["delivery_date"], utc=True)
    return schedule[["store", "dc", "delivery_date"]].drop_duplicates()


def assumed_schedule(deliveries, horizon=max_horizon):
    """Stand-in when no schedule is given: repeat the deliveries (store, DC, date) of the last week."""
    last_week = deliveries["Monday_of_week"].max()
    last_lines = deliveries.loc[deliveries["Monday_of_week"] == last_week, ["store", "dc", "delivery_date"]].drop_duplicates()
    return pd.concat(
        [last_lines.assign(delivery_date=last_lines["delivery_date"] + pd.Timedelta(weeks=weeks_ahead))
         for weeks_ahead in range(1, horizon + 1)],
        ignore_index=True)


def monday_of(dates):
    """The Monday of the week of each date."""
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.day_of_week, unit="D")
