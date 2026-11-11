"""Checks on the inputs. A problem that would silently give a wrong forecast raises an error."""
import pandas as pd

from .config import active_weeks, max_horizon, min_last_week_ratio

required_columns = ["store", "country_store", "dc", "delivery_date", "Monday_of_week", "delivery_quantity", "sales_index"]


def validate_deliveries(deliveries):
    """Raise a ValueError that lists every problem found in the delivery data."""
    missing_columns = [column for column in required_columns if column not in deliveries.columns]
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")

    problems = (_value_problems(deliveries) + _week_problems(deliveries) + _missing_week_problems(deliveries)
                + _last_week_problems(deliveries))
    if problems:
        raise ValueError("invalid delivery data:\n- " + "\n- ".join(problems))


def validate_schedule(schedule, deliveries):
    """The schedule must hold dates of the weeks to forecast, a DC on every line, and only stores that are forecast."""
    first_new_week = deliveries["Monday_of_week"].max() + pd.Timedelta(weeks=1)
    after_last_new_week = first_new_week + pd.Timedelta(weeks=max_horizon)
    problems = []
    if schedule.empty:
        problems.append("the schedule is empty")
    if schedule[["store", "dc", "delivery_date"]].isna().any().any():
        problems.append("the schedule has missing stores, DCs or dates")
    else:
        if (schedule["delivery_date"] < first_new_week).any():
            problems.append(f"the schedule has dates before {first_new_week.date()}, the first week to forecast")
        if (schedule["delivery_date"] >= after_last_new_week).any():
            problems.append(f"the schedule has dates from {after_last_new_week.date()}, beyond the {max_horizon} weeks to forecast")
    unknown = sorted(set(schedule["store"].dropna()) - _active_stores(deliveries))
    if unknown:
        problems.append(f"the schedule has {len(unknown)} stores without a delivery in the last {active_weeks} weeks: {unknown[:10]}")
    if problems:
        raise ValueError("invalid schedule:\n- " + "\n- ".join(problems))


def validate_age(deliveries, today, max_age_days):
    """The delivery data must end at most `max_age_days` days before `today`."""
    end_of_data = deliveries["Monday_of_week"].max() + pd.Timedelta(weeks=1)
    age_days = (today - end_of_data).days
    if age_days > max_age_days:
        raise ValueError(f"the delivery data ends on {end_of_data.date()}, {age_days} days ago (at most {max_age_days} allowed)")


def _active_stores(deliveries):
    """Stores with a delivery in the last active_weeks weeks: the stores that are forecast."""
    recent = deliveries["Monday_of_week"] > deliveries["Monday_of_week"].max() - pd.Timedelta(weeks=active_weeks)
    return set(deliveries.loc[recent, "store"])


def _value_problems(deliveries):
    problems = []
    for column, count in deliveries[required_columns].isna().sum().items():
        if count and column != "sales_index":     # sales are allowed to be missing, the model handles it
            problems.append(f"{count} missing values in {column}")
    if (deliveries["delivery_quantity"] < 0).any():
        problems.append("negative delivery quantities")
    if deliveries.duplicated().any():
        problems.append(f"{deliveries.duplicated().sum()} duplicated rows")
    repeated_lines = deliveries.duplicated(["store", "dc", "delivery_date"]).sum()
    if repeated_lines:
        problems.append(f"{repeated_lines} rows repeat a store, DC and delivery date")
    return problems


def _week_problems(deliveries):
    problems = []
    if (deliveries["Monday_of_week"].dt.day_of_week != 0).any():
        problems.append("Monday_of_week holds dates that are not a Monday")
    days_into_week = (deliveries["delivery_date"] - deliveries["Monday_of_week"]).dt.days
    if not days_into_week.between(0, 6).all():
        problems.append("delivery_date lies outside the week of Monday_of_week")
    if (deliveries.groupby("store")["country_store"].nunique() > 1).any():
        problems.append("a store has more than one country_store")
    return problems


def _missing_week_problems(deliveries):
    """A week without any delivery inside the history would count as zero for every store."""
    first, last = deliveries["Monday_of_week"].min(), deliveries["Monday_of_week"].max()
    missing = pd.date_range(first, last, freq="7D").difference(deliveries["Monday_of_week"].unique())
    if len(missing):
        return [f"{len(missing)} weeks without any delivery: {[week.date().isoformat() for week in missing[:5]]}"]
    return []


def _last_week_problems(deliveries):
    """The forecast assumes the last week is complete: it should reach its Saturday and have a normal size."""
    problems = []
    weekly_quantity = deliveries.groupby("Monday_of_week")["delivery_quantity"].sum()
    last_week = weekly_quantity.index.max()

    last_delivery = deliveries.loc[deliveries["Monday_of_week"] == last_week, "delivery_date"].max()
    if last_delivery.day_of_week < 5:
        problems.append(f"the last week ({last_week.date()}) ends on {last_delivery.date()}, before its Saturday: incomplete?")

    usual = weekly_quantity.iloc[-9:-1].median()
    if weekly_quantity.iloc[-1] < min_last_week_ratio * usual:
        problems.append(f"the last week ({last_week.date()}) holds under {min_last_week_ratio:.0%} of the usual quantity: incomplete?")
    return problems
