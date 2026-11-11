"""Splitting the weekly forecast of a store over its delivery dates, and over the DCs that deliver on each date."""
import pandas as pd

from .data import monday_of


def weekday_shares(deliveries, until, weeks=26):
    """Average share of a store's weekly quantity delivered on each weekday, over the `weeks` weeks before `until`."""
    recent = _recent(deliveries, until, weeks)
    daily = recent.groupby(["store", "Monday_of_week", "delivery_date"], as_index=False)["delivery_quantity"].sum()
    daily["weekday"] = daily["delivery_date"].dt.day_of_week
    weekly_total = daily.groupby(["store", "Monday_of_week"])["delivery_quantity"].transform("sum")
    daily["share"] = daily["delivery_quantity"] / weekly_total
    return daily.groupby(["store", "weekday"], as_index=False)["share"].mean()


def split_by_delivery_date(weekly_forecast, schedule, shares):
    """Quantity per store and delivery date: the weekly forecast times the weekday share of each scheduled date.

    The shares of a week are rescaled to sum to one. A weekday without history gets an equal share.
    """
    dates = schedule[["store", "delivery_date"]].drop_duplicates().copy()
    dates["weekday"] = dates["delivery_date"].dt.day_of_week
    dates["Monday_of_week"] = monday_of(dates["delivery_date"])

    dates = dates.merge(shares, on=["store", "weekday"], how="left")
    store_week = dates.groupby(["store", "Monday_of_week"])
    dates["share"] = dates["share"].fillna(1 / store_week["delivery_date"].transform("count"))
    dates["share"] = dates["share"] / store_week["share"].transform("sum")

    daily = dates.merge(weekly_forecast, on=["store", "Monday_of_week"])
    daily["quantity"] = daily["forecast"] * daily["share"]
    return daily[["store", "delivery_date", "quantity"]]


def split_by_dc(store_daily, schedule, deliveries, until, weeks=26):
    """Quantity per store, DC and delivery date: the store's quantity of a date split over the DCs scheduled that date.

    The schedule (store, dc, delivery_date) says which DCs deliver to the store on the date. The weights come from the
    history quantity of each DC for that store: on that weekday if every DC has history on it, else in total.
    A DC without history gets the average weight of the other DCs of that store and date (equal if none has history).
    """
    lines = schedule[["store", "dc", "delivery_date"]].drop_duplicates().copy()
    lines["weekday"] = lines["delivery_date"].dt.day_of_week

    recent = _recent(deliveries, until, weeks).assign(weekday=lambda rows: rows["delivery_date"].dt.day_of_week)
    weekday_quantity = (recent.groupby(["store", "dc", "weekday"])["delivery_quantity"].sum()
                        .rename("weekday_quantity").reset_index())
    total_quantity = recent.groupby(["store", "dc"])["delivery_quantity"].sum().rename("total_quantity").reset_index()
    lines = lines.merge(weekday_quantity, on=["store", "dc", "weekday"], how="left")
    lines = lines.merge(total_quantity, on=["store", "dc"], how="left")

    # one scale for all DCs of a store and date: a weekday quantity and a total quantity must not be mixed
    date_keys = ["store", "delivery_date"]
    all_have_weekday = lines["weekday_quantity"].notna().groupby([lines["store"], lines["delivery_date"]]).transform("all")
    lines["weight"] = lines["weekday_quantity"].where(all_have_weekday, lines["total_quantity"])
    lines["weight"] = lines["weight"].fillna(lines.groupby(date_keys)["weight"].transform("mean")).fillna(1.0)
    lines["share"] = lines["weight"] / lines.groupby(date_keys)["weight"].transform("sum")

    daily = lines.merge(store_daily, on=["store", "delivery_date"])
    daily["quantity"] = daily["quantity"] * daily["share"]
    return daily[["store", "dc", "delivery_date", "quantity"]]


def _recent(deliveries, until, weeks):
    """Delivery rows of the `weeks` weeks before `until`."""
    in_window = (deliveries["Monday_of_week"] < until) & (deliveries["Monday_of_week"] >= until - pd.Timedelta(weeks=weeks))
    return deliveries[in_window]
