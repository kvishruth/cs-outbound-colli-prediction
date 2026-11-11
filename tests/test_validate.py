import pandas as pd
import pytest

from cs_outbound_coli_prediction.cli import main
from cs_outbound_coli_prediction.config import max_horizon
from cs_outbound_coli_prediction.data import assumed_schedule
from cs_outbound_coli_prediction.validate import (
    validate_age,
    validate_deliveries,
    validate_schedule,
)


def test_valid_data_passes(deliveries):
    validate_deliveries(deliveries)


def test_duplicated_rows_are_reported(deliveries):
    with pytest.raises(ValueError, match="duplicated rows"):
        validate_deliveries(pd.concat([deliveries, deliveries.iloc[:1]]))


def test_negative_quantities_are_reported(deliveries):
    bad = deliveries.copy()
    bad.iloc[0, bad.columns.get_loc("delivery_quantity")] = -5
    with pytest.raises(ValueError, match="negative"):
        validate_deliveries(bad)


def test_incomplete_last_week_is_reported(deliveries):
    last_week = deliveries["Monday_of_week"].max()
    cut_off = (deliveries["Monday_of_week"] == last_week) & (deliveries["delivery_date"].dt.day_of_week >= 3)
    with pytest.raises(ValueError, match="before its Saturday"):
        validate_deliveries(deliveries[~cut_off])


def test_schedule_must_be_after_the_data(deliveries):
    last_week = deliveries["Monday_of_week"].max()
    past = pd.DataFrame({"store": ["S0001"], "dc": ["DC01"], "delivery_date": [last_week]})
    with pytest.raises(ValueError, match="before"):
        validate_schedule(past, deliveries)


def test_week_missing_inside_the_history_is_reported(deliveries):
    missing_week = deliveries["Monday_of_week"].drop_duplicates().sort_values().iloc[30]
    with pytest.raises(ValueError, match="1 weeks without any delivery"):
        validate_deliveries(deliveries[deliveries["Monday_of_week"] != missing_week])


def test_repeated_store_dc_and_date_is_reported(deliveries):
    repeated = deliveries.iloc[:1].assign(delivery_quantity=deliveries.iloc[0]["delivery_quantity"] + 1)
    with pytest.raises(ValueError, match="repeat a store, DC and delivery date"):
        validate_deliveries(pd.concat([deliveries, repeated]))


def test_assumed_schedule_is_valid(deliveries):
    validate_schedule(assumed_schedule(deliveries), deliveries)


def test_empty_schedule_is_reported(deliveries):
    with pytest.raises(ValueError, match="schedule is empty"):
        validate_schedule(assumed_schedule(deliveries).iloc[0:0], deliveries)


def test_schedule_without_a_dc_is_reported(deliveries):
    schedule = assumed_schedule(deliveries)
    schedule.loc[schedule.index[0], "dc"] = None
    with pytest.raises(ValueError, match="missing stores, DCs or dates"):
        validate_schedule(schedule, deliveries)


def test_schedule_beyond_the_horizon_is_reported(deliveries):
    schedule = assumed_schedule(deliveries)
    schedule["delivery_date"] += pd.Timedelta(weeks=max_horizon)
    with pytest.raises(ValueError, match="beyond"):
        validate_schedule(schedule, deliveries)


def test_schedule_with_an_unknown_store_is_reported(deliveries):
    schedule = assumed_schedule(deliveries)
    schedule.loc[schedule.index[0], "store"] = "S_NEW"
    with pytest.raises(ValueError, match="S_NEW"):
        validate_schedule(schedule, deliveries)


def test_old_data_is_reported(deliveries):
    end_of_data = deliveries["Monday_of_week"].max() + pd.Timedelta(weeks=1)
    validate_age(deliveries, end_of_data + pd.Timedelta(days=10), max_age_days=10)
    with pytest.raises(ValueError, match="11 days ago"):
        validate_age(deliveries, end_of_data + pd.Timedelta(days=11), max_age_days=10)


def test_forecast_command_refuses_old_data(tmp_path):
    with pytest.raises(ValueError, match="days ago"):
        main(["forecast", "--models-dir", str(tmp_path), "--output-dir", str(tmp_path)])
