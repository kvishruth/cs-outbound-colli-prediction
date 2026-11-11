import pandas as pd
import pytest

from cs_outbound_coli_prediction.data import load_schedule
from cs_outbound_coli_prediction.split import split_by_dc

monday = pd.Timestamp("2026-01-05", tz="UTC")


def delivery(store, dc, day, quantity, week=0):
    date = monday + pd.Timedelta(weeks=week, days=day)
    return {"store": store, "dc": dc, "delivery_date": date, "Monday_of_week": monday + pd.Timedelta(weeks=week),
            "delivery_quantity": quantity}


def test_store_quantity_is_split_over_the_scheduled_dcs_by_history():
    # last week: DC1 delivered 300 and DC2 delivered 100 on Mondays
    history = pd.DataFrame([delivery("S1", "DC1", 0, 300), delivery("S1", "DC2", 0, 100)])
    until = monday + pd.Timedelta(weeks=1)
    next_monday = monday + pd.Timedelta(weeks=1)
    store_daily = pd.DataFrame({"store": ["S1"], "delivery_date": [next_monday], "quantity": [200.0]})
    schedule = pd.DataFrame({"store": ["S1", "S1"], "dc": ["DC1", "DC2"], "delivery_date": [next_monday] * 2})

    daily = split_by_dc(store_daily, schedule, history, until).set_index("dc")["quantity"]
    assert daily["DC1"] == pytest.approx(150)
    assert daily["DC2"] == pytest.approx(50)


def test_only_the_dc_scheduled_that_date_gets_the_quantity_and_new_dcs_share_equally():
    history = pd.DataFrame([delivery("S1", "DC1", 0, 300)])
    until = monday + pd.Timedelta(weeks=1)
    date = monday + pd.Timedelta(weeks=1, days=2)            # a Wednesday, no history
    store_daily = pd.DataFrame({"store": ["S1"], "delivery_date": [date], "quantity": [100.0]})

    only_new_dc = pd.DataFrame({"store": ["S1"], "dc": ["DC9"], "delivery_date": [date]})
    assert split_by_dc(store_daily, only_new_dc, history, until)["quantity"].iloc[0] == pytest.approx(100)

    two_dcs = pd.DataFrame({"store": ["S1", "S1"], "dc": ["DC1", "DC9"], "delivery_date": [date, date]})
    daily = split_by_dc(store_daily, two_dcs, history, until)
    assert daily["quantity"].sum() == pytest.approx(100)


def test_weights_of_the_dcs_use_one_scale():
    # Monday: DC1 delivered 100, DC2 never. Tuesday: DC1 900, DC2 400. Totals: DC1 1000, DC2 400.
    history = pd.DataFrame([delivery("S1", "DC1", 0, 100), delivery("S1", "DC1", 1, 900), delivery("S1", "DC2", 1, 400)])
    until = monday + pd.Timedelta(weeks=1)
    next_monday = monday + pd.Timedelta(weeks=1)
    store_daily = pd.DataFrame({"store": ["S1"], "delivery_date": [next_monday], "quantity": [1400.0]})
    schedule = pd.DataFrame({"store": ["S1", "S1"], "dc": ["DC1", "DC2"], "delivery_date": [next_monday] * 2})

    daily = split_by_dc(store_daily, schedule, history, until).set_index("dc")["quantity"]
    # DC2 has no Monday history, so both DCs are compared on their totals (1000 : 400), not 100 : 400
    assert daily["DC1"] == pytest.approx(1000)
    assert daily["DC2"] == pytest.approx(400)


def test_a_dc_without_history_gets_the_average_weight_of_the_others():
    history = pd.DataFrame([delivery("S1", "DC1", 0, 300), delivery("S1", "DC2", 0, 100)])
    until = monday + pd.Timedelta(weeks=1)
    next_monday = monday + pd.Timedelta(weeks=1)
    store_daily = pd.DataFrame({"store": ["S1"], "delivery_date": [next_monday], "quantity": [400.0]})
    schedule = pd.DataFrame({"store": ["S1"] * 3, "dc": ["DC1", "DC2", "DC3"], "delivery_date": [next_monday] * 3})

    daily = split_by_dc(store_daily, schedule, history, until).set_index("dc")["quantity"]
    assert daily["DC3"] == pytest.approx(400 * 200 / 600)      # average weight 200 of 300, 100 and 200


def test_schedule_needs_a_dc_column(tmp_path):
    path = tmp_path / "schedule.csv"
    pd.DataFrame({"store": ["S1"], "delivery_date": ["2026-01-12"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="dc"):
        load_schedule(path)
