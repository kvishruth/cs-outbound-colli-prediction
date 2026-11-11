import pandas as pd
import pytest

from cs_outbound_coli_prediction.features import in_christmas_period
from cs_outbound_coli_prediction.quantiles import (
    add_quantiles,
    day_type_table,
    dc_quantile_table,
    fit_factors,
    pinball_loss,
    quantile_metrics,
    rolling_dc_forecasts,
)


@pytest.fixture(scope="module")
def calibration(deliveries):
    return rolling_dc_forecasts(deliveries)


def test_pinball_loss_punishes_under_forecasts_more_at_a_high_quantile():
    actual = pd.Series([100.0])
    assert pinball_loss(actual, pd.Series([80.0]), 0.9) == pytest.approx(0.18)    # 20 too low
    assert pinball_loss(actual, pd.Series([120.0]), 0.9) == pytest.approx(0.02)   # 20 too high


def test_factors_per_dc_with_a_pooled_row_for_dcs_with_few_days():
    many = pd.DataFrame({"dc": "A", "actual": 150.0, "forecast": 100.0}, index=range(40))
    few = pd.DataFrame({"dc": "B", "actual": 100.0, "forecast": 100.0}, index=range(5))
    factors = fit_factors(pd.concat([many, few], ignore_index=True))

    assert factors.loc["A", "p80"] == pytest.approx(1.5)
    assert "B" not in factors.index
    forecast = pd.DataFrame({"dc": ["A", "B"], "quantity": [100.0, 100.0]})
    result = add_quantiles(forecast, factors)
    assert result.loc[0, "quantity_p90"] == pytest.approx(150)
    assert result.loc[1, "quantity_p90"] == pytest.approx(100 * factors.loc["all", "p90"])


def test_rolling_folds_only_hold_their_own_weeks(calibration):
    sizes = calibration.groupby("fold").size()
    assert list(sizes.index) == [0, 1, 2]
    assert sizes.max() < 1.5 * sizes.min()
    first_dates = calibration.groupby("fold")["delivery_date"].min()
    assert first_dates[2] < first_dates[1] < first_dates[0]


def test_quantile_metrics_cover_roughly_the_target_share(calibration):
    metrics = quantile_metrics(calibration)
    assert 0.6 < metrics["holdout_coverage_p80"] < metrics["holdout_coverage_p90"] <= 1
    assert metrics["holdout_pinball_p90"] < metrics["holdout_pinball_p90_point_forecast"]


def test_calibration_holds_both_models(calibration):
    assert {"fold", "dc", "delivery_date", "actual", "forecast", "fallback"} <= set(calibration.columns)
    assert (calibration["fallback"] >= 0).all()


def test_quantile_table_per_dc(calibration):
    table = dc_quantile_table(calibration)
    assert set(table.index) == set(calibration.loc[calibration["fold"] == 0, "dc"])
    assert {"p80_coverage", "p80_pinball", "p90_coverage", "p90_pinball"} == set(table.columns)
    assert (table["p80_coverage"] <= table["p90_coverage"]).all()


def test_christmas_period_covers_the_week_of_25_december_and_four_weeks_after():
    mondays = pd.Series(pd.to_datetime(["2025-12-15", "2025-12-22", "2026-01-19", "2026-01-26", "2026-06-01"], utc=True))
    assert in_christmas_period(mondays).tolist() == [False, True, True, False, False]


def test_day_type_table_has_peak_and_holiday_rows(calibration):
    table = day_type_table(calibration)
    assert list(table.index) == ["all days", "peak days (busiest 10% by actual)", "other days",
                                 "holiday weeks (Christmas + 4)", "other weeks"]
    assert table.loc["all days", "actual_colli"] == pytest.approx(calibration["actual"].sum())
    assert table.loc["all days", "dc_days"] == len(calibration)
    assert table.loc["holiday weeks (Christmas + 4)", "dc_days"] > 0              # the folds reach Christmas 2025
    peak, everything = table.loc["peak days (busiest 10% by actual)"], table.loc["all days"]
    assert peak["main_bias"] < everything["main_bias"]                            # picked by the actual: forecasts are too low
    assert peak["main_p90_coverage"] < everything["main_p90_coverage"]
