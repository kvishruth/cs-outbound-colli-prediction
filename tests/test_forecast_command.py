import pandas as pd

from cs_outbound_coli_prediction.cli import main
from cs_outbound_coli_prediction.data import assumed_schedule, monday_of
from cs_outbound_coli_prediction.registry import promote, save_version

factors = pd.DataFrame({"p80": [1.1], "p90": [1.2]}, index=["all"])


class Broken:
    def predict(self, table):
        raise RuntimeError("model unavailable")


def run_forecast(tmp_path, deliveries, models, schedule=None):
    """Save the models as the current version, run the forecast command and return its exit code."""
    input_path = tmp_path / "deliveries.csv"
    deliveries.to_csv(input_path, index=False)
    models_dir = tmp_path / "models"
    promote(save_version({**models, "quantile_factors": factors}, input_path, deliveries, {}, models_dir), models_dir)

    command = ["forecast", "--input", str(input_path), "--models-dir", str(models_dir),
               "--output-dir", str(tmp_path / "out"), "--max-age-days", "100000"]
    if schedule is not None:
        schedule.to_csv(tmp_path / "schedule.csv", index=False)
        command += ["--schedule", str(tmp_path / "schedule.csv")]
    return main(command)


def test_forecast_exits_with_0_when_all_is_well(tmp_path, deliveries, models):
    assert run_forecast(tmp_path, deliveries, models) == 0
    assert (tmp_path / "out" / "dc_daily_forecast.csv").exists()


def test_forecast_exits_with_1_when_the_fallback_model_was_used(tmp_path, deliveries, models):
    assert run_forecast(tmp_path, deliveries, {**models, "main": Broken()}) == 1
    assert (tmp_path / "out" / "weekly_forecast.csv").exists()          # the forecast is still written


def test_forecast_exits_with_1_when_weeks_have_no_scheduled_date(tmp_path, deliveries, models):
    schedule = assumed_schedule(deliveries)
    second_week = monday_of(schedule["delivery_date"]) == monday_of(schedule["delivery_date"]).min() + pd.Timedelta(weeks=1)
    assert run_forecast(tmp_path, deliveries, models, schedule[~second_week]) == 1
