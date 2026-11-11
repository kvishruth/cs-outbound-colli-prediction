import pandas as pd

from cs_outbound_coli_prediction.cli import main
from cs_outbound_coli_prediction.data import assumed_schedule, load_schedule
from cs_outbound_coli_prediction.validate import validate_schedule


def test_make_schedule_writes_a_schedule_that_forecast_accepts(tmp_path, deliveries):
    input_path, output_path = tmp_path / "deliveries.csv", tmp_path / "folder" / "plan.csv"
    deliveries.to_csv(input_path, index=False)

    assert main(["make-schedule", "--input", str(input_path), "--output", str(output_path)]) == 0

    schedule = load_schedule(output_path)
    validate_schedule(schedule, deliveries)
    pd.testing.assert_frame_equal(schedule.reset_index(drop=True), assumed_schedule(deliveries).reset_index(drop=True))
