import numpy as np

from cs_outbound_coli_prediction.backtest import backtest
from cs_outbound_coli_prediction.schedule_backtest import repeated_schedule_metrics


def test_repeating_the_last_schedule_is_less_accurate_than_knowing_it(deliveries):
    known = backtest(deliveries)[0]["holdout_wape_dc_date"]
    repeated = repeated_schedule_metrics(deliveries)

    assert set(repeated) == {f"holdout_{score}_dc_date_repeated_schedule_{model}"
                             for score in ("wape", "bias") for model in ("main", "fallback")}
    assert np.isfinite(list(repeated.values())).all()
    assert repeated["holdout_wape_dc_date_repeated_schedule_main"] > known
