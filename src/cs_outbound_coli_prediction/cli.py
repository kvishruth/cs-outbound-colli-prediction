"""Command line: `backtest` scores the models, `train` saves a new model version, `forecast` writes the forecasts,
`make-schedule` writes a starting schedule to edit."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import config
from .backtest import backtest
from .data import assumed_schedule, load_deliveries, load_schedule
from .pipeline import dc_totals, forecast_daily, train_models, unscheduled_weeks
from .quantiles import (
    add_quantiles,
    day_type_table,
    dc_quantile_table,
    fit_factors,
    quantile_metrics,
    rolling_dc_forecasts,
)
from .registry import load_current, promote, save_version
from .schedule_backtest import repeated_schedule_metrics
from .validate import validate_age, validate_deliveries, validate_schedule

logger = logging.getLogger(__name__)


def score_models(deliveries):
    """Backtest the models: metrics, breakdown tables and the out-of-sample DC forecasts used for the quantiles."""
    calibration = rolling_dc_forecasts(deliveries)       # out-of-sample DC x date forecasts of three holdouts
    metrics, tables = backtest(deliveries)
    metrics |= quantile_metrics(calibration)
    metrics |= repeated_schedule_metrics(deliveries)
    tables["dc"] = tables["dc"].join(dc_quantile_table(calibration))
    tables["day_type"] = day_type_table(calibration)
    return metrics, tables, calibration


def run_backtest(args):
    deliveries = load_deliveries(args.input)
    validate_deliveries(deliveries)
    metrics, tables, _ = score_models(deliveries)
    print(json.dumps(metrics, indent=2))
    for name, table in tables.items():
        order = {"weekday": "calendar order", "day_type": "all folds"}.get(name, "worst first, top 10")
        print(f"\nWAPE and bias per {name} (furthest horizon, {order})")
        print(table.round(3).head(10).to_string())


def run_train(args):
    deliveries = load_deliveries(args.input)
    validate_deliveries(deliveries)

    metrics, tables, calibration = score_models(deliveries)
    models = train_models(deliveries)
    models["quantile_factors"] = fit_factors(calibration)
    version = save_version(models, args.input, deliveries, metrics, args.models_dir, tables)
    logger.info("saved version %s (holdout WAPE %.3f, fallback %.3f)",
                version, metrics["holdout_wape_main"], metrics["holdout_wape_fallback"])

    # the chosen model only goes live if it beats the fallback model on the holdout
    if metrics["holdout_wape_main"] < metrics["holdout_wape_fallback"]:
        promote(version, args.models_dir)
        logger.info("version %s is now current", version)
    else:
        logger.warning("version %s does not beat the fallback model and is not promoted", version)


def run_forecast(args):
    deliveries = load_deliveries(args.input)
    validate_deliveries(deliveries)
    validate_age(deliveries, pd.Timestamp.now(tz="UTC"), args.max_age_days)

    if args.schedule:
        schedule = load_schedule(args.schedule)
    else:
        logger.warning("no schedule given: repeating the delivery dates of the last week")
        schedule = assumed_schedule(deliveries)
    validate_schedule(schedule, deliveries)

    models, metadata = load_current(args.models_dir)
    weekly, daily = forecast_daily(deliveries, schedule, models)
    weekly["model_version"] = metadata["version"]

    origin = (deliveries["Monday_of_week"].max() + pd.Timedelta(weeks=1)).date()
    output_dir = args.output_dir or config.output_dir / str(origin)
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly_forecast.csv", index=False)
    daily.to_csv(output_dir / "daily_forecast.csv", index=False)
    add_quantiles(dc_totals(daily), models["quantile_factors"]).to_csv(output_dir / "dc_daily_forecast.csv", index=False)
    logger.info("model version %s: wrote %d weekly and %d daily forecasts to %s",
                metadata["version"], len(weekly), len(daily), output_dir)

    # the files are written, but a degraded forecast must not look like a successful run: exit code 1
    problems = []
    fallback_count = int((weekly["model"] == "fallback").sum())
    if fallback_count:
        problems.append(f"{fallback_count} of {len(weekly)} weekly forecasts come from the fallback model")
    lost_count = unscheduled_weeks(weekly, schedule)
    if lost_count:
        problems.append(f"{lost_count} weekly forecasts have no scheduled delivery date and are not in the daily forecast")
    for problem in problems:
        logger.error(problem)
    return 1 if problems else 0


def run_make_schedule(args):
    """Write the stand-in schedule (the delivery lines of the last week, repeated) so it can be edited into the real plan."""
    deliveries = load_deliveries(args.input)
    validate_deliveries(deliveries)
    schedule = assumed_schedule(deliveries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(args.output, index=False)
    logger.warning("wrote %d lines to %s: the delivery lines of the last week, repeated for %d weeks. "
                   "Edit them into the real plan before forecasting", len(schedule), args.output, config.max_horizon)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cs-outbound-colli-prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, run, help_text in [
        ("backtest", run_backtest, "score the chosen and the fallback model on the last weeks"),
        ("train", run_train, "backtest, train and save a new model version (promoted if it beats the fallback)"),
        ("forecast", run_forecast, "forecast the next weeks and split them per delivery date"),
        ("make-schedule", run_make_schedule, "write the delivery lines of the last week, repeated, as a schedule to edit"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(run=run)
        command.add_argument("--input", type=Path, default=config.input_path)
        if name not in ("backtest", "make-schedule"):
            command.add_argument("--models-dir", type=Path, default=config.models_dir)
        if name == "make-schedule":
            command.add_argument("--output", type=Path, default=config.input_path.parent / "plan.csv",
                                 help="default: %(default)s")
        if name == "forecast":
            command.add_argument("--schedule", type=Path, help="CSV with store, dc and delivery_date of the coming weeks")
            command.add_argument("--max-age-days", type=int, default=config.max_data_age_days,
                                 help="refuse to forecast if the data ends longer ago (default: %(default)s)")
            command.add_argument("--output-dir", type=Path, help="default: outputs/<first forecast week>")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.run(args)
