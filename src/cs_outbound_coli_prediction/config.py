"""Settings of the pipeline. The values come from experimentation/feature_engineering.ipynb."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
input_path = root / "data" / "input" / "store_deliveries.csv"
models_dir = root / "artifacts" / "models"
output_dir = root / "outputs"

seed = 0
thread_count = 4           # threads per model; with all cores the two libraries compete and training gets slower
max_horizon = 8            # forecast the current week and the next 7 (run on Monday, data up to Monday 00:00)
recent_weeks_count = 3     # most recent known weeks used as lag features
active_weeks = 13          # a store is forecast if it received a delivery in the last 13 weeks
holdout_weeks = 12         # backtest: score the last 12 weeks, training stops max_horizon weeks before them
max_data_age_days = 10     # forecast only if the delivery data ends at most this many days ago
min_last_week_ratio = 0.4  # the last week must hold at least 40% of the usual weekly quantity (Christmas week: about 50%)
christmas_weeks_after = 4  # holiday weeks: the Christmas week and the 4 weeks after it (the volume stays low until mid-January)
quantile_levels = (0.8, 0.9)  # quantiles of the DC x date forecast (P80, P90)
target = "weekly_quantity"
categorical_columns = ["country_store", "store"]

feature_columns = [
    "qty_last_1", "qty_last_2", "qty_last_3", "qty_lag_52",
    "mean_4", "std_4", "mean_13", "std_13", "mean_52", "std_52",
    "sales_last_1", "sales_last_2", "sales_last_3", "sales_mean_4",
    "week_of_year", "month", "week_sin", "week_cos", "xmas_week", "xmas_1w_before", "xmas_2w_before",
    "n_days", "days_ratio", "horizon",
    *categorical_columns,
]

lgb_params = {
    "objective": "l1", "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40,
    "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
    "random_state": seed, "n_jobs": thread_count, "verbose": -1,
}
xgb_params = {
    "objective": "count:poisson", "n_estimators": 200, "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 20.0,
    "tree_method": "hist", "enable_categorical": True, "random_state": seed, "n_jobs": thread_count,
}
