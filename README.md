# Outbound coli prediction

Forecast of outbound colli per DC and delivery date, built from weekly store forecasts, for the current week and
the next 7, with P80 and P90 for staffing. Business chain: worker picking cost per colli, then the decision on workers per DC and per day,
then this forecast (ML target), scored by WAPE, bias and pinball loss. See `HANDOVER.md` for the framing, assumptions,
results and open questions, and `experimentation/feature_engineering.ipynb` for the experiments behind the model.

```bash
uv run cs-outbound-colli-prediction backtest
uv run cs-outbound-colli-prediction train
uv run cs-outbound-colli-prediction forecast --schedule plan.csv   # plan.csv: store, dc, delivery_date of the coming weeks
uv run pytest
```

Docker: the image holds the code and libraries only. The data, the models and the outputs are mounted, so they stay on
your machine.

```bash
docker build -t cs-outbound-colli-prediction .
run="docker run --rm -v $PWD/data/input:/app/data/input -v $PWD/artifacts:/app/artifacts -v $PWD/outputs:/app/outputs cs-outbound-colli-prediction"
$run backtest
$run train
$run forecast --schedule /app/data/input/plan.csv
```

The commit is saved as `unknown` in the metadata of a version trained in the image, because `.git` is not copied.
The tests and the notebooks are not in the image either: they need the data and the development libraries.

| Module | Purpose |
|---|---|
| `config.py` | paths, horizon, feature list, tuned parameters |
| `data.py` | read deliveries and schedule |
| `validate.py` | checks on the inputs |
| `features.py` | weekly features per forecast horizon |
| `models.py` | chosen model (LightGBM + XGBoost) and fallback model (trailing mean) |
| `backtest.py` | train on the past, score the last 12 weeks, with WAPE and bias per store, DC and weekday |
| `schedule_backtest.py` | DC and date accuracy when the delivery lines of the last week are repeated instead of a known schedule |
| `registry.py` | versioned model folders with metadata; refuses to load a version trained with other library versions |
| `quantiles.py` | P80 and P90 per DC and delivery date from past forecast errors; coverage, pinball loss, peak and holiday rows |
| `split.py` | split the weekly forecast over delivery dates and DCs |
| `pipeline.py` | train and forecast with fallback |
| `cli.py` | the `backtest`, `train` and `forecast` commands |
