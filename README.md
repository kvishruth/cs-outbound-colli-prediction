# Outbound colli prediction

Forecast of outbound colli per DC and delivery date, built from weekly store forecasts, for the current week and
the next 7, with P80 and P90 for staffing. Business chain: worker picking cost per colli, then the decision on workers per DC and per day,
then this forecast (ML target), scored by WAPE, bias and pinball loss. The handover note and the experiment notebooks are
not part of this repository.

## Setup

You need Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) (`uv run` installs the libraries from `uv.lock`).

```bash
git clone <repository url> && cd cs-outbound-colli-prediction
mkdir -p data/input artifacts outputs      # these folders are not in the repository (see .gitignore)
# put the delivery data in data/input/store_deliveries.csv, or pass --input <file> to every command
```

| Folder | Holds | Created by |
|---|---|---|
| `data/input/` | the delivery data, and the schedule of the coming weeks | you |
| `artifacts/models/` | saved model versions and `current.txt`, the version `forecast` uses | `train` |
| `outputs/<first forecast week>/` | the forecast files | `forecast` |

The paths are set in `src/cs_outbound_coli_prediction/config.py`, relative to the project folder. Run the project from the
clone (`uv run` or Docker): a normal `pip install .` would put the paths inside `site-packages`.

## Input data

**Deliveries** (`store_deliveries.csv`), one row per delivery line. Required columns: `store`, `country_store`, `dc`,
`delivery_date`, `Monday_of_week`, `delivery_quantity`, `sales_index` (sales may be empty). `Monday_of_week` is the Monday of
the week of `delivery_date`. The data has to end with a complete week.

**Schedule** (a CSV for `--schedule`): the planned deliveries of the coming weeks, with the columns `store`, `dc` and
`delivery_date`. Without `--schedule` the delivery lines of the last week are repeated, which is much less accurate (DC and date
WAPE 0.335 instead of 0.103 in the backtest).

The inputs are checked before anything runs, and a problem raises an error that lists what is wrong: missing columns or values,
negative quantities, repeated rows or delivery lines, a week without deliveries inside the history, an incomplete last week,
an empty schedule, a schedule without a DC or beyond the 8 weeks, and a store in the schedule without a recent delivery.

## Run

```bash
uv run cs-outbound-colli-prediction backtest     # score the model and the fallback on the last 12 weeks
uv run cs-outbound-colli-prediction train        # backtest, save a model version, promote it if it beats the fallback
uv run cs-outbound-colli-prediction forecast --schedule plan.csv
uv run pytest                                    # needs the data in data/input/
```

- `forecast` needs a trained version, so run `train` first.
- `forecast` refuses data that ended more than 10 days ago. For older data, for example the fabricated data set, add
  `--max-age-days 1000`.
- `forecast` writes `weekly_forecast.csv`, `daily_forecast.csv` and `dc_daily_forecast.csv` (with P80 and P90) to
  `outputs/<first forecast week>/`, and exits with code 1 if any weekly forecast came from the fallback model or has no scheduled
  delivery date. The files are still written.
- A saved version only loads with the same versions of LightGBM, XGBoost, pandas, numpy and joblib it was trained with.
  After an upgrade, run `train` again.

## Docker

The image holds the code and libraries only. The data, the models and the outputs are mounted, so they stay on your machine.
Create the three folders first (see Setup): if a mounted folder does not exist, Docker creates it owned by root and the
container, which runs as uid 1000, cannot write to it.

```bash
docker build -t cs-outbound-colli-prediction .
run="docker run --rm -v $PWD/data/input:/app/data/input -v $PWD/artifacts:/app/artifacts -v $PWD/outputs:/app/outputs cs-outbound-colli-prediction"
$run backtest
$run train
$run forecast --schedule /app/data/input/plan.csv      # add --max-age-days 1000 for old data
```

The commit is saved as `unknown` in the metadata of a version trained in the image, because `.git` is not copied.
The tests and the notebooks are not in the image either: they need the data and the development libraries.

## Modules

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
