# Outbound colli prediction

Forecast of outbound colli per DC and delivery date, built from weekly store forecasts, for the current week and
the next 7, with P80 and P90 for staffing. Business chain: worker picking cost per colli, then the decision on workers per DC and per day,
then this forecast (ML target), scored by WAPE, bias and pinball loss. The handover note and the experiment notebooks are
not part of this repository.

There are two ways to run it: with `uv` (steps 1 to 9) or with Docker (the Docker section). The steps are the same.

## How to run

### Step 1. Install the tools

- Python 3.13 or newer and [uv](https://docs.astral.sh/uv/). `uv run` installs the libraries from `uv.lock` the first time.
- Docker, only if you want the container route.

### Step 2. Get the code

```bash
git clone <repository url>
cd cs-outbound-colli-prediction
```

Run everything from this folder. The paths are set in `src/cs_outbound_coli_prediction/config.py`, relative to the project
folder, so `uv run` or Docker from the clone is the way to run it. A normal `pip install .` would put the paths inside
`site-packages`.

### Step 3. Create the folders

```bash
mkdir -p data/input artifacts outputs
```

They are not in the repository (see `.gitignore`).

| Folder | Holds | Created by |
|---|---|---|
| `data/input/` | the delivery data, and the schedule of the coming weeks | you |
| `artifacts/models/` | saved model versions and `current.txt`, the version `forecast` uses | `train` |
| `outputs/<first forecast week>/` | the forecast files. The subfolder is created by `forecast` | `forecast` |

### Step 4. Add the delivery data

Put the delivery data in `data/input/store_deliveries.csv`, or pass `--input <file>` to every command.
The data is not in the repository: you need your own file.

One row per delivery line. Required columns:

| Column | Meaning |
|---|---|
| `store`, `country_store`, `dc` | where the delivery goes and from which DC |
| `delivery_date` | the date of the delivery |
| `Monday_of_week` | the Monday of the week of `delivery_date` |
| `delivery_quantity` | colli delivered |
| `sales_index` | sales of the store in that week (may be empty) |

The data has to end with a complete week. The inputs are checked before anything runs, and a problem raises an error that
lists what is wrong: missing columns or values, negative quantities, repeated rows or delivery lines, a week without
deliveries inside the history, and an incomplete last week.

### Step 5. Score the model (optional)

```bash
uv run cs-outbound-colli-prediction backtest
```

Scores the model and the fallback on the last 12 weeks: WAPE and bias per level, horizon, store, DC, weekday, peak days and
holiday weeks, the coverage and pinball loss of P80 and P90, and the accuracy when the schedule is not known. Nothing is saved.

### Step 6. Train a model version

```bash
uv run cs-outbound-colli-prediction train
```

Runs the backtest, trains on all history and saves a new version in `artifacts/models/<version>/`. The version is promoted
(written to `current.txt`) only if it beats the fallback on the holdout.

### Step 7. Prepare the schedule

The schedule is a CSV with the columns `store`, `dc` and `delivery_date`: the planned deliveries of the coming weeks, up to
8 weeks ahead, starting after the last week in the data. The pipeline does not decide it: it comes from whoever plans the
deliveries.

```bash
uv run cs-outbound-colli-prediction make-schedule     # writes data/input/plan.csv, use --output to change
```

`make-schedule` writes a starting file: the delivery lines of the last week, repeated for the next 8 weeks. Edit it into the
real plan. Without `--schedule` in step 8 the same lines are used without saving them, which is much less accurate (DC and
date WAPE 0.335 instead of 0.103 in the backtest).

The schedule is checked too: it must not be empty, must have a DC on every line, must not go beyond the 8 weeks, and every
store in it needs a delivery in the last 13 weeks.

### Step 8. Forecast

```bash
uv run cs-outbound-colli-prediction forecast --schedule data/input/plan.csv
```

- It needs a trained version, so run step 6 first.
- It refuses data that ended more than 10 days ago. For older data, for example the fabricated data set, add
  `--max-age-days 1000`.
- The first forecast week is the last week in the data plus one week. The forecast covers that week and the 7 after it.

### Step 9. Read the results

`forecast` writes three files to `outputs/<first forecast week>/` (use `--output-dir` for another folder). A second run for the
same week overwrites them.

| File | Rows | Columns |
|---|---|---|
| `dc_daily_forecast.csv` | per DC and delivery date | `dc`, `delivery_date`, `quantity`, `quantity_p80`, `quantity_p90` |
| `daily_forecast.csv` | per store, DC and delivery date | `store`, `dc`, `delivery_date`, `quantity` |
| `weekly_forecast.csv` | per store and week | `store`, `Monday_of_week`, `forecast`, `model` (main or fallback), `model_version` |

`forecast` exits with code 1 if any weekly forecast came from the fallback model, or has no scheduled delivery date. The files
are still written, so check the log.

## Checks

```bash
uv run ruff check src tests     # lint
uv run pytest                   # 41 tests, need the data in data/input/
```

On every push to `main` and every pull request, GitHub Actions (`.github/workflows/ci.yml`) installs the libraries with
`uv sync --locked`, runs the lint and checks that the command starts. It does not run the tests, because they need the data
file, which is not in the repository.

## Docker

The image holds the code and libraries only. The data, the models and the outputs are mounted, so they stay on your machine.
Do steps 2 to 4 first. Create the three folders before `docker run`: if a mounted folder does not exist, Docker creates it
owned by root and the container, which runs as uid 1000, cannot write to it.

```bash
docker build -t cs-outbound-colli-prediction .
run="docker run --rm -v $PWD/data/input:/app/data/input -v $PWD/artifacts:/app/artifacts -v $PWD/outputs:/app/outputs cs-outbound-colli-prediction"
$run backtest                                          # step 5
$run train                                             # step 6
$run make-schedule --output /app/data/input/plan.csv   # step 7
$run forecast --schedule /app/data/input/plan.csv      # step 8, add --max-age-days 1000 for old data
```

The commit is saved as `unknown` in the metadata of a version trained in the image, because `.git` is not copied.
The tests and the notebooks are not in the image either: they need the data and the development libraries.

## Good to know

- **Library versions:** a saved version only loads with the same versions of LightGBM, XGBoost, pandas, numpy and joblib it was
  trained with. After an upgrade, run `train` again.
- **Dates are derived, not set:** the first forecast week comes from the last week in the data, and the holdout of the
  backtest is its last 12 weeks. The only fixed number is the horizon of 8 weeks (`max_horizon` in `config.py`).

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
| `cli.py` | the `backtest`, `train`, `make-schedule` and `forecast` commands |
