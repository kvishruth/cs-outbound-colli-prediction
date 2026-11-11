"""The chosen model and the fallback model. Both predict the weekly quantity of a feature table."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import categorical_columns, feature_columns, lgb_params, target, xgb_params


class Forecaster:
    """Chosen model: the average of a LightGBM (L1 objective) and an XGBoost (Poisson objective)."""

    def fit(self, table):
        # category levels come from the training data, unseen stores become missing at prediction time
        self.categories = {column: pd.CategoricalDtype(sorted(table[column].unique()))
                           for column in categorical_columns}
        features = self._features(table)
        self.models = [lgb.LGBMRegressor(**lgb_params).fit(features, table[target]),
                       xgb.XGBRegressor(**xgb_params).fit(features, table[target])]
        return self

    def predict(self, table):
        features = self._features(table)
        return np.clip(np.mean([model.predict(features) for model in self.models], axis=0), 0, None)

    def _features(self, table):
        return table.astype(self.categories)[feature_columns]


class FallbackForecaster:
    """Fallback model: trailing 52-week mean (13-week mean while a store has a short history).

    It needs no training and only uses the quantity history, so it still works when the main model or
    the sales and schedule inputs fail. It was the best simple baseline in the notebook.
    """

    def predict(self, table):
        return table["mean_52"].fillna(table["mean_13"]).to_numpy()
