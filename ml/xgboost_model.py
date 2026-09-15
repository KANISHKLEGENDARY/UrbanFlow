# This file defines the XGBoost model for prediction of demand at various zones. Instead of writing the hyperparameters in this file itself they are written inside config.py and are rendered by the model from there whenever it is being called. The __init__() function intializwes those hyperparameter for the model. Then the XGBoost model is being trained on the training dataset. The strategy of early stopping is being used here for training. If the model does not improves rmse score consecutively for the 200 rounds then it is being stopped there only and the round in which its performance was best is being saved so that in future while training ensemble unecessary rounds of training are not made. Then the model is being used for prediction on actual test dataset and after that the model identifies the features which have contributed a lot in its prediction. 

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from xgboost import XGBRegressor
import config


class XGBoostModel:

    def __init__(self, params: dict = None):
        self.params = params or config.XGB_PARAMS.copy()
        self.model = None
        self.best_iteration = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
    ) -> "XGBoostModel":

        params = self.params.copy()
        early_stopping = params.pop("early_stopping_rounds", config.EARLY_STOPPING_ROUNDS)
        eval_metric = params.pop("eval_metric", "rmse")

        if X_val is not None:
            params["early_stopping_rounds"] = early_stopping

        self.model = XGBRegressor(**params, eval_metric=eval_metric)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = 100 

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.best_iteration = getattr(self.model, "best_iteration", self.params.get("n_estimators"))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def get_feature_importance(self, feature_names: list = None) -> dict:
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))

    @property
    def name(self) -> str:
        return "XGBoost"
