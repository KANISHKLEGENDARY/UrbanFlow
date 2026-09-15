# This file defines the LightGBM model for prediction of demand at various zones. Instead of writing the hyperparameters in this file itself they are written inside config.py and are rendered by the model from there whenever it is being called. The __init__() function intializwes those hyperparameter for the model. Then the LightGBM model is being trained on the training dataset. The strategy of early stopping is being used here for training. If the model does not improves rmse score consecutively for the 200 rounds then it is being stopped there only and the round in which its performance was best is being saved so that in future while training ensemble unecessary rounds of training are not made. Then the model is being used for prediction on actual test dataset and after that the model identifies the features which have contributed a lot in its prediction. 

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import lightgbm as lgb
import config


class LightGBMModel:

    def __init__(self, params: dict = None):
        self.params = params or config.LGB_PARAMS.copy()
        self.model = None
        self.best_iteration = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
    ) -> "LightGBMModel":

        params = self.params.copy()

        self.model = lgb.LGBMRegressor(**params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "rmse"
            fit_kwargs["callbacks"] = [
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            ]

        if feature_names:
            fit_kwargs["feature_name"] = feature_names

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.best_iteration = self.model.best_iteration_ if hasattr(self.model, "best_iteration_") else params.get("n_estimators")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
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
        return "LightGBM"
