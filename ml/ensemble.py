# This file is responsible for the ensemble training of both the Gradient boosting models:- XGBoost and LightGBM boost. First the weighted Blender is trained which learns the different values of weights from the data itself by using scipy's minimize() method and chooses the one with the best R2 score and minimumm rmse value. Also the predictions are mutliplied by the weights decided by the minimize() method. Then in this file only StackingEnsemble is also created which trains a ridge regression model on both the Gradient Boosting Models. The ridge regression trains a straight line on the predictions of both the models then makes a final prediction. It controls its training by a parameter alpha which avoids this model to get overfit on the training dataset. At last a report is being prepared of all the individual predictions of the models, the weighted blend, ridge model along with their r2 scores and the best one is being chosen for the final predictions.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
import config


class WeightedBlender:
    """
    Simple weighted average of model predictions.

    Finds optimal weights by minimizing validation R² error.
    If XGBoost gives 0.85 R² and LightGBM gives 0.84 R², the blend
    might give 0.87 R² with weights like [0.55, 0.45].
    """

    def __init__(self):
        self.weights = None

    def optimize_weights(
        self,
        oof_predictions: list[np.ndarray],
        y_true: np.ndarray,
    ) -> np.ndarray:
        """
        Find optimal blend weights by minimizing -R² on OOF predictions.

        Args:
            oof_predictions: List of OOF prediction arrays, one per model
            y_true:          Actual target values

        Returns:
            Optimal weight array (sums to 1.0)
        """
        n_models = len(oof_predictions)

        def objective(weights):
            """Negative R² (we minimize, so negative = maximize R²)."""
            blended = sum(w * p for w, p in zip(weights, oof_predictions))
            return -r2_score(y_true, blended)

        # Start with equal weights
        initial = np.ones(n_models) / n_models

        # Constrain: weights sum to 1, each weight between 0 and 1
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints={"type": "eq", "fun": lambda w: sum(w) - 1},
        )

        self.weights = result.x
        return self.weights

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Blend predictions using optimized weights."""
        if self.weights is None:
            raise RuntimeError("Call optimize_weights() first.")
        blended = sum(w * p for w, p in zip(self.weights, predictions))
        return np.clip(blended, 0, None)


class StackingEnsemble:
    """
    Stacking ensemble with a Ridge regression meta-learner.

    Level 0: XGBoost and LightGBM generate out-of-fold (OOF) predictions
    Level 1: Ridge regression learns "when to trust XGBoost vs. LightGBM"

    More powerful than simple blending because the meta-learner can learn
    non-uniform weighting (e.g., trust XGBoost more for peak hours).
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Ridge regularization strength. Higher = more conservative.
        """
        self.meta_model = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(
        self,
        oof_predictions: list[np.ndarray],
        y_true: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Train the meta-learner on OOF predictions.

        Args:
            oof_predictions: List of OOF prediction arrays from base models
            y_true:          Actual target values

        Returns:
            self
        """
        # Stack OOF predictions as columns → meta-features
        meta_X = np.column_stack(oof_predictions)
        self.meta_model.fit(meta_X, y_true)
        self.is_fitted = True
        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """
        Generate ensemble prediction using the meta-learner.

        Args:
            predictions: List of prediction arrays from base models

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        meta_X = np.column_stack(predictions)
        preds = self.meta_model.predict(meta_X)
        return np.clip(preds, 0, None)


def evaluate_ensemble(
    oof_xgb: np.ndarray,
    oof_lgb: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Compare individual models vs. ensemble performance.

    Returns a report dict with R² scores for each approach.
    """
    report = {}

    # Individual models
    report["xgboost_r2"] = r2_score(y_true, oof_xgb)
    report["lightgbm_r2"] = r2_score(y_true, oof_lgb)

    # Simple average (equal weights)
    avg_pred = (oof_xgb + oof_lgb) / 2
    report["simple_avg_r2"] = r2_score(y_true, avg_pred)

    # Optimized blending
    blender = WeightedBlender()
    weights = blender.optimize_weights([oof_xgb, oof_lgb], y_true)
    blend_pred = blender.predict([oof_xgb, oof_lgb])
    report["blended_r2"] = r2_score(y_true, blend_pred)
    report["blend_weights"] = {"xgboost": weights[0], "lightgbm": weights[1]}

    # Stacking
    stacker = StackingEnsemble()
    stacker.fit([oof_xgb, oof_lgb], y_true)
    stack_pred = stacker.predict([oof_xgb, oof_lgb])
    report["stacked_r2"] = r2_score(y_true, stack_pred)
    report["stack_coefficients"] = {
        "xgboost": stacker.meta_model.coef_[0],
        "lightgbm": stacker.meta_model.coef_[1],
        "intercept": stacker.meta_model.intercept_,
    }

    # Pick best method
    methods = {
        "xgboost": report["xgboost_r2"],
        "lightgbm": report["lightgbm_r2"],
        "simple_avg": report["simple_avg_r2"],
        "blended": report["blended_r2"],
        "stacked": report["stacked_r2"],
    }
    report["best_method"] = max(methods, key=methods.get)
    report["best_r2"] = methods[report["best_method"]]

    return report, blender, stacker
