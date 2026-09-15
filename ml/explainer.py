# This code file is responsible for idenitfying the reasons behind the predictions made by the models. The SHAP being used is run on the individual XGBoost and LightGbm models as it needs to be run on the trees not on the mathematical predictions so that it can identify the reasons behind the predictions made by the model. This file first calculates the shap values for every feature and stores them for further analysis. Those shap values are then used for creating bar charts depicting the top features for evry model responsible for the prediction. It is being done on the random 2000 rows as it would have taken a lot of time if it was made to run on 65 million rows. Then the beeswarm plots are being generated which tells that if the value of a particular feature gets increased or decreased thn what is its effect on the demand of rides. It is also being generated for both models. At last waterfall plots are being created and they help the cistomer know the reason that the reasons responsible for the demand of specific fare from them. At last there is a function that runs evrything included in this file. The outputs which are produced are as follows:-
# Produces:
#     outputs/shap/feature_importance.png  — Global feature ranking
#     outputs/shap/summary_beeswarm.png    — How feature values affect predictions
#     outputs/shap/waterfall_sample.png    — Single prediction breakdown
#     outputs/shap/dependence_*.png        — Feature value vs. SHAP impact


import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "urbanflow_matplotlib"),
)

import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import config


class SHAPExplainer:

    def __init__(self, model, feature_names: list):
        
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def compute_shap_values(self, X: np.ndarray, max_samples: int = 2000) -> np.ndarray:

        if len(X) > max_samples:
            idx = np.random.RandomState(config.RANDOM_STATE).choice(
                len(X), max_samples, replace=False
            )
            X = X[idx]

        self.shap_values = self.explainer.shap_values(X)
        self._X_sample = X
        return self.shap_values

    def plot_feature_importance(self, save_path: Path = None):

        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        save_path = save_path or config.SHAP_DIR / "feature_importance.png"

        fig, ax = plt.subplots(figsize=(10, 8))

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        top_n = min(20, len(self.feature_names))
        top_idx = sorted_idx[:top_n]

        ax.barh(
            range(top_n),
            mean_abs[top_idx][::-1],
            color="#4C72B0",
            edgecolor="white",
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_names[i] for i in top_idx[::-1]])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP)", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")

    def plot_summary(self, save_path: Path = None):

        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        save_path = save_path or config.SHAP_DIR / "summary_beeswarm.png"

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self._X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Summary — Feature Impact on Demand", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")

    def plot_waterfall(self, X_single: np.ndarray, index: int = 0, save_path: Path = None):

        save_path = save_path or config.SHAP_DIR / "waterfall_sample.png"

        sv = self.explainer.shap_values(X_single.reshape(1, -1))
        explanation = shap.Explanation(
            values=sv[0],
            base_values=self.explainer.expected_value,
            data=X_single,
            feature_names=self.feature_names,
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"Prediction Breakdown — Sample #{index}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")

    def explain_single(self, X_single: np.ndarray) -> dict:
        
        sv = self.explainer.shap_values(X_single.reshape(1, -1))
        base = float(self.explainer.expected_value)
        prediction = float(base + sv[0].sum())

        # Sort features by absolute impact
        contributions = []
        for i, (name, val, shap_val) in enumerate(
            zip(self.feature_names, X_single, sv[0])
        ):
            contributions.append({
                "feature": name,
                "value": round(float(val), 4),
                "impact": round(float(shap_val), 4),
            })

        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "base_value": round(base, 4),
            "prediction": round(prediction, 4),
            "top_factors": contributions[:10],
        }


def run_full_shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list,
    model_name: str = "model",
) -> SHAPExplainer:
    
    print(f"\n  Running SHAP analysis for {model_name}...")

    explainer = SHAPExplainer(model, feature_names)
    explainer.compute_shap_values(X, max_samples=2000)

    explainer.plot_feature_importance(
        config.SHAP_DIR / f"feature_importance_{model_name}.png"
    )
    explainer.plot_summary(
        config.SHAP_DIR / f"summary_{model_name}.png"
    )

    if len(X) > 0:
        explainer.plot_waterfall(
            X[0], index=0,
            save_path=config.SHAP_DIR / f"waterfall_{model_name}.png"
        )

    return explainer
