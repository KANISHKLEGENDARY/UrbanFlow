"""
This code file first loads the train and test data using pandas. After loading the data it displays the number of rows, months, zones for confirming that the data is being loaded properly in the pipeline. In the step-2 instead of random K-Fold cross validation, GroupKFold validation is being used here. This is being done to avoid leakage of data between training and testing sets so that model do not pretend to perform well. GroupKFold cross validation ensures that all the rows belonging to the same zone remain either in the training or entirely in the validation set. Further the leak free feature engineering is being done to maintain the motive of avoiding data leakage. For each fold the code computes features by using only the training rows. Then the validation rows get their features computed using the statistics calculated from the training fold only. Then in the next step for each fold two gradient boosting models are trained:- XGBoost, LightGBM by using the method of early stopping.(The model train to maximum number trees but stop if the validation score does not improve for a certain number of rounds.) The out of fold predictions obtained from the training of both the models is then used to find the best performing model made by the ensemble of both XGBoost and LightGBM boost. Three ways are considered:- normal avg((pred1+pred2)/2), weighted avg(w1*pred1 + w2*pred2) and stacking(training another model on top of the predictions of the previous models). Then both the models are trained on full training dataset. The test dataset is then also used by both the models for predictions and then those predictions are then tweaked according to the best performing model obtained from the 3 strategies used. The .pkl files of models is being created and at last shap analysis is run to find the reason for a particular prediction of the model.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
import joblib

import config
from ml.features import compute_all_features, get_feature_names
from ml.xgboost_model import XGBoostModel
from ml.lightgbm_model import LightGBMModel
from ml.ensemble import evaluate_ensemble, WeightedBlender, StackingEnsemble
from ml.explainer import run_full_shap_analysis


def load_data():
    """Load preprocessed train and test data."""
    train_path = config.PROCESSED_DIR / "train.parquet"
    test_path = config.PROCESSED_DIR / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {train_path}\n"
            f"Run 'python data/download_nyc_data.py' then 'python data/preprocess.py' first."
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train_months = ", ".join(str(m) for m in config.TRAIN_MONTHS)
    test_months = ", ".join(str(m) for m in config.TEST_MONTHS)
    print(f"  Train: {len(train):,} rows (months {train_months})")
    print(f"  Test:  {len(test):,} rows (months {test_months})")
    print(f"  Zones: {train['zone_id'].nunique()} train, {test['zone_id'].nunique()} test")

    return train, test


def run_cross_validation(train: pd.DataFrame, features: list):
    """
    GroupKFold cross-validation with leak-free feature engineering.

    Same strategy as your Flipkart GRiD hackathon:
    - Split by zone_id → validation zones never seen during training
    - Recompute demand features from training fold only per fold
    - Both XGBoost and LightGBM trained per fold, OOF predictions collected

    Returns:
        oof_xgb, oof_lgb: Out-of-fold predictions for each model
        fold_models: Dict of trained models per fold (for reference)
    """
    groups = train[config.CV_GROUP_COLUMN].values
    gkf = GroupKFold(n_splits=config.N_FOLDS)

    oof_xgb = np.zeros(len(train))
    oof_lgb = np.zeros(len(train))
    fold_models = {"xgboost": [], "lightgbm": []}

    y_full = train["demand"].values

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, y_full, groups)):
        fold_start = time.time()
        val_zones = train.iloc[val_idx][config.CV_GROUP_COLUMN].nunique()
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1}/{config.N_FOLDS} | {val_zones} validation zones")
        print(f"{'='*50}")

        # Split data
        tr_df = train.iloc[tr_idx].copy()
        val_df = train.iloc[val_idx].copy()

        # Compute fold-safe features (ref_df = training fold only)
        tr_df = compute_all_features(tr_df, ref_df=tr_df)
        val_df = compute_all_features(val_df, ref_df=tr_df)  # <-- key: uses tr_df as ref

        # Prepare arrays
        X_tr = tr_df[features].values
        y_tr = tr_df["demand"].values
        X_val = val_df[features].values
        y_val = val_df["demand"].values

        # Train XGBoost
        print(f"\n  Training XGBoost (fold {fold+1})...")
        xgb_model = XGBoostModel()
        xgb_model.fit(X_tr, y_tr, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        fold_models["xgboost"].append(xgb_model)

        xgb_r2 = r2_score(y_val, oof_xgb[val_idx])
        print(f"  XGBoost fold R²: {xgb_r2:.4f} | best_iter: {xgb_model.best_iteration}")

        # Train LightGBM
        print(f"\n  Training LightGBM (fold {fold+1})...")
        lgb_model = LightGBMModel()
        lgb_model.fit(X_tr, y_tr, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        fold_models["lightgbm"].append(lgb_model)

        lgb_r2 = r2_score(y_val, oof_lgb[val_idx])
        elapsed = time.time() - fold_start
        print(f"  LightGBM fold R²: {lgb_r2:.4f} | best_iter: {lgb_model.best_iteration}")
        print(f"  Fold time: {elapsed:.1f}s")

    return oof_xgb, oof_lgb, fold_models


def train_final_models(train: pd.DataFrame, features: list):
    """
    Retrain on ALL training data (no validation holdout) for final predictions.
    These are the models that get saved and served by the API.
    Uses a capped iteration count based on CV early stopping averages.
    """
    print("\n  Retraining on full training data...")

    # Compute features with full train as reference
    train_feat = compute_all_features(train, ref_df=train)

    X = train_feat[features].values
    y = train_feat["demand"].values

    # XGBoost -- capped at 2000 iters (avg best from CV)
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params["n_estimators"] = 2000
    xgb_params.pop("early_stopping_rounds", None)
    xgb_params.pop("eval_metric", None)
    xgb_model = XGBoostModel(params=xgb_params)
    xgb_model.fit(X, y)

    # LightGBM -- capped at 2000 iters
    lgb_params = config.LGB_PARAMS.copy()
    lgb_params["n_estimators"] = 2000
    lgb_model = LightGBMModel(params=lgb_params)
    lgb_model.fit(X, y)

    return xgb_model, lgb_model, train_feat


def predict_test(
    test: pd.DataFrame,
    train: pd.DataFrame,
    xgb_model: XGBoostModel,
    lgb_model: LightGBMModel,
    blender: WeightedBlender,
    features: list,
):
    """Generate ensemble predictions on the test set."""
    # Compute test features using full train as reference
    test_feat = compute_all_features(test, ref_df=train)
    X_test = test_feat[features].values

    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)
    ensemble_preds = blender.predict([xgb_preds, lgb_preds])

    return ensemble_preds, test_feat


def save_models(xgb_model, lgb_model, blender, explainer_obj=None):
    """Save trained models to disk for API serving."""
    joblib.dump(xgb_model, config.MODEL_DIR / "xgboost_final.pkl")
    joblib.dump(lgb_model, config.MODEL_DIR / "lightgbm_final.pkl")
    joblib.dump(blender, config.MODEL_DIR / "ensemble_weights.pkl")
    if explainer_obj:
        joblib.dump(explainer_obj, config.MODEL_DIR / "shap_explainer.pkl")
    print(f"  Models saved to {config.MODEL_DIR}")


def main():
    total_start = time.time()

    print("=" * 60)
    print("  UrbanFlow — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────
    print("\n[1/7] Loading preprocessed data...")
    train, test = load_data()

    features = get_feature_names()
    print(f"  Features ({len(features)}): {features}")

    # ── Step 2: Cross-validation ──────────────────────────────
    print("\n[2/7] Running GroupKFold cross-validation...")
    oof_xgb, oof_lgb, fold_models = run_cross_validation(train, features)

    # ── Step 3: Evaluate ensemble ─────────────────────────────
    print("\n[3/7] Evaluating ensemble methods...")
    y_true = train["demand"].values

    # in train.py, evaluate step
    report, blender, stacker = evaluate_ensemble(oof_xgb, oof_lgb, y_true)

    # pick the actual best combiner object, not just the best number
    best_combiner = stacker if report["best_method"] == "stacked" else blender

    print(f"\n  {'Model':<20} {'R²':>10}")
    print(f"  {'-'*30}")
    print(f"  {'XGBoost':<20} {report['xgboost_r2']:>10.4f}")
    print(f"  {'LightGBM':<20} {report['lightgbm_r2']:>10.4f}")
    print(f"  {'Simple Average':<20} {report['simple_avg_r2']:>10.4f}")
    print(f"  {'Optimized Blend':<20} {report['blended_r2']:>10.4f}")
    print(f"  {'Stacking (Ridge)':<20} {report['stacked_r2']:>10.4f}")
    print(f"\n  Best method: {report['best_method']} (R² = {report['best_r2']:.4f})")
    print(f"  Blend weights: XGB={report['blend_weights']['xgboost']:.3f}, "
          f"LGB={report['blend_weights']['lightgbm']:.3f}")

    # ── Step 4: Train final models ────────────────────────────
    print("\n[4/7] Training final models on full data...")
    xgb_final, lgb_final, train_feat = train_final_models(train, features)

    # ── Step 5: Test predictions ──────────────────────────────
    print("\n[5/7] Generating test predictions...")
    test_preds, test_feat = predict_test(
        test, train, xgb_final, lgb_final, best_combiner, features
    )

    # Evaluate on test set (we have labels since we split by time)
    test_r2 = r2_score(test["demand"].values, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test["demand"].values, test_preds))

    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test score (competition-style): {max(0, 100 * test_r2):.2f} / 100")

    # Save core serving artifacts before SHAP plotting. SHAP depends on
    # Matplotlib/font availability, but trained models should not be lost
    # if plot generation fails on a local workstation.
    print("\n  Saving core model artifacts before SHAP...")
    save_models(xgb_final, lgb_final, best_combiner)

    # ── Step 6: SHAP analysis ─────────────────────────────────
    print("\n[6/7] Running SHAP explainability analysis...")
    shap_explainer = None
    try:
        X_train_feat = train_feat[features].values

        shap_explainer = run_full_shap_analysis(
            xgb_final.model, X_train_feat, features, model_name="xgboost"
        )
        run_full_shap_analysis(
            lgb_final.model, X_train_feat, features, model_name="lightgbm"
        )
    except Exception as exc:
        print(f"  [WARN] SHAP analysis skipped: {exc}")

    # ── Step 7: Save everything ───────────────────────────────
    print("\n[7/7] Saving models and reports...")
    save_models(xgb_final, lgb_final, best_combiner, shap_explainer)

    # Save training report
    report_path = config.REPORTS_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("UrbanFlow — Training Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: NYC TLC HVFHV (Uber/Lyft) Feb-Apr 2026\n")
        f.write(f"Aggregation: 15-minute windows ({config.SLOTS_PER_DAY} slots/day)\n")
        f.write(f"Train rows: {len(train):,} (months {', '.join(str(m) for m in config.TRAIN_MONTHS)})\n")
        f.write(f"Test rows:  {len(test):,} (months {', '.join(str(m) for m in config.TEST_MONTHS)})\n")
        f.write(f"Features:   {len(features)}\n\n")
        f.write(f"Cross-Validation (GroupKFold, {config.N_FOLDS} folds by zone):\n")
        f.write(f"  XGBoost OOF R²:      {report['xgboost_r2']:.4f}\n")
        f.write(f"  LightGBM OOF R²:     {report['lightgbm_r2']:.4f}\n")
        f.write(f"  Blended OOF R²:      {report['blended_r2']:.4f}\n")
        f.write(f"  Stacked OOF R²:      {report['stacked_r2']:.4f}\n")
        f.write(f"  Best: {report['best_method']} ({report['best_r2']:.4f})\n\n")
        f.write(f"Test Set Performance:\n")
        f.write(f"  R²:   {test_r2:.4f}\n")
        f.write(f"  RMSE: {test_rmse:.4f}\n")
        f.write(f"  Score: {max(0, 100 * test_r2):.2f} / 100\n")

    print(f"  Report saved: {report_path}")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  [OK] Training complete!")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best OOF R²: {report['best_r2']:.4f} ({report['best_method']})")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"  Test Score:  {max(0, 100 * test_r2):.2f} / 100")
    print(f"\n  Models: {config.MODEL_DIR}")
    print(f"  SHAP:   {config.SHAP_DIR}")
    print(f"  Report: {report_path}")
    print(f"\n  Next step: uvicorn api.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
