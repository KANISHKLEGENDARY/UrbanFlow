# This code is responsible for generation of features which are being used for the predictions of demand of rides at various zoens in NYC. The different functions have different resposibilites of different feature genration.
# 1. compute_zone_demand_stats : This function computes the mean, std and max demand for each zone from the training data and then merges it with the target data.
# 2. compute_zone_slot_demand : This function computes the mean demand for each zone for each time slot from the training data and then merges it with the target data.
# 3. compute_borough_demand : This function computes the mean demand for each borough from the training data and then merges it with the target data.
# 4. compute_lag_features : This function computes the lag features for each zone from the training data and then merges it with the target data.
# 5. compute_rolling_features : This function computes the rolling features for each zone from the training data and then merges it with the target data.
# 6. compute_all_features : This function computes all the features from the training data and then merges it with the target data.
# At last the get_feature_names is responsible for calling and feature engineering of all the above features.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import config


def compute_zone_demand_stats(target_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:

    df = target_df.copy()
    fallback = ref_df["demand"].mean()

    stats = ref_df.groupby("zone_id")["demand"].agg(["mean", "std", "max"]).reset_index()
    stats.columns = ["zone_id", "zone_demand_mean", "zone_demand_std", "zone_demand_max"]
    stats["zone_demand_std"] = stats["zone_demand_std"].fillna(0)

    df = df.merge(stats, on="zone_id", how="left")

    for col in ["zone_demand_mean", "zone_demand_std", "zone_demand_max"]:
        df[col] = df[col].fillna(fallback if "mean" in col else 0)

    return df


def compute_zone_slot_demand(target_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:

    df = target_df.copy()
    fallback = ref_df["demand"].mean()

    zs = ref_df.groupby(["zone_id", "time_slot"])["demand"].mean().reset_index()
    zs.columns = ["zone_id", "time_slot", "zone_slot_demand"]

    df = df.merge(zs, on=["zone_id", "time_slot"], how="left")
    df["zone_slot_demand"] = df["zone_slot_demand"].fillna(fallback)

    return df


def compute_borough_demand(target_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:

    df = target_df.copy()
    fallback = ref_df["demand"].mean()

    bd = ref_df.groupby("borough_enc")["demand"].mean().reset_index()
    bd.columns = ["borough_enc", "borough_demand_mean"]

    df = df.merge(bd, on="borough_enc", how="left")
    df["borough_demand_mean"] = df["borough_demand_mean"].fillna(fallback)

    return df


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["zone_id", "pickup_slot"]).reset_index(drop=True)

    for lag in [1, 4, 96]:
        col_name = f"demand_lag_{lag}"
        df[col_name] = df.groupby("zone_id")["demand"].shift(lag)
        zone_means = df.groupby("zone_id")["demand"].transform("mean")
        df[col_name] = df[col_name].fillna(zone_means)

    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["zone_id", "pickup_slot"]).reset_index(drop=True)

    for window in [8, 96]:
        col_name = f"demand_rolling_{window}"
        df[col_name] = (
            df.groupby("zone_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        zone_means = df.groupby("zone_id")["demand"].transform("mean")
        df[col_name] = df[col_name].fillna(zone_means)

    return df


def compute_all_features(
    target_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> pd.DataFrame:

    df = target_df.copy()

    df = compute_zone_demand_stats(df, ref_df)
    df = compute_zone_slot_demand(df, ref_df)
    df = compute_borough_demand(df, ref_df)

    df = compute_lag_features(df)
    df = compute_rolling_features(df)

    return df


def get_feature_names() -> list:
    return config.ALL_FEATURES
