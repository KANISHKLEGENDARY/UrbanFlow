# This file is responsible for creating making rough numbers into the actual output. This file first checks there are .pkl present for all the trained models. If yes then it proceeds for the predictions and if not then it shows error of model's file being not present. It also checks if the number of features on which the models are trained on are the same as required for the prediction. If not then this code asks for retraining of the models. Then this code calculates the statistical features of the zones (individually and for all zones also) on which the models are trained and the uses those features to make predictions. Then this model breaks down the request of the user into separate hour, minute, day of the week, month etc. and stores them in a list. Then after all of this the prediction is made for all the zones which is being shown as a heatmap on the dashboard. Surge is calculated based on predicted demand using linear scaling above a 0.5 threshold, with an additional peak-hour boost during rush hours All of the models are loaded once to ensure efficient memory utilisation as loading models of 500 mb every will take around 5 to 10 seconds and a delay will be there between the prediction's request and response. 

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import config


class DemandService:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.blender = None
        self.shap_explainer = None
        self.zone_lookup = None
        self.zone_stats = None
        self.is_loaded = False

    def load_models(self):
        try:
            self.xgb_model = joblib.load(config.MODEL_DIR / "xgboost_final.pkl")
            self.lgb_model = joblib.load(config.MODEL_DIR / "lightgbm_final.pkl")
            self.blender = joblib.load(config.MODEL_DIR / "ensemble_weights.pkl")

            self._validate_model_feature_counts()

            shap_path = config.MODEL_DIR / "shap_explainer.pkl"
            if shap_path.exists():
                self.shap_explainer = joblib.load(shap_path)

            if config.ZONE_LOOKUP_PATH.exists():
                self.zone_lookup = pd.read_csv(config.ZONE_LOOKUP_PATH)

            train_path = config.PROCESSED_DIR / "train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                self.zone_stats = self._compute_zone_stats(train_df)
                self._train_df = train_df

            self.is_loaded = True
            print("[DemandService] Models loaded successfully")
            return True

        except FileNotFoundError as e:
            print(f"[DemandService] Model files not found: {e}")
            print("[DemandService] Run 'python ml/train.py' first.")
            return False
        except Exception as e:
            print(f"[DemandService] Could not load models: {e}")
            print("[DemandService] Run 'python ml/train.py' to create Phase 2 model artifacts.")
            self.is_loaded = False
            return False

    def _validate_model_feature_counts(self) -> None:
        expected = len(config.ALL_FEATURES)
        for name, wrapper in {
            "XGBoost": self.xgb_model,
            "LightGBM": self.lgb_model,
        }.items():
            model = getattr(wrapper, "model", None)
            actual = getattr(model, "n_features_in_", None)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"{name} model expects {actual} features, but Phase 2 "
                    f"configuration defines {expected} features."
                )

    def _compute_zone_stats(self, train_df: pd.DataFrame) -> dict:
        stats = {}

        zone_means = train_df.groupby("zone_id")["demand"].mean().to_dict()
        zone_stds = train_df.groupby("zone_id")["demand"].std().fillna(0).to_dict()
        zone_maxs = train_df.groupby("zone_id")["demand"].max().to_dict()

        zone_slot_means = train_df.groupby(["zone_id", "time_slot"])["demand"].mean().to_dict()

        borough_means = train_df.groupby("borough_enc")["demand"].mean().to_dict()

        zone_meta = train_df.groupby("zone_id").agg({
            "zone_lat": "first",
            "zone_lng": "first",
            "borough_enc": "first",
            "borough": "first",
            "zone_name": "first",
        }).to_dict("index")

        global_mean = train_df["demand"].mean()

        raw_maxs = train_df.groupby("zone_id")["demand_raw"].max().to_dict() if "demand_raw" in train_df.columns else {}

        stats["zone_means"] = zone_means
        stats["zone_stds"] = zone_stds
        stats["zone_maxs"] = zone_maxs
        stats["zone_slot_means"] = zone_slot_means
        stats["borough_means"] = borough_means
        stats["zone_meta"] = zone_meta
        stats["global_mean"] = global_mean
        stats["raw_maxs"] = raw_maxs

        return stats

    def _build_features(
        self, zone_id: int, hour: int, day_of_week: int,
        day_of_month: int = 15, minute: int = 0, month: int = 3,
    ) -> np.ndarray:

        time_slot = hour * config.SLOTS_PER_HOUR + minute // 15  
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
        is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        slot_sin = np.sin(2 * np.pi * time_slot / config.SLOTS_PER_DAY)
        slot_cos = np.cos(2 * np.pi * time_slot / config.SLOTS_PER_DAY)

        meta = self.zone_stats["zone_meta"].get(zone_id, {})
        zone_lat = meta.get("zone_lat", 40.7128)
        zone_lng = meta.get("zone_lng", -74.006)
        borough_enc = meta.get("borough_enc", 0)

        zone_demand_mean = self.zone_stats["zone_means"].get(zone_id, self.zone_stats["global_mean"])
        zone_demand_std = self.zone_stats["zone_stds"].get(zone_id, 0)
        zone_demand_max = self.zone_stats["zone_maxs"].get(zone_id, self.zone_stats["global_mean"])
        zone_slot_demand = self.zone_stats["zone_slot_means"].get((zone_id, time_slot), zone_demand_mean)
        borough_demand_mean = self.zone_stats["borough_means"].get(borough_enc, self.zone_stats["global_mean"])

        prev_slot = (time_slot - 1) % config.SLOTS_PER_DAY
        prev_hour_slot = (time_slot - 4) % config.SLOTS_PER_DAY
        demand_lag_1 = self.zone_stats["zone_slot_means"].get((zone_id, prev_slot), zone_demand_mean)
        demand_lag_4 = self.zone_stats["zone_slot_means"].get((zone_id, prev_hour_slot), zone_demand_mean)
        demand_lag_96 = zone_slot_demand  # same slot yesterday ≈ same avg

        demand_rolling_8 = zone_demand_mean
        demand_rolling_96 = zone_demand_mean

        features = np.array([
            hour, minute, time_slot,
            day_of_week, day_of_month, month,
            is_weekend, is_peak, is_night,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            slot_sin, slot_cos,
            zone_id, borough_enc, zone_lat, zone_lng,
            zone_demand_mean, zone_demand_std, zone_demand_max,
            zone_slot_demand, borough_demand_mean,
            demand_lag_1, demand_lag_4, demand_lag_96,
            demand_rolling_8, demand_rolling_96,
        ], dtype=np.float64)

        return features

    def predict_zone(
        self, zone_id: int, hour: int, day_of_week: int,
        day_of_month: int = 15, minute: int = 0, month: int = 3,
    ) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        features = self._build_features(zone_id, hour, day_of_week, day_of_month, minute, month)
        X = features.reshape(1, -1)

        xgb_pred = self.xgb_model.predict(X)[0]
        lgb_pred = self.lgb_model.predict(X)[0]
        ensemble_pred = self.blender.predict([np.array([xgb_pred]), np.array([lgb_pred])])[0]
        ensemble_pred = float(np.clip(ensemble_pred, 0, 1))

        meta = self.zone_stats["zone_meta"].get(zone_id, {})
        zone_name = meta.get("zone_name", f"Zone {zone_id}")
        borough = meta.get("borough", "Unknown")

        raw_max = self.zone_stats["raw_maxs"].get(zone_id, 100)
        demand_raw = ensemble_pred * raw_max

        surge = self._compute_surge(ensemble_pred, hour)

        if ensemble_pred < 0.25:
            level = "low"
        elif ensemble_pred < 0.5:
            level = "moderate"
        elif ensemble_pred < 0.75:
            level = "high"
        else:
            level = "critical"

        return {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "borough": borough,
            "predicted_demand": round(ensemble_pred, 4),
            "demand_raw": round(demand_raw, 1),
            "surge_multiplier": round(surge, 2),
            "demand_level": level,
        }

    def predict_all_zones(
        self, hour: int, day_of_week: int,
        day_of_month: int = 15, minute: int = 0, month: int = 3,
    ) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded.")

        all_zone_ids = sorted(self.zone_stats["zone_meta"].keys())
        results = []

        feature_list = []
        for zid in all_zone_ids:
            feature_list.append(self._build_features(zid, hour, day_of_week, day_of_month, minute, month))

        X = np.array(feature_list)
        xgb_preds = self.xgb_model.predict(X)
        lgb_preds = self.lgb_model.predict(X)
        ensemble_preds = self.blender.predict([xgb_preds, lgb_preds])

        for i, zid in enumerate(all_zone_ids):
            pred = float(np.clip(ensemble_preds[i], 0, 1))
            meta = self.zone_stats["zone_meta"].get(zid, {})

            level = "low" if pred < 0.25 else "moderate" if pred < 0.5 else "high" if pred < 0.75 else "critical"

            results.append({
                "zone_id": zid,
                "zone_name": meta.get("zone_name", f"Zone {zid}"),
                "borough": meta.get("borough", "Unknown"),
                "lat": meta.get("zone_lat", 40.7128),
                "lng": meta.get("zone_lng", -74.006),
                "predicted_demand": round(pred, 4),
                "demand_level": level,
                "surge_multiplier": round(self._compute_surge(pred, hour), 2),
            })

        return results

    def _compute_surge(self, demand: float, hour: int) -> float:
        if demand <= config.SURGE_THRESHOLD:
            return config.SURGE_BASE

        scale = (demand - config.SURGE_THRESHOLD) / (1.0 - config.SURGE_THRESHOLD)

        peak_boost = 0.2 if hour in [7, 8, 9, 17, 18, 19] else 0.0

        surge = config.SURGE_BASE + scale * (config.SURGE_MAX - config.SURGE_BASE) + peak_boost
        return min(surge, config.SURGE_MAX)

    def get_zone_list(self) -> list[dict]:
        if not self.zone_stats:
            return []
        zones = []
        for zid, meta in sorted(self.zone_stats["zone_meta"].items()):
            zones.append({
                "zone_id": zid,
                "zone_name": meta.get("zone_name", f"Zone {zid}"),
                "borough": meta.get("borough", "Unknown"),
                "lat": meta.get("zone_lat", 40.7128),
                "lng": meta.get("zone_lng", -74.006),
            })
        return zones

demand_service = DemandService()
