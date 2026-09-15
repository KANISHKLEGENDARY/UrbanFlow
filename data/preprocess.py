"""
This code is responsible for cleaning and preprocessing the dataset. First the csv file containing the zone IDs of all the zones is loaded to identify each and every zone in human readable format. To handle a large dataset containing around 65M rows, we iterate over the rows in chunks to keep the memory usage low. In the next step the chunks that comes in a quality check is done that removes anything outside the window of Feb, Mar, Apr. It removes fares with negative or unrealistically high fares, removes the rows with with 0 miles or absurdly long distances, removes trips where the pickup or drop-off zone is "Unknown" (IDs 264 and 265), removes trips shorter than 1 minute or longer than 3 hours. Next the zones the being divided into the 15 minutes windows and the demand of drivers is being calculated and added to the dataset. In the next step the aggregate of all the chunks is being created according to the zone IDs and their pickups are also added. To avoid discontinuity in the dataset the code adds zeros demand to the timeslots that are not availabble for a particular zone. To avoid biasness in the predictions the aggragate dataset being created is being normalised between 0 and 1 such that in future, model does not learn that big number means more importance. Then the metadata like zone_name and borough are added to the file to make it more human friendly. The ML model can't read the strings like:- 08:15 so the code breaks it down into hour, minute, day_of_week, etc. Then at last the dataste is divided into train and test splits. The train data stricktly consists of February and March and the test data consists of April. This is done to make sure that the model is trained on the historical data and then it is being used for future predicton.
"""

import gc
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import config


DATE_TOKEN_RE = re.compile(r"(\d{4})-(\d{2})")


def get_target_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    periods = []
    for path in config.RAW_DATA_FILES:
        match = DATE_TOKEN_RE.search(path.name)
        if match:
            periods.append(pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M"))

    if periods:
        periods = sorted(periods)
        return periods[0].start_time, (periods[-1] + 1).start_time

    train_month = min(config.TRAIN_MONTHS + config.TEST_MONTHS)
    test_month = max(config.TRAIN_MONTHS + config.TEST_MONTHS)
    year = 2026
    return (
        pd.Timestamp(year=year, month=train_month, day=1),
        (pd.Period(f"{year}-{test_month:02d}", freq="M") + 1).start_time,
    )


def iter_raw_batches():
    found_any = False
    for path in config.RAW_DATA_FILES:
        if not path.exists():
            print(f"  [WARN] File not found, skipping: {path.name}")
            continue

        found_any = True
        parquet_file = pq.ParquetFile(path)
        print(
            f"  Reading {path.name}: {parquet_file.metadata.num_rows:,} rows "
            f"across {parquet_file.num_row_groups} row groups"
        )

        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=config.HVFHV_LOAD_COLUMNS,
            )
            df = table.to_pandas()
            print(
                f"    row group {row_group_idx + 1:02d}/{parquet_file.num_row_groups}: "
                f"{len(df):,} rows"
            )
            yield path, row_group_idx + 1, parquet_file.num_row_groups, df

    if not found_any:
        raise FileNotFoundError(
            "No HVFHV data files found! Expected files:\n"
            + "\n".join(f"  {p}" for p in config.RAW_DATA_FILES)
        )


def load_raw_data() -> pd.DataFrame:
    dfs = []
    for path in config.RAW_DATA_FILES:
        if not path.exists():
            print(f"  [WARN] File not found, skipping: {path.name}")
            continue
        print(f"  Loading {path.name}...")
        df = pd.read_parquet(path, columns=config.HVFHV_LOAD_COLUMNS)
        print(f"    → {len(df):,} rows")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No HVFHV data files found! Expected files:\n"
            + "\n".join(f"  {p}" for p in config.RAW_DATA_FILES)
        )

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows × {combined.shape[1]} columns")
    return combined


def load_zone_lookup() -> pd.DataFrame:
    if not config.ZONE_LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Zone lookup not found: {config.ZONE_LOOKUP_PATH}\n"
            f"Run 'python data/download_nyc_data.py' first."
        )
    zones = pd.read_csv(config.ZONE_LOOKUP_PATH)
    print(f"  Loaded zone lookup: {len(zones)} zones")
    return zones


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    col = config.HVFHV_COLUMNS

    df[col["pickup_datetime"]] = pd.to_datetime(df[col["pickup_datetime"]])
    df[col["dropoff_datetime"]] = pd.to_datetime(df[col["dropoff_datetime"]])

    min_date, max_date = get_target_date_bounds()
    df = df[
        (df[col["pickup_datetime"]] >= min_date) &
        (df[col["pickup_datetime"]] < max_date)
    ]
    after_date = len(df)
    print(f"  Date filter:     removed {original - after_date:,} rows")

    before = len(df)
    df = df[
        (df[col["fare"]] >= config.MIN_FARE) &
        (df[col["fare"]] <= config.MAX_FARE)
    ]
    print(f"  Fare filter:     removed {before - len(df):,} rows")

    before = len(df)
    df = df[
        (df[col["distance"]] >= config.MIN_TRIP_DISTANCE) &
        (df[col["distance"]] <= config.MAX_TRIP_DISTANCE)
    ]
    print(f"  Distance filter: removed {before - len(df):,} rows")

    before = len(df)
    df = df[
        ~df[col["pickup_location"]].isin(config.INVALID_LOCATION_IDS) &
        ~df[col["dropoff_location"]].isin(config.INVALID_LOCATION_IDS)
    ]
    print(f"  Zone filter:     removed {before - len(df):,} rows")

    before = len(df)
    df = df[
        (df[col["duration_sec"]] >= config.MIN_TRIP_DURATION_SEC) &
        (df[col["duration_sec"]] <= config.MAX_TRIP_DURATION_SEC)
    ]
    print(f"  Duration filter: removed {before - len(df):,} rows")

    df = df.copy()
    retained_pct = (len(df) / original * 100) if original else 0
    print(f"  Clean shape:     {len(df):,} rows ({retained_pct:.1f}% retained)")
    return df


def aggregate_demand(df: pd.DataFrame) -> pd.DataFrame:
    col = config.HVFHV_COLUMNS
    df = df.copy()

    df["pickup_slot"] = df[col["pickup_datetime"]].dt.floor(config.AGGREGATION_FREQ)

    demand = (
        df.groupby([col["pickup_location"], "pickup_slot"])
        .agg(
            demand=(col["pickup_location"], "size"),
            fare_sum=(col["fare"], "sum"),
            distance_sum=(col["distance"], "sum"),
            duration_sum=(col["duration_sec"], "sum"),
        )
        .reset_index()
    )
    demand.rename(columns={col["pickup_location"]: "zone_id"}, inplace=True)

    df["dropoff_slot"] = df[col["dropoff_datetime"]].dt.floor(config.AGGREGATION_FREQ)
    supply = (
        df.groupby([col["dropoff_location"], "dropoff_slot"])
        .size()
        .reset_index(name="supply")
    )
    supply.rename(
        columns={col["dropoff_location"]: "zone_id", "dropoff_slot": "pickup_slot"},
        inplace=True,
    )

    demand = demand.merge(supply, on=["zone_id", "pickup_slot"], how="left")
    demand["supply"] = demand["supply"].fillna(0).astype(int)

    print(
        f"  Aggregated batch: {len(demand):,} zone-slot records, "
        f"{demand['zone_id'].nunique()} zones"
    )
    return demand


def combine_demand_aggregates(aggregates: list[pd.DataFrame]) -> pd.DataFrame:
    if not aggregates:
        raise ValueError("No demand aggregates were produced from the raw files.")

    demand = pd.concat(aggregates, ignore_index=True)
    demand = (
        demand.groupby(["zone_id", "pickup_slot"], as_index=False)
        .agg(
            demand=("demand", "sum"),
            fare_sum=("fare_sum", "sum"),
            distance_sum=("distance_sum", "sum"),
            duration_sum=("duration_sum", "sum"),
            supply=("supply", "sum"),
        )
    )

    demand["avg_fare"] = np.where(
        demand["demand"] > 0,
        demand["fare_sum"] / demand["demand"],
        0,
    )
    demand["avg_distance"] = np.where(
        demand["demand"] > 0,
        demand["distance_sum"] / demand["demand"],
        0,
    )
    demand["avg_duration_min"] = np.where(
        demand["demand"] > 0,
        demand["duration_sum"] / demand["demand"] / 60,
        0,
    )
    demand["supply"] = demand["supply"].fillna(0).astype(int)
    demand["demand_supply_gap"] = demand["demand"] - demand["supply"]
    demand = demand[
        [
            "zone_id", "pickup_slot", "demand",
            "avg_fare", "avg_distance", "avg_duration_min",
            "supply", "demand_supply_gap",
        ]
    ]

    print(f"  Combined aggregate: {len(demand):,} observed zone-slot records")
    print(f"  Observed zones: {demand['zone_id'].nunique()}")
    print(f"  Observed time slots: {demand['pickup_slot'].nunique()}")
    return demand


def complete_zone_slot_grid(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    min_date, max_date = get_target_date_bounds()
    slot_end = max_date - pd.Timedelta(minutes=15)
    all_slots = pd.date_range(min_date, slot_end, freq=config.AGGREGATION_FREQ)
    valid_zones = sorted(
        zones.loc[
            ~zones["LocationID"].isin(config.INVALID_LOCATION_IDS),
            "LocationID",
        ].astype(int).unique()
    )

    full_index = pd.MultiIndex.from_product(
        [valid_zones, all_slots],
        names=["zone_id", "pickup_slot"],
    ).to_frame(index=False)

    df = full_index.merge(df, on=["zone_id", "pickup_slot"], how="left")
    fill_zero_cols = [
        "demand", "avg_fare", "avg_distance", "avg_duration_min",
        "supply", "demand_supply_gap",
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)
    df["demand"] = df["demand"].astype(int)
    df["supply"] = df["supply"].astype(int)
    df["demand_supply_gap"] = df["demand"] - df["supply"]

    print(f"  Completed grid: {len(df):,} zone-slot records")
    print(f"  Grid zones: {len(valid_zones)}")
    print(f"  Grid time slots: {len(all_slots)}")
    return df


def normalize_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zone_max = df.groupby("zone_id")["demand"].transform("max")
    df["demand_raw"] = df["demand"]  # keep original count
    df["demand"] = df["demand"] / zone_max.clip(lower=1)  # avoid div by 0
    df["demand"] = df["demand"].clip(0, 1)

    print(f"  Demand stats after normalization:")
    print(f"    Min:    {df['demand'].min():.4f}")
    print(f"    Mean:   {df['demand'].mean():.4f}")
    print(f"    Median: {df['demand'].median():.4f}")
    print(f"    Max:    {df['demand'].max():.4f}")
    return df


def add_zone_metadata(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        zones[["LocationID", "Borough", "Zone", "service_zone"]],
        left_on="zone_id",
        right_on="LocationID",
        how="left",
    )
    df.drop(columns=["LocationID"], inplace=True)
    df.rename(columns={"Borough": "borough", "Zone": "zone_name"}, inplace=True)

    borough_centroids = {
        "Manhattan": (40.7831, -73.9712),
        "Brooklyn": (40.6782, -73.9442),
        "Queens": (40.7282, -73.7949),
        "Bronx": (40.8448, -73.8648),
        "Staten Island": (40.5795, -74.1502),
        "EWR": (40.6895, -74.1745),
    }

    def get_centroid(row):
        base = borough_centroids.get(row["borough"], (40.7128, -74.0060))
        offset_lat = (row["zone_id"] % 50 - 25) * 0.002
        offset_lng = (row["zone_id"] % 37 - 18) * 0.002
        return base[0] + offset_lat, base[1] + offset_lng

    centroids = df[["zone_id", "borough"]].drop_duplicates().apply(
        lambda r: pd.Series(get_centroid(r), index=["zone_lat", "zone_lng"]),
        axis=1,
    )
    centroid_map = df[["zone_id", "borough"]].drop_duplicates().copy()
    centroid_map[["zone_lat", "zone_lng"]] = centroids.values

    df = df.merge(centroid_map[["zone_id", "zone_lat", "zone_lng"]], on="zone_id", how="left")

    print(f"  Added zone metadata: {df['borough'].nunique()} boroughs")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["pickup_slot"].dt.hour
    df["minute_of_hour"] = df["pickup_slot"].dt.minute           
    df["time_slot"] = df["hour"] * config.SLOTS_PER_HOUR + df["minute_of_hour"] // 15  
    df["day_of_week"] = df["pickup_slot"].dt.dayofweek 
    df["day_of_month"] = df["pickup_slot"].dt.day
    df["month"] = df["pickup_slot"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)


    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / config.SLOTS_PER_DAY)
    df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / config.SLOTS_PER_DAY)

    n_features = 16  
    print(f"  Added {n_features} time features (incl. time_slot 0–95, slot_sin/cos)")
    return df


def encode_borough(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    borough_map = {
        "Manhattan": 0, "Brooklyn": 1, "Queens": 2,
        "Bronx": 3, "Staten Island": 4, "EWR": 5,
    }
    df["borough_enc"] = df["borough"].map(borough_map).fillna(5).astype(int)
    return df


def split_train_test(df: pd.DataFrame):
    train = df[df["month"].isin(config.TRAIN_MONTHS)].copy()
    test = df[df["month"].isin(config.TEST_MONTHS)].copy()

    train_months = ", ".join(str(m) for m in config.TRAIN_MONTHS)
    test_months = ", ".join(str(m) for m in config.TEST_MONTHS)
    print(f"  Train: {len(train):,} rows (months {train_months})")
    print(f"  Test:  {len(test):,} rows (months {test_months})")
    return train, test


def main():
    print("=" * 60)
    print("UrbanFlow — Data Preprocessing Pipeline v2.0")
    print("  Dataset: NYC TLC HVFHV (Uber/Lyft)")
    print("  Aggregation: 15-minute windows")
    print("=" * 60)

    # Step 1: Load zone lookup
    print("\n[1/8] Loading zone lookup...")
    zones = load_zone_lookup()

    # Step 2: Stream raw trips and aggregate by row group
    print("\n[2/8] Streaming raw HVFHV trip data...")
    aggregates = []
    total_clean_rows = 0
    for _, _, _, trips in iter_raw_batches():
        print("  Cleaning row group...")
        trips = clean_trips(trips)
        total_clean_rows += len(trips)

        if not trips.empty:
            print("  Aggregating row group...")
            aggregates.append(aggregate_demand(trips))

        del trips
        gc.collect()

    print(f"  Total clean trips retained: {total_clean_rows:,}")

    # Step 3: Combine row-group aggregates
    print("\n[3/8] Combining 15-min demand aggregates...")
    demand = combine_demand_aggregates(aggregates)
    del aggregates
    gc.collect()

    demand["month"] = demand["pickup_slot"].dt.month
    # Step 4: Complete zero-demand zone-slot rows
    print("\n[4/8] Completing zone-slot grid...")
    demand = complete_zone_slot_grid(demand, zones)

    # Free memory — raw trips no longer needed
    # Raw row-group data is already released inside the streaming loop.
    gc.collect()
    print("  Memory cleanup complete")

    # Step 5: Normalize
    print("\n[5/8] Normalizing demand to [0, 1]...")
    demand = normalize_demand(demand)

    # Step 6: Add metadata
    print("\n[6/8] Adding zone metadata...")
    demand = add_zone_metadata(demand, zones)

    # Step 7: Time features + encoding
    print("\n[7/8] Extracting time features...")
    demand = extract_time_features(demand)
    demand = encode_borough(demand)

    # Step 8: Save
    print("\n[8/8] Saving processed data...")

    # Save full dataset
    demand.to_parquet(config.PROCESSED_DEMAND_PATH, index=False)
    print(f"  Full dataset: {config.PROCESSED_DEMAND_PATH}")
    print(f"  Shape: {demand.shape}")

    # Train/test split
    train, test = split_train_test(demand)
    train.to_parquet(config.PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(config.PROCESSED_DIR / "test.parquet", index=False)
    print(f"  Train saved: {config.PROCESSED_DIR / 'train.parquet'}")
    print(f"  Test saved:  {config.PROCESSED_DIR / 'test.parquet'}")

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Preprocessing complete!")
    print(f"  Total zone-slot records: {len(demand):,}")
    print(f"  Zones: {demand['zone_id'].nunique()}")
    print(f"  Date range: {demand['pickup_slot'].min()} to {demand['pickup_slot'].max()}")
    print(f"  Columns: {list(demand.columns)}")
    print(f"\n  Next step: python ml/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
