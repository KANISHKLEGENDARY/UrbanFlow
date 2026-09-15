import pandas as pd

demand = pd.read_parquet("data/processed/demand_15min.parquet")

zone_max_full = demand.groupby("zone_id")["demand_raw"].max()
train_only = demand[demand["month"].isin([2, 3])]
zone_max_train_only = train_only.groupby("zone_id")["demand_raw"].max()

mismatch = (zone_max_full != zone_max_train_only).sum()
print(f"{mismatch} zones would have differed under the old (leaky) logic")
