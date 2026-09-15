# Understanding the UrbanFlow Dashboard — A Complete Guide

---

## Phase 2 Upgrade Documentation: HVFHV + 15-Minute Models

This project has now been updated through **Phase 2 only** of `implementation_plan_v2.md`. No Phase 3 ETA/OSMNX work, PostgreSQL work, authentication, Docker, or monitoring changes were added.

### What Changed

UrbanFlow moved from the original hourly Yellow Taxi training setup to a Phase 2 ride-hailing setup based on NYC TLC **HVFHV** data, which represents high-volume for-hire vehicle trips such as Uber and Lyft.

The raw inputs now come from:

- `fhvhv_tripdata_2026-02.parquet`
- `fhvhv_tripdata_2026-03.parquet`
- `fhvhv_tripdata_2026-04.parquet`

The model now predicts demand for **15-minute windows** instead of 1-hour windows. Each day is represented as 96 slots:

- `00:00` = slot 0
- `00:15` = slot 1
- `00:30` = slot 2
- `00:45` = slot 3
- `17:15` = slot 69
- `23:45` = slot 95

The generated Phase 2 dataset now contains:

| Output | Rows | Date Range | Purpose |
|---|---:|---|---|
| `data/processed/demand_15min.parquet` | 2,247,072 | Feb 1-Apr 30, 2026 | Full 15-minute zone-slot dataset |
| `data/processed/train.parquet` | 1,489,632 | Feb 1-Mar 31, 2026 | Model training |
| `data/processed/test.parquet` | 757,440 | Apr 1-Apr 30, 2026 | Future-month evaluation |

All 263 valid TLC zones are represented for every 15-minute slot. If a zone had no pickups in a slot, the preprocessor writes an explicit zero-demand row. This matters because lag and rolling features should move across real 15-minute intervals, not only across slots where trips happened.

### How Preprocessing Works Now

The file `data/preprocess.py` is the main data pipeline. It reads the large HVFHV parquet files by **row group** instead of loading all three months into memory at once. Each row group is cleaned, aggregated, and released before the next row group is loaded.

The cleaning stage keeps only realistic trips:

- pickup dates within the configured raw file months
- fare between configured minimum and maximum bounds
- trip distance within configured bounds
- pickup and dropoff zones excluding invalid `264` and `265`
- native HVFHV `trip_time` between 1 minute and 3 hours

After cleaning, trips are grouped by pickup zone and `pickup_slot`, where `pickup_slot` is pickup time floored to the nearest 15 minutes.

For each zone-slot, the pipeline computes normalized demand, raw pickup count, average fare, average distance, average duration, dropoff-based supply proxy, demand-supply gap, zone metadata, approximate coordinates, and time features.

The core target is still normalized per zone:

```text
demand = pickup_count_for_slot / max_pickup_count_for_that_zone
```

This means the model predicts how busy a zone is relative to its own historical maximum, which keeps smaller zones meaningful instead of letting Manhattan dominate every map.

### New Time Features

The Phase 2 model uses 29 features. The major additions are:

| Feature | Meaning |
|---|---|
| `minute_of_hour` | One of `0`, `15`, `30`, `45` |
| `time_slot` | Daily slot from `0` to `95` |
| `slot_sin`, `slot_cos` | Cyclical encoding for the 96-slot day |
| `zone_slot_demand` | Historical average demand for a zone at a specific 15-minute slot |
| `demand_lag_1` | Same-zone demand 15 minutes ago |
| `demand_lag_4` | Same-zone demand 1 hour ago |
| `demand_lag_96` | Same-zone demand at the same slot yesterday |
| `demand_rolling_8` | Past 2-hour rolling demand |
| `demand_rolling_96` | Past 24-hour rolling demand |

`ml/features.py` computes fold-safe historical statistics from the training reference frame only. This prevents validation or test data from leaking into zone averages. Rolling features now use prior rows only, so the current target value is not included in its own rolling average.

### How Training Works After Phase 2

The file `ml/train.py` trains the same two-model ensemble strategy:

- XGBoost
- LightGBM

The completed Phase 2 training results are:

| Metric | Value |
|---|---:|
| XGBoost OOF R2 | 0.7454 |
| LightGBM OOF R2 | 0.7637 |
| Simple Average OOF R2 | 0.7559 |
| Optimized Blend OOF R2 | 0.7637 |
| Stacked OOF R2 | 0.7885 |
| Test R2 | 0.8239 |
| Test RMSE | 0.0706 |
| Test Score | 82.39 / 100 |

The serving ensemble currently saves the optimized blend weights:

```text
XGBoost = 0.000
LightGBM = 1.000
```

LightGBM performed best for the saved weighted-serving path. Stacking scored highest in cross-validation, but the existing API serving design uses the weighted blender artifact, so Phase 2 stops at the planned retrain/update scope without adding a new serving architecture.

### How API Prediction Works Now

The API now accepts quarter-hour timing:

```json
{
  "zone_id": 161,
  "hour": 17,
  "minute": 15,
  "day_of_week": 2,
  "day_of_month": 15,
  "month": 3
}
```

`api/schemas.py` validates that `minute` is exactly one of:

```text
0, 15, 30, 45
```

The demand service converts `hour` and `minute` into:

```text
time_slot = hour * 4 + minute / 15
```

Then it builds the 29-feature vector in the exact order expected by the trained models. For real-time lag values, the API uses historical zone-slot averages as approximations because the project does not yet include a live streaming pipeline. That is still within Phase 2; Phase 4+ infrastructure was intentionally not added.

The service also validates model compatibility at startup. If an old 24-feature model is present, the API refuses to mark models as loaded because Phase 2 expects 29 features.

### How The Dashboard Works Now

The React dashboard now has a **96-position time slider** instead of a 24-hour slider. The slider label displays human-readable quarter-hour time, such as `8:00 AM`, `8:15 AM`, `5:30 PM`, or `11:45 PM`.

Whenever the slider changes, the frontend sends both `hour` and `minute` to:

- `/api/v1/demand/heatmap`
- `/api/v1/pricing/calculate`
- `/api/v1/explain/zone/{zone_id}`

This keeps the heatmap, surge pricing panel, and SHAP explanation synchronized to the same 15-minute prediction window.

### Phase 2 Verification Completed

The following checks were completed after the update:

- Python modules compile successfully.
- `data/preprocess.py` generated the 15-minute processed parquet files.
- Full 5-fold training completed.
- Final XGBoost and LightGBM model files were saved with 29 input features.
- SHAP plots were regenerated.
- Demand service loaded the Phase 2 models successfully.
- A real prediction for zone 161 at `17:15` returned a valid result.
- Backend route functions returned heatmap, pricing, and explanation outputs with minute support.
- Invalid minute values are rejected by schema validation.
- `npm.cmd run build` completed successfully for the React frontend.

### Files Updated For Phase 2

| File | Main Change |
|---|---|
| `config.py` | HVFHV files, 15-minute aggregation, 29-feature list |
| `data/download_nyc_data.py` | Can stream-download missing HVFHV parquet files |
| `data/preprocess.py` | Streaming HVFHV row-group preprocessing and full 15-minute zone-slot grid |
| `ml/features.py` | 15-minute slot features, lag features, rolling features |
| `ml/train.py` | Saves core model artifacts before optional SHAP plotting |
| `ml/explainer.py` | Uses a temp Matplotlib cache path for local plotting reliability |
| `ml/lightgbm_model.py` | Suppresses harmless LightGBM feature-name warning during API prediction |
| `api/schemas.py` | Adds and validates quarter-hour `minute` fields |
| `api/routes/demand.py` | Returns selected minute in heatmap response |
| `api/routes/pricing.py` | Sends minute/month/day fields into demand prediction |
| `api/routes/explain.py` | Explains the selected quarter-hour prediction |
| `api/services/demand_service.py` | Builds 29-feature Phase 2 vectors and rejects stale models |
| `api/services/pricing_service.py` | Describes surge factors at exact quarter-hour windows |
| `api/main.py` | Version metadata updated to v2.0 |
| `frontend/src/App.jsx` | 96-slot time slider and minute-aware API calls |
| `frontend/src/App.css` | Wider time label for quarter-hour display |
| `outputs/models/*` | Retrained Phase 2 model artifacts |
| `outputs/shap/*` | Regenerated SHAP plots |
| `outputs/reports/training_report.txt` | Updated Phase 2 training metrics |

---

## Part 1: The Big Picture — What Problem Does This Dashboard Solve?

Imagine you are running Uber or Ola in New York City. You have thousands of drivers and millions of riders. Every single hour of every day, you need to answer these questions:

- **Where** in the city do we need more drivers right now?
- **How many** riders are likely to request rides in each area?
- **Should we increase prices** in a particular area because demand is too high?
- **Why** is demand high in a particular area? Is it because of rush hour? Is it because that area is always busy? Is it a weekend night effect?

This dashboard answers ALL of those questions visually. It takes three months of real NYC TLC HVFHV ride-hailing data from February-April 2026, trains an AI model to learn 15-minute demand patterns, and then lets you explore predictions for any quarter-hour slot of any day.

**Who uses dashboards like these in the real world?**

- **Uber/Lyft/Ola Operations Teams** — They look at demand heatmaps to decide where to send "incentive notifications" to drivers. If Midtown Manhattan is predicted to be at 90% capacity at 5 PM, they push notifications to drivers in Brooklyn saying "Head to Midtown — high demand expected in 30 minutes."

- **Pricing Teams** — They use surge pricing panels to set dynamic pricing rules. If demand is predicted to exceed supply, prices go up to attract more drivers and slightly reduce rider demand until equilibrium is reached.

- **City Transportation Planners** — They use demand data to decide where to add bus routes, build metro stations, or allow more taxi licenses.

- **Airport Operations** — JFK Airport managers look at predicted demand to decide how many taxis to queue at the arrivals terminal at different hours.

Now let me explain every single thing you see on the screen.

---

## Part 2: The Zone Demand Heatmap — The Percentage Cards

### What is a "Zone"?

New York City is officially divided into **263 taxi zones** by the NYC Taxi and Limousine Commission. These are not just random squares on a map — they correspond to actual neighborhoods that people recognize and use in daily life.

For example:
- Zone 161 = **Midtown Center** (the area around Times Square, Grand Central Station)
- Zone 138 = **LaGuardia Airport**
- Zone 132 = **JFK Airport**
- Zone 79 = **East Village** (a popular nightlife neighborhood)
- Zone 234 = **Union Square** (a major transit hub)

In your screenshot, you can see zones like "Highbridge Park," "Inwood Hill Park," "Roosevelt Island," "Penn Station/Madis," "Times Sq/Theatre D" — these are all real, recognized neighborhoods in Manhattan.

The dashboard shows **249 zones** (out of 263) because 14 zones were filtered out during data processing — they were either labeled "Unknown" in the official data or had too few trips to be meaningful.

### What does the percentage mean?

This is the most important number on the entire dashboard. Let me explain it step by step with a concrete example.

**Think of it like a "busyness meter" for each neighborhood, relative to that neighborhood's own history.**

Let's say you're looking at **Penn Station** (one of the busiest transit hubs in the world). Over the entire month of January 2023, the busiest hour Penn Station ever had was **600 taxi pickups in a single hour**. That happened on a Tuesday at 5:30 PM when everyone was leaving work and trying to catch trains home.

Now, the AI model predicts that **for the hour and day you've selected on the dashboard controls**, Penn Station will have approximately **312 pickups**.

The percentage is: `312 / 600 = 52%`

That means: **"Penn Station is predicted to be at 52% of its own busiest-ever level."**

Now compare this with **Highbridge Park** (a quiet park in northern Manhattan). Its busiest hour ever had only **3 taxi pickups**. If the model predicts **3 pickups** for the selected time, its percentage is `3/3 = 100%`.

**This is why a small neighborhood can show 100% while Times Square shows 50%.** The percentage is NOT comparing zones against each other — it's comparing each zone against its own personal best. Highbridge Park at 100% means "this park is as busy as it ever gets," while Times Square at 50% means "Times Square is only half as busy as it can get."

### Why is this design useful in the real world?

If you were an Uber operations manager and all zones were compared against each other (global normalization), the dashboard would be useless. Manhattan would always be red (high) and Staten Island would always be green (low). You'd learn nothing new.

But with per-zone normalization, you can spot **unusual patterns**:
- "Wait, Roosevelt Island is at 63%? That's unusual for a Tuesday afternoon. Let me check — oh, there's a tech conference on the island today."
- "Highbridge Park at 100%? There must be an event at the park. Let's send extra drivers to that area."
- "Times Square at only 50% on a Friday evening? That's lower than expected. Maybe there's a street closure affecting pickups."

The dashboard becomes a tool for **detecting anomalies and making decisions**, not just seeing the obvious (Manhattan is busy).

### What do the colors mean?

The color of the percentage text and the thin colored bar at the top of each card tell you the **demand severity level** at a glance:

| Color | Level | Range | What it means in practice |
|-------|-------|-------|--------------------------|
| 🟢 **Green** | Low | 0% – 24% | This zone is quiet. Plenty of available drivers. Normal pricing. No action needed. |
| 🟡 **Yellow** | Moderate | 25% – 49% | Normal activity. Some demand but supply is keeping up. Keep monitoring. |
| 🟠 **Orange** | High | 50% – 74% | Demand is elevated. You might want to start nudging drivers toward this area. Consider light surge pricing. |
| 🔴 **Red** | Critical | 75% – 100% | This zone is at or near peak capacity. Riders will experience long wait times. Surge pricing should be active. Send driver incentives immediately. |

**Real-world example:**

Imagine it's New Year's Eve at 11:45 PM. You open the dashboard. The entire Manhattan section is **red/critical** — every zone is at 80-100%. Meanwhile, residential areas in Staten Island are **green/low** at 10-15%. 

As an operations manager, you would:
1. Activate surge pricing across Manhattan (to attract more drivers with higher earnings)
2. Send push notifications to drivers in Brooklyn and Queens: "Head to Manhattan — 3x surge pricing active"
3. Alert the customer support team: "Expect complaints about wait times and high prices in Manhattan for the next 2 hours"

### What does "Manhattan 66 zones" mean?

The zones are **grouped by borough** (Manhattan, Brooklyn, Queens, Bronx, Staten Island). Manhattan alone contains 66 of the 249 zones. If you scroll down in the dashboard, you'd see separate sections for Brooklyn (zones like Williamsburg, Park Slope, DUMBO), Queens (zones like Astoria, Flushing, JFK Airport), and so on.

Within each borough, the zones are **sorted by demand percentage, highest first**. This means the zones that need the most attention are always at the top.

---

## Part 3: The Zone Detail Panel (Right Side)

When you click on any zone card in the heatmap, a detail panel appears on the right side of the screen. This panel has three major sections. Let me explain each one in exhaustive detail.

---

### Section A: Zone Header and Demand Gauge

At the top of the detail panel, you see:

```
Highbridge Park
Manhattan · Zone 45
                                    [CRITICAL]
```

- **Zone Name** ("Highbridge Park") — The official NYC taxi zone name. This is what local New Yorkers would recognize as a neighborhood name.

- **Borough · Zone ID** ("Manhattan · Zone 45") — The borough and the official numeric ID from the NYC Taxi and Limousine Commission database. Zone IDs range from 1 to 263.

- **Demand Level Badge** ("[CRITICAL]") — A colored label showing the severity level (same as the card colors: low/moderate/high/critical).

Below the header, there's a **Demand Gauge** — a horizontal progress bar:

```
Predicted Demand                    100%
[████████████████████████████████████]
```

This is the same percentage as on the card, but displayed as a visual bar. The bar fills from left to right, and its color transitions from green (on the left) to whatever the demand level color is (on the right). So a 100% critical zone would show a full bar going from green to red, while a 30% moderate zone would show a bar about one-third full going from green to yellow.

**Why show this when the percentage is already on the card?** Because when you're comparing zones, you often click between them quickly. The gauge gives you an instant visual sense of "how full" this zone is without needing to read the number.

---

### Section B: ⚡ Surge Pricing

This is the amber/yellow-bordered box that shows pricing information. This is where the dashboard moves from **analytics** (just showing data) to **actionable intelligence** (telling you what to DO about the data).

Let me walk through every field with a detailed real-world scenario.

**Scenario:** It's a Tuesday evening at 6 PM. You click on "Penn Station/Madison Square Garden" which is showing 65% demand. The surge pricing section shows:

```
⚡ Surge Pricing                    1.9x

$15.00 base → $28.50

High Demand                        +0.6x
Evening Rush                       +0.3x
```

Here's what every part means:

#### The Surge Multiplier (e.g., "1.9x")

This is the **recommended price multiplier**. If a normal ride costs $15, a 1.9x surge means the ride should cost $15 × 1.9 = $28.50.

**Why does surge pricing exist?** It's an economic mechanism to balance supply and demand in real time. When too many people want rides and there aren't enough drivers:
- **Higher prices discourage some riders** from taking rides (they might take the subway instead, or wait 30 minutes until the surge drops)
- **Higher prices attract more drivers** to that area (drivers see higher earnings and choose to drive toward surge zones)
- The result: wait times go down, and people who truly need a ride can get one (they're willing to pay more)

Without surge pricing, during peak demand, EVERYONE would request a ride at the same low price, and the app would just show "No drivers available" for hours.

#### The Base Fare and Adjusted Fare ("$15.00 base → $28.50")

- **$15.00** is the **base fare** — what the ride would normally cost without any surge
- **$28.50** is the **adjusted fare** — what the rider would actually pay: `$15.00 × 1.9 = $28.50`

In the dashboard, the base fare is set to $15.00 by default (you can change it in the API), but in a real Uber-like system, the base fare would be calculated from the actual trip distance and duration.

#### The Factor Breakdown

This is the part that makes this dashboard special. Most ride-hailing apps just show you "1.9x surge" and you have no idea why. This dashboard **explains every component** that went into that multiplier:

**Factor: "High Demand" → +0.6x**

This means: The predicted demand in this zone (65%) is in the "high" range (50%-75%). The pricing system adds 0.6x to the base multiplier because of this demand level.

**In practical terms:** "There are significantly more people requesting rides in this area than usual. This alone justifies a moderate price increase."

**Factor: "Evening Rush" → +0.3x**

This means: The selected hour (6 PM) falls within the evening commute window (5 PM – 7 PM). The pricing system adds an extra 0.3x because rush hours systematically have more demand than the AI model alone might capture.

**In practical terms:** "It's evening rush hour. People are leaving offices, trying to get home, heading to dinner. Historically, this time of day always has high demand regardless of the specific zone."

**How the total is calculated:**

```
Start with base:         1.0x  (no surge)
+ High Demand:          +0.6x
+ Evening Rush:         +0.3x
─────────────────────────────
Total:                   1.9x
```

The system caps the maximum at 3.0x, so even if all factors stack up, a rider never pays more than 3× the base fare.

#### All Possible Factors

Here are ALL the factors the system can show, with real-world explanations:

| Factor | When it appears | Impact | Real-world explanation |
|--------|----------------|--------|----------------------|
| **Normal Demand** | Demand is below 50% | +0.0x | "Things are quiet. No reason to charge extra." |
| **Moderate Demand** | Demand is 50%-ish | +0.3x | "Demand is picking up. A small price bump helps balance supply." |
| **High Demand** | Demand is 50%-75% | +0.6x | "This area is getting busy. We need to attract more drivers here." |
| **Critical Demand** | Demand is above 75% | +1.2x | "This area is maxed out. Significant price increase to manage demand and rush drivers over." |
| **Morning Rush** | Hour is 7, 8, or 9 AM | +0.3x | "Morning commute. Everyone is heading to work at the same time." |
| **Evening Rush** | Hour is 5, 6, or 7 PM | +0.3x | "Evening commute. Everyone is heading home or to dinner." |
| **Late Night** | Hour is 12 AM – 4 AM | +0.2x | "Very few drivers are active after midnight. The ones who ARE driving deserve a premium." |
| **Weekend Nightlife** | Friday/Saturday night (9 PM – 2 AM) | +0.2x | "Bars and restaurants are closing. Everyone wants a ride home at the same time." |

**A dramatic real-world example:**

It's Saturday night at 1 AM near the nightlife district in East Village. Demand is at 92% (critical). The breakdown would be:

```
⚡ Surge Pricing                    3.0x

$15.00 base → $45.00

Critical Demand                    +1.2x
Late Night                         +0.2x
Weekend Nightlife                  +0.2x
```

Total would be 1.0 + 1.2 + 0.2 + 0.2 = 2.6x. But even this doesn't hit the 3.0x cap, so the actual surge is 2.6x. The rider would pay $15 × 2.6 = $39.00 instead of $15.00.

If this was ALSO a critical demand zone AND all factors stacked up, it could theoretically reach `1.0 + 1.2 + 0.3 + 0.2 + 0.2 = 2.9x` — still under the 3.0x cap.

---

### Section C: 🧠 AI Explanation (SHAP)

This is the most technically interesting section and also the hardest to understand intuitively. Let me build up the explanation gradually.

#### The fundamental question this section answers:

**"The AI model predicted 65% demand for Penn Station at 6 PM on Tuesday. But WHY did it predict 65% instead of 30% or 90%? What information drove that specific number?"**

This is different from the surge pricing factors above. The surge pricing is a simple set of hand-written rules. The AI explanation tells you what the **machine learning model itself** learned from the data.

#### The concept: How SHAP explanations work (no math needed)

Think of the AI model as a chef cooking a dish. The final dish tastes a certain way (the demand prediction). SHAP analysis is like asking the chef: "How much did each ingredient contribute to the final taste?"

- The **base value** is like starting with plain water — the model's "average" prediction if it knew nothing about the specific zone or time. For this model, the base value is roughly 0.47 (meaning 47% is the average demand across all zones and hours).

- Each **feature** (ingredient) then pushes the prediction UP or DOWN from that average:
  - "Knowing it's Penn Station pushes it UP by +0.15" (Penn Station is historically busier than average)
  - "Knowing it's 6 PM pushes it UP by +0.08" (6 PM is a peak hour)
  - "Knowing it's a Tuesday pushes it DOWN by -0.03" (Tuesdays are slightly less busy than Wednesday/Thursday)
  - And so on, for all 29 features

- The final prediction = base_value + sum of all feature contributions

#### What you see on screen:

```
🧠 AI Explanation
Why this prediction?

zone_slot_demand    [████████████████]  +0.1825
zone_demand_mean    [██████████]        +0.0934
hour                [████████]          +0.0521
borough_enc         [████]              +0.0218
is_peak             [████]              -0.0312
demand_lag_1        [███]               +0.0189
```

Let me explain each row:

#### Feature: `zone_slot_demand` → +0.1825

**What it is:** "What is the historical average demand at this specific zone during this specific hour of day?"

**In plain English:** The model looked at all the Tuesdays at 6 PM at Penn Station in the training data and found that demand is historically high at this zone-hour combination. This single piece of information pushed the prediction UP by 0.1825 (about 18 percentage points above the average).

**Real-world significance:** This is almost always the #1 most important feature. It captures the basic pattern: "Midtown at 5 PM is always busy. Residential areas at 3 AM are always quiet." The model doesn't need to re-learn this every time — it memorized the pattern from training data.

**Practical example:** If you're building a ride-hailing service in a new city, and this feature is dominant, it tells you: "Your model mainly relies on learning historical patterns. If patterns change (e.g., a new office building opens), you need to retrain the model with recent data."

#### Feature: `zone_demand_mean` → +0.0934

**What it is:** "What is the overall average demand for this zone across ALL hours?"

**In plain English:** Penn Station is generally a high-demand zone regardless of time. Even at 3 AM, it has more activity than most zones at their peak. This "zone reputation" pushes the prediction UP by about 9 percentage points.

**The difference from `zone_slot_demand`:** `zone_slot_demand` knows that Penn Station at 6:15 PM is different from Penn Station at 3:00 AM. `zone_demand_mean` only knows that Penn Station is generally busy — it doesn't distinguish between time slots. Both pieces of information are useful: one captures time-specific patterns, the other captures the zone's baseline popularity.

#### Feature: `hour` → +0.0521

**What it is:** Just the numeric hour of day (0-23). In this case, 17 (5 PM) or 18 (6 PM).

**In plain English:** Evening hours in general tend to have higher demand across the city. Even without knowing which zone you're asking about, the model has learned that 6 PM is a busy time. This adds about 5 percentage points.

**Why is this less impactful than `zone_slot_demand`?** Because `zone_slot_demand` already captures most of the slot-specific information. The raw `hour` feature adds marginal extra signal — like the model saying "not only is Penn Station at 6:15 PM historically busy (`zone_slot_demand`), but also, 6 PM in general is a busy time across the whole city (`hour`)."

#### Feature: `borough_enc` → +0.0218

**What it is:** Which borough the zone is in (Manhattan=0, Brooklyn=1, Queens=2, Bronx=3, Staten Island=4).

**In plain English:** Being in Manhattan generally means higher demand than being in Staten Island. This pushes the prediction up by about 2 percentage points. It's a small effect because the zone-specific features (`zone_demand_mean`, `zone_slot_demand`) already capture most of the "Manhattan is busy" information.

#### Feature: `is_peak` → -0.0312

**What it is:** A binary flag (1 or 0) indicating whether this hour is a peak commute hour (7-9 AM or 5-7 PM).

**In plain English:** Wait — this is NEGATIVE? How can being at peak hour push the prediction DOWN? 

This happens because of **feature interaction**. The model has already accounted for the "peak hour" effect through `zone_slot_demand`, `time_slot`, and `hour`. The `is_peak` flag provides a slight correction: "After accounting for everything else, the simple peak/non-peak binary flag actually slightly overshoots — we need to pull back a tiny bit." This kind of negative correction on an intuitively "positive" feature is common in complex models and is one reason SHAP is so valuable — it reveals counterintuitive effects that you wouldn't see with simple feature importance.

#### Feature: `demand_lag_1` → +0.0189

**What it is:** "What was the demand at this zone 1 hour ago?"

**In plain English:** If Penn Station was busy at 5 PM, it's likely to still be busy at 6 PM. Demand patterns are "sticky" — they don't change dramatically from one hour to the next. This temporal momentum adds about 2 percentage points.

**Important caveat:** During training, this feature used the ACTUAL demand from the previous hour. But when the API serves predictions, it doesn't have real-time data, so it approximates with the historical average. This means the lag features are less accurate in the API than during training. In a production system like Uber, you'd connect this to a real-time data stream.

#### The Bar Lengths and Colors

- **Green bars** (pointing right) = this feature pushed the prediction **UP** (positive SHAP value)
- **Red bars** (pointing right) = this feature pushed the prediction **DOWN** (negative SHAP value)
- **Bar length** = proportional to the absolute impact. A bar twice as long means that feature had twice as much influence on the prediction.

---

## Part 4: The Other Dashboard Sections

### The Stats Cards (Top of Dashboard)

At the top of the dashboard, there are 4 summary cards showing city-wide metrics for the currently selected hour and day:

| Card | Example Value | What it means |
|------|--------------|---------------|
| **📊 Average Demand** | "46.8% / 249 zones" | Across all 249 zones in the city, the average predicted demand is 46.8%. This gives you a sense of overall city activity. On a quiet Sunday morning, this might be 15%. On a Friday evening, it might be 60%. |
| **🔴 Critical Zones** | "45 / 62 high demand" | 45 zones are at 75%+ demand (critical level), and an additional 62 are at 50-75% (high level). If you're an ops manager and this number is unusually high, you might call in extra drivers from your backup pool. |
| **⚡ Avg Surge** | "1.4x" | The average surge multiplier across all zones. If this is above 1.5x, it means the city as a whole is experiencing significant pricing pressure. |
| **🔥 Hottest Zone** | "Highbridge Park / 100% capacity" | The single zone with the highest predicted demand right now. This is your #1 priority — the area most likely to have long wait times and unhappy riders. |

**Real-world usage:** An Uber operations manager might have this dashboard on a wall-mounted screen in their operations center. They glance at the stats cards every few minutes. If "Critical Zones" suddenly jumps from 20 to 80, they know something is happening (a major event ending, a sudden rainstorm making people hail cabs) and need to take action.

### The Top Zones Leaderboard

On the right side, below the zone detail panel, there's a leaderboard showing the **Top 10 highest-demand zones**:

```
1. Highbridge Park       Manhattan    [████████████████] 100%
2. Inwood Hill Park      Manhattan    [█████████████]     86%
3. Roosevelt Island      Manhattan    [████████████]      63%
...
```

Each row shows:
- **Rank** (1-10)
- **Zone name** and **borough**
- **Demand bar** — visual bar colored by demand level
- **Percentage badge** — color-coded

**Real-world usage:** This is the "action list." If you're dispatching drivers, you'd look at this leaderboard and send drivers to the top 3-5 zones first. It's sorted by predicted demand so you can prioritize.

### The Feature Importance Chart (Bottom)

This is a horizontal bar chart showing **which features the model considers most important across ALL predictions** (not just one specific zone):

```
🏆 Global Feature Importance (XGBoost)

#1  zone_slot_demand    [████████████████████████████████]  0.342
#2  zone_demand_mean    [████████████████████████]          0.198
#3  demand_lag_1        [████████████████████]              0.156
#4  hour                [██████████████]                    0.089
...
```

**The difference from the SHAP explanation section:**
- The SHAP section (🧠 AI Explanation) shows feature importance **for one specific prediction** (e.g., Penn Station at 6 PM Tuesday)
- The Feature Importance chart shows importance **globally across ALL predictions** in the training data

**Why this matters:**

If `zone_slot_demand` is by far the most important feature globally, it tells you:
- "The model heavily relies on knowing historical demand patterns for each zone at each hour"
- "If you want to improve the model, getting more historical data for each zone would help more than adding weather data or event data"
- "If historical patterns change (e.g., a subway line closes and traffic patterns shift), the model will be slow to adapt because it's relying on old patterns"

This is the kind of insight that data scientists use to decide what to work on next.

---

## Part 5: The Time Controls

At the top of the dashboard, there are two controls:

### Hour Slider

A horizontal slider from 0 (midnight) to 23 (11 PM). When you drag this slider, **every single prediction on the entire dashboard updates instantly** — all 249 zone cards recalculate, the stats cards update, the leaderboard reshuffles, and if you have a zone selected, its surge pricing and SHAP explanation recalculate.

**Try this experiment:** Set the hour to 3 AM and watch the dashboard. Most zones will be green/low. Then slowly drag the slider to 8 AM — watch the zones turn yellow and orange as morning rush hour kicks in. Drag to 5 PM — many zones will turn red/critical. Drag to 11 PM — demand drops again except in nightlife areas like East Village and Lower East Side.

You're watching the **heartbeat of New York City** across the day.

### Day of Week Pills

Seven buttons: Mon, Tue, Wed, Thu, Fri, Sat, Sun. Clicking a different day updates all predictions.

**Try this experiment:** Set the hour to 11 PM. Click "Tuesday" — most zones are moderate. Now click "Saturday" — nightlife zones will jump to high/critical because Saturday nights are significantly busier than Tuesday nights. This shows you the **weekday vs. weekend pattern** that the model learned.

---

## Part 6: Putting It All Together — A Complete Real-World Scenario

**Scenario:** You're the operations manager at a ride-hailing company in NYC. It's a Friday evening. You open the UrbanFlow dashboard at 5:30 PM.

**Step 1: Check the big picture**
- Stats cards show: Average Demand = 62%, Critical Zones = 78, Avg Surge = 1.8x
- This is higher than usual. Friday evening rush is hitting hard.

**Step 2: Find the hotspots**
- The Top Zones leaderboard shows Penn Station, Times Square, Grand Central, and Midtown South all at 80%+ (critical)
- This makes sense — people are leaving offices and catching trains

**Step 3: Drill into the worst zone**
- You click "Penn Station" (85% demand)
- Surge Pricing shows 2.2x: Critical Demand (+1.2x) + Evening Rush (+0.3x)
- A $20 ride would cost $44
- AI Explanation shows `zone_slot_demand` is the #1 factor — Penn Station around 5 PM Friday is historically one of the busiest zone-slot combinations in the city

**Step 4: Take action**
- You send driver incentives to neighborhoods around Penn Station
- You alert the customer support team about high prices
- You notice that JFK Airport is at only 40% — flight arrivals are low right now. You redirect some airport-area drivers toward Midtown

**Step 5: Plan ahead**
- You drag the hour slider to 7 PM — demand at Penn Station drops to 55% as the commute wave passes
- You drag to 10 PM — nightlife zones in East Village and Lower East Side are predicted to rise to 70%
- You proactively position drivers in those areas BEFORE the demand hits

**This is the power of predictive demand intelligence.** You're not reacting to problems — you're anticipating them.
