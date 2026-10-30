"""Build the final dataset: Favorita sales + stores + weather + oil + holidays + date features."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data" / "raw"
OUT = ROOT / "Data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load ----------
train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
stores = pd.read_csv(RAW / "stores.csv").rename(columns={"type": "store_type"})
oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
weather = pd.concat(
    [pd.read_csv(f, parse_dates=["date"]) for f in (RAW / "weather").glob("*.csv")],
    ignore_index=True,
)
# rain_sum is almost identical to precipitation_sum (no snow here), so drop it
weather = weather.drop(columns=["rain_sum"])

# ---------- 2. Add store info (city, state, type, cluster) ----------
df = train.merge(stores, on="store_nbr", how="left")

# ---------- 3. Add weather by date + city (left join keeps all sales rows) ----------
df = df.merge(weather, on=["date", "city"], how="left")
df["is_rainy"] = (df["precipitation_sum"] >= 1).astype(int)  # 1 mm or more

# ---------- 4. Oil price: full daily calendar, fill gaps ----------
days = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})
oil = days.merge(oil, on="date", how="left")
oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()  # last known price, then fill the start
oil = oil.rename(columns={"dcoilwtico": "oil_price"})
df = df.merge(oil, on="date", how="left")

# ---------- 5. Holidays ----------
# keep real days off only; drop holidays that were transferred to another date
h = holidays[
    (~holidays["transferred"])
    & (holidays["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]))
]
national_dates = set(h.loc[h["locale"] == "National", "date"])
regional = (
    h[h["locale"] == "Regional"][["date", "locale_name"]]
    .drop_duplicates()
    .rename(columns={"locale_name": "state"})
    .assign(is_regional_holiday=1)
)
local = (
    h[h["locale"] == "Local"][["date", "locale_name"]]
    .drop_duplicates()
    .rename(columns={"locale_name": "city"})
    .assign(is_local_holiday=1)
)
df["is_national_holiday"] = df["date"].isin(national_dates).astype(int)
df = df.merge(regional, on=["date", "state"], how="left")
df = df.merge(local, on=["date", "city"], how="left")
flags = ["is_regional_holiday", "is_local_holiday"]
df[flags] = df[flags].fillna(0).astype(int)
df["is_holiday"] = df[["is_national_holiday"] + flags].max(axis=1)

# ---------- 6. Date features ----------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.day_of_week        # 0 = Monday
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
# public-sector wages are paid on the 15th and the last day of the month
df["is_payday"] = ((df["day"] == 15) | df["date"].dt.is_month_end).astype(int)
# simplified season: rainy Dec-May, dry Jun-Nov
df["season"] = np.where(df["month"].isin([12, 1, 2, 3, 4, 5]), "rainy", "dry")

# ---------- 7. Validate ----------
assert len(df) == len(train), "row count changed during merge!"
assert not df.duplicated(["date", "store_nbr", "family"]).any(), "duplicate rows"
assert df["temperature_2m_mean"].notna().all(), "some rows have no weather"
assert df["oil_price"].notna().all(), "some rows have no oil price"
print("Shape:", df.shape)
print(df.isna().sum())

# ---------- 7b. Drop rows before each store's first sale (store not open yet) ----------
first_sale = df[df["sales"] > 0].groupby("store_nbr")["date"].min().rename("first_sale_date")
df = df.merge(first_sale, on="store_nbr", how="left")
n_before = len(df)
df = df[df["date"] >= df["first_sale_date"]].drop(columns="first_sale_date")
print(f"Dropped {n_before - len(df)} rows before store opening; {len(df)} rows left")
print("Final shape:", df.shape)

# ---------- 8. Save ----------
df.to_csv(OUT / "merged.csv", index=False)
print("Saved", OUT / "merged.csv")