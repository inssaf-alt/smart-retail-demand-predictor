"""Download daily historical weather for each Favorita city from Open-Meteo."""
from pathlib import Path
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
cities = pd.read_csv(ROOT / "Data" / "city_coordinates.csv")
out_dir = ROOT / "Data" / "raw" / "weather"
out_dir.mkdir(parents=True, exist_ok=True)

URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum"

for _, row in cities.iterrows():
    file = out_dir / f"{row['city'].replace(' ', '_')}.csv"
    if file.exists():                      # skip cities already downloaded
        continue

    params = {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "start_date": "2013-01-01",
        "end_date": "2017-08-15",          # last date in train.csv
        "daily": DAILY,
        "timezone": "America/Guayaquil",   # so dates line up with the sales dates
    }

    for attempt in range(5):               # retry if the server says "too many requests"
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            break
        print(f"{row['city']}: status {r.status_code}, waiting 30s...")
        time.sleep(30)
    else:
        print(f"FAILED: {row['city']}")
        continue

    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    df.insert(1, "city", row["city"])
    df.to_csv(file, index=False)
    print("saved", file.name, df.shape)
    time.sleep(2)                          # be polite to the free API