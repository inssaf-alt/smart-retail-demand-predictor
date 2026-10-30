# Data collection & preprocessing notes

Project: effect of weather and other factors on retail sales (Favorita stores, Ecuador)
Scripts: `src/download_weather.py` -> `src/build_dataset.py` -> `Data/processed/merged.csv`

---

## 1. Decision log (what we did and why)

| # | Decision | Why |
|---|----------|-----|
| 1 | Retail data: Kaggle "Store Sales - Time Series Forecasting" (`train.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`) | Real data, has city names, holidays and oil, many references |
| 2 | Weather data: Open-Meteo Historical Weather API, one coordinate pair per city, timezone `America/Guayaquil` | Free, CC BY 4.0, daily data covering 2013-2017, dates line up with sales dates |
| 3 | 22 cities with approximate city-centre coordinates (`Data/city_coordinates.csv`) | `stores.csv` only has city names. 
| 4 | Dropped `rain_sum` | Almost identical to `precipitation_sum` (no snow in these cities) |
| 5 | Added `is_rainy` = precipitation >= 1 mm | Simple rainy/dry indicator |
| 6 | Merged weather with a left join on `date` + `city` | Keeps every sales row |
| 7 | Oil: built a full daily calendar, forward-filled, then back-filled the start | Oil prices are not recorded on weekends/holidays (43 missing values plus absent days) |
| 8 | Holidays: kept types Holiday, Transfer, Additional, Bridge; dropped rows with `transferred = True`; matched Regional holidays by state and Local holidays by city | A transferred holiday was not celebrated on that date; regional/local holidays only apply to some stores |
| 9 | `is_payday` = 15th or last day of month | Kaggle data description: public-sector wages are paid on these days |
| 10 | `season` = rainy (Dec-May) / dry (Jun-Nov) | Simplified; does not fit every region |
| 11 | `transactions.csv` not used | Transactions are counted the same day as sales, so using it as a feature in a forecast would leak the answer |
| 12 | Removed each store's rows before its first sale | Those zeros mean "store not open yet", not "no demand" (222,057 rows removed) |
| 13 | Kept the zeros after a store's first sale | They are real values (product not sold that day) |
| 14 | Did not fill the 4 missing dates (Dec 25 each year, 2013-2016) | Stores were closed |

## 2. Key numbers

- `train.csv`: 3,000,888 rows, 6 columns, 2013-01-01 to 2017-08-15 (1,688 days)
- `stores.csv`: 54 stores, 22 cities
- Product families: 33 (confirmed with `nunique()`)
- Missing values / duplicates: none in train, stores, holidays, transactions. `oil.csv`: 1,218 rows, 43 missing prices
- Holidays: 350 rows, 38 dates with more than one row, 12 transferred
- Negative sales: 0. Store numbers in train all exist in stores
- Zero-sales rows: 31.3% of the original `train.csv`, 25.8% of the final dataset (after removing rows before store openings)
- Weather: 22 cities x 1,688 days = 37,136 rows, no missing values. Daily max temperature 11.7-35.1 C, min temperature 1.9-26.3 C, precipitation 0-171.5 mm/day (mean 6.2 mm)
- Rows dropped before store opening: 222,057 (7.4%). Late-opening stores: 36, 53, 20, 29, 21, 42, 22, 52 (store 52 opened 2017-04-20)
- **Final dataset: 2,778,831 rows x 28 columns** (`Data/processed/merged.csv`)

## 3. Sources to cite (verify each before the final report)

- Kaggle competition "Store Sales - Time Series Forecasting": https://www.kaggle.com/competitions/store-sales-time-series-forecasting (check rules and licence)
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api (CC BY 4.0).
- Open-Meteo data downloaded on 2026-10-30
- Underlying data sets named by Open-Meteo (default model selection):
  - Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023). ERA5 hourly data on single levels from 1940 to present [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
  - Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC
- Resolution (Open-Meteo data sources table): ERA5 0.25 deg (~25 km), ERA5-Land 0.1 deg (~11 km), both hourly, available from 1940/1950 to present. ECMWF IFS (9 km) only starts in 2017, so it cannot cover most of our period. We used Open-Meteo's default model selection
- Open-Meteo citation: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Requirements: give appropriate credit, link to the licence, indicate that changes were made (we cleaned, merged and derived new features), and put a link "Weather data by Open-Meteo.com" (https://open-meteo.com/) next to wherever the weather data is shown (report tables/figures, README)
- CERRA is on the Open-Meteo citation list but only covers Europe, so we do not cite it

## 4. Limitations

- Weather is matched at city level, not store level (about 2.5 stores per city on average, many stores share a city)
- Tropical country: temperature varies little over the year, most variation is between cities (altitude)
- `season` is a simplified rainy/dry split
- Coordinates are approximate city centres
- Weather values are reanalysis grid-cell estimates (about 11-25 km), not station measurements. Nearby cities (for example Salinas and Libertad) can get almost identical weather, and in the Andes the elevation changes a lot within one grid cell. Modelled rainfall can differ from actual local rain
- Check: Salinas and Libertad (about 10 km apart) have identical mean temperature on only 6.8% of days, but their daily precipitation correlation is 1.0, so they effectively share one rainfall series. The number of independent weather locations is therefore smaller than 22
- Store 52 has only about 4 months of history, so it is hard to predict in a chronological test set
- Rain vs sales across cities is confounded by city size. Compare sales per store or deviation from each city's usual level

## 5. How to reproduce

1. Download the Kaggle files into `Data/raw/` (`train.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`, `transactions.csv`)
2. `pip install -r requirements.txt`
3. `python src/download_weather.py`
4. `python src/build_dataset.py`

---

## 6. Draft for the report: Datasets and preprocessing (edit before using)

### 3.1 Datasets

**Retail sales.** We use the *Store Sales - Time Series Forecasting* dataset from Kaggle [1]. It contains daily unit sales of 33 product families at 54 stores of an Ecuadorian grocery retailer from 2013-01-01 to 2017-08-15 (3,000,888 rows in `train.csv`). Supporting files describe the stores (city, state, type, cluster), national, regional and local holidays, daily oil prices, and daily transactions per store. The target variable is `sales`.

**Weather.** Daily weather for the 22 cities where the stores are located was downloaded from the Open-Meteo Historical Weather API [2], using one coordinate pair per city and the time zone America/Guayaquil. The variables are daily maximum, minimum and mean temperature and total precipitation. The data is licensed under CC BY 4.0.

### 3.2 Preprocessing

**Quality checks.** `train`, `stores`, `holidays` and `transactions` have no missing values or duplicate rows. `oil` has 43 missing prices and no values on weekends and holidays. There are no negative sales, all store numbers in `train` exist in `stores`, and the only calendar days absent from `train` are 25 December of 2013-2016. The weather data is complete (22 cities x 1,688 days). About 31% of the original sales rows are zero. Removing the rows from before each store opened lowers this to 25.8%, and the remaining zeros are real observations of no sales.

**Merging.** Sales were joined with store information on `store_nbr`, then with weather on `date` and `city` (left join, so no sales row is lost), and with oil prices on `date`. Oil prices were reindexed to a full daily calendar and gaps were filled with the last known price (and the next price for the first days). A merge check confirmed that the number of rows did not change and that no (`date`, `store_nbr`, `family`) combination is duplicated.

**Holiday handling.** Only holiday types that represent actual days off (Holiday, Transfer, Additional, Bridge) were kept, and holidays marked as transferred were removed. National holidays apply to all stores, regional holidays to stores in the matching state, and local holidays to stores in the matching city.

**Feature engineering.** We created calendar features (year, month, day, day of week, week of year, weekend flag), a payday flag (15th and last day of the month, when public-sector wages are paid), a simplified season (rainy December-May, dry June-November), a rainy-day flag (precipitation of at least 1 mm), and holiday flags at national, regional, local and combined level.

**Store openings.** Several stores opened after the start of the data. For each store, rows before its first day with positive sales were removed, because these zeros represent a closed store rather than zero demand. This removed 222,057 rows (7.4%). Zeros after a store's first sale were kept, since they are real observations.

**Final dataset.** The final table has 2,778,831 rows and 28 columns with no missing values.

### References (placeholders, fill in the real format)

[1] Kaggle, "Store Sales - Time Series Forecasting", https://www.kaggle.com/competitions/store-sales-time-series-forecasting
[2] Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
[3] Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
[4] Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC