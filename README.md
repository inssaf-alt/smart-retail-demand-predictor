# smart-retail-demand-predictor
Machine learning project for predicting retail sales using weather and external factors.

1. Download the Kaggle files into `Data/raw/` (`train.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`, `transactions.csv`)
2. `pip install -r requirements.txt`
3. `python src/download_weather.py`
4. `python src/build_dataset.py`

## Data sources

- **Retail sales:** Kaggle, "Store Sales - Time Series Forecasting" (https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Download it yourself into `Data/raw/` (not included in the repo).
- **Weather:** [Weather data by Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The weather was cleaned, merged with the sales data, and new features were derived from it (changes were made). The underlying data are the ERA5 and ERA5-Land reanalyses (ECMWF).

References:
- Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
- Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
- Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC

Preprocessing decisions, numbers, and limitations: see `notes.md`.
