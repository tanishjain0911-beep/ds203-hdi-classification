# Data dictionary: `hdi_wdi_2023.csv`

One row per country (193 countries), reference year 2023.
Built in `notebooks/01_data.ipynb`. The year used for every value is in `feature_year_used.csv`.

## Sources
- **Target (HDI):** UNDP Human Development Report 2025 release, *Composite indices complete time series*, column `hdi_2023`.
  https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv
- **Predictors:** World Bank World Development Indicators, downloaded as CSV from https://data.worldbank.org/indicator/ (search the code → Download → CSV), accessed 25 Nov 2026. Saved as `raw/wb_<code>.csv`.
- **Region:** the `Metadata_Country` CSV included in the same World Bank download, saved as `raw/wb_country_metadata.csv`.

## Construction rules
- **Year rule:** use the 2023 value; if missing, the most recent value from 2022, then 2021. No value after 2023 is used. If none exists, the value stays missing (NaN) for the imputation step.
- **Aggregates removed:** HDR rows with ISO3 codes starting "ZZ" and World Bank entries with an empty region (aggregates such as World or income groups).
- **Matching:** merged on ISO3 codes. All 195 HDR countries matched; no manual corrections were needed. 22 World Bank economies without an HDI (territories) are excluded.
- **Dropped:** Monaco and DPR Korea (no 2023 HDI, so no label).
- **Missing values are kept** (not imputed here). Imputation happens later, inside the model pipeline, on training data only.

## Columns
| Column | Type | Source / code | Description |
|---|---|---|---|
| `iso3` | identifier | — | ISO3 country code (**not a predictor**) |
| `country` | identifier | UNDP | Country name (**not a predictor**) |
| `region` | categorical | World Bank metadata | Geographic region (7 levels) |
| `DominantSector` | categorical, engineered | from `agriculture_gdp`, `industry_gdp` | Largest of agriculture, industry, services (= 100 − agriculture − industry); missing if a share is missing or the services share would be negative |
| `gdp_growth` | numeric | NY.GDP.MKTP.KD.ZG | GDP growth (annual %) |
| `inflation` | numeric | FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) |
| `unemployment` | numeric | SL.UEM.TOTL.ZS | Unemployment, total (% of labour force, modelled ILO estimate) |
| `infant_mortality` | numeric | SP.DYN.IMRT.IN | Infant mortality (per 1,000 live births) |
| `electricity_access` | numeric | EG.ELC.ACCS.ZS | Access to electricity (% of population) |
| `internet_users` | numeric | IT.NET.USER.ZS | Individuals using the Internet (% of population) |
| `agriculture_gdp` | numeric | NV.AGR.TOTL.ZS | Agriculture, forestry and fishing, value added (% of GDP) |
| `industry_gdp` | numeric | NV.IND.TOTL.ZS | Industry (incl. construction), value added (% of GDP) |
| `trade_gdp` | numeric | NE.TRD.GNFS.ZS | Trade (% of GDP) |
| `hdi` | numeric | UNDP `hdi_2023` | HDI score: **used only to build the target, never a predictor** |
| `hdi_class` | target | derived | 0 = Low (HDI < 0.550), 1 = Medium/High (0.550 ≤ HDI < 0.800), 2 = Very High (HDI ≥ 0.800). Project-defined grouping, not an official UNDP classification |

## Missingness (% of 193 countries) and look-back use
| Feature | % missing | values from 2021–2022 |
|---|---|---|
| gdp_growth | 2.1 | 1 |
| inflation | 7.8 | 3 |
| unemployment | 7.3 | 3 |
| infant_mortality | 1.0 | 0 |
| electricity_access | 0.0 | 0 |
| internet_users | 5.7 | 3 |
| agriculture_gdp | 3.6 | 3 |
| industry_gdp | 3.6 | 3 |
| trade_gdp | 11.9 | 3 |
| region | 0.0 | — |
| DominantSector | 3.6 | — |

## Excluded to avoid target leakage
HDI score/category, life expectancy, expected and mean years of schooling, and GNI per capita are not used as predictors. `iso3` and `country` are identifiers only.