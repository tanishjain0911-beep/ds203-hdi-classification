# Economic Development Classification Using HDI

**DS203 Project, Classification track (2026)**

Can widely available economic, infrastructure, health and geographic indicators classify countries into
HDI-based development categories? We build a country-level dataset for the reference year 2023, explore it, and
compare a majority-class baseline, multinomial logistic regression, a decision tree and a random forest inside
leakage-free scikit-learn pipelines.

| | |
|---|---|
| **Report** | [`report/report.pdf`](report/report.pdf) |
| **Video (10 min)** | _link to be added_ |
| **Dataset** | [`data/hdi_wdi_2023.csv`](data/hdi_wdi_2023.csv) (193 countries), described in [`data/data_dictionary.md`](data/data_dictionary.md) |

## Team

| Member | Roll no. | Responsibility |
|---|---|---|
| Tanish Jain | 25B4244 | Data acquisition, dataset construction, preprocessing experiments (`01_data`, `03a_preprocessing`) |
| Ruturva Vaghasia | 25B4205 | Model training, tuning, evaluation and interpretation (`03b_modelling`) |
| Arnav Jain | 25B2704 | Exploratory data analysis, documentation and presentation (`02_EDA`) |

## Problem

The target is a three-class grouping of the published 2023 HDI score, defined for this project
(UNDP officially reports four categories; these classes are **not** an official classification):

| Class | Label | HDI range | Countries |
|---|---|---|---|
| 0 | Low | HDI < 0.550 | 26 (13.5%) |
| 1 | Medium/High | 0.550 ≤ HDI < 0.800 | 93 (48.2%) |
| 2 | Very High | HDI ≥ 0.800 | 74 (38.3%) |

**Predictors (11):** nine World Bank indicators (GDP growth, inflation, unemployment, infant mortality, access
to electricity, internet users, agriculture and industry value added as % of GDP, trade as % of GDP), the
World Bank region, and the engineered `DominantSector` (largest of agriculture, industry and services, where
services = 100 − agriculture − industry).

**Excluded to avoid target leakage:** the HDI score and category, life expectancy, expected and mean years of
schooling, and GNI per capita. Country names and ISO3 codes are used only as identifiers.

## Repository structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── hdi_wdi_2023.csv        final fixed dataset (one row per country)
│   ├── feature_year_used.csv   year (2021–2023) actually used for every country × indicator
│   ├── data_dictionary.md      sources, indicator codes, year rule, columns, missingness
│   └── raw/                    source files exactly as downloaded
│       ├── HDR25_composite_indices.csv     UNDP HDR 2025 composite indices time series
│       ├── wb_<indicator code>.csv         one World Bank WDI file per indicator
│       └── wb_country_metadata.csv         World Bank country metadata (region)
├── notebooks/
│   ├── 01_data.ipynb             build the dataset from the raw files
│   ├── 02_EDA.ipynb              exploratory data analysis (the 6 EDA tasks of the brief)
│   ├── 03a_preprocessing.ipynb   split, pipeline, imputation, PCA and feature-set experiments
│   └── 03b_modelling.ipynb       baseline, LR, decision tree, random forest; test evaluation; interpretation
├── figures/                      all plots, written by the notebooks
└── report/                       LaTeX report (report.pdf)
```

## How to run

**Requirements:** Python 3.10 or later.

```bash
git clone https://github.com/tanishjain0911-beep/ds203-hdi-classification.git
cd ds203-hdi-classification
pip install -r requirements.txt
jupyter lab            # or: python -m jupyterlab
```

Open the notebooks from the `notebooks/` folder (all paths are relative to it) and run each one with
**Kernel → Restart Kernel and Run All Cells**, in this order:

| Order | Notebook | Reads | Writes | Run time |
|---|---|---|---|---|
| 1 | `01_data.ipynb` | `data/raw/*` | `data/hdi_wdi_2023.csv`, `data/feature_year_used.csv` | seconds |
| 2 | `02_EDA.ipynb` | `data/hdi_wdi_2023.csv` | 7 EDA figures | seconds |
| 3 | `03a_preprocessing.ipynb` | `data/hdi_wdi_2023.csv` | 3 PCA figures | about 1 min |
| 4 | `03b_modelling.ipynb` | `data/hdi_wdi_2023.csv` | confusion matrices, coefficient and importance figures | 2–3 min |

The final dataset is already included, so notebooks 2–4 can be run without running `01_data.ipynb` first.
No internet connection is needed. All randomness uses `random_state = 42`, so the results are reproducible.

## Data sources

| Data | Source | Accessed |
|---|---|---|
| HDI 2023 (target) | UNDP Human Development Report 2025 release, *Composite indices complete time series* (CSV), column `hdi_2023`: https://hdr.undp.org/data-center/documentation-and-downloads | 23 Nov 2026 |
| 9 indicators | World Bank World Development Indicators, CSV download per indicator code: https://data.worldbank.org/indicator/ | 25 Nov 2026 |
| Region | World Bank country metadata (the `Metadata_Country` file included in the WDI download) | 25 Nov 2026 |

**Construction rules** (details in [`data/data_dictionary.md`](data/data_dictionary.md)):
- Reference year 2023. If a 2023 value is missing, the most recent value from 2022, then 2021, is used;
  no value after 2023 is ever used. 19 values come from 2021–2022, and the year of every value is recorded.
- Aggregates removed (HDR codes starting with `ZZ`; World Bank entries without a region).
- Merged on ISO3 codes: all 195 HDR countries matched, so no manual code corrections were needed.
  22 World Bank territories without an HDI are excluded.
- Monaco and DPR Korea are dropped (no 2023 HDI), giving **193 countries**.
- Missing predictor values (at most 11.9%, for trade) are kept for the imputation experiment.

## Method

1. **Split:** stratified 60 / 20 / 20 train / validation / test split (115 / 39 / 39 countries), with the class
   proportions preserved in every part.
2. **Preprocessing:** all learned steps sit inside a `Pipeline` with a `ColumnTransformer`, so they are fitted on
   training data only and refitted in every cross-validation fold:
   - sign-preserving log, `sign(x)·log(1+|x|)`, for the skewed `inflation` and `trade_gdp`;
   - numerical: mean imputation, then standardisation;
   - categorical: an explicit `"Missing"` category, then one-hot encoding with `handle_unknown="ignore"`.
3. **Preprocessing experiments** (`03a`): mean vs median imputation, with and without missingness indicators;
   PCA (explained variance, loadings, models with vs without PCA); numerical-only vs categorical-only vs
   combined feature sets.
4. **Models** (`03b`): majority-class baseline, multinomial logistic regression (L2, tuned `C`), decision tree
   (tuned `max_depth`, `min_samples_leaf`) and random forest (300 trees; tuned `max_depth`, `min_samples_leaf`).
   Tuning uses `GridSearchCV` with stratified 5-fold cross-validation on the training set and macro F1.
5. **Evaluation:** the validation set compares the tuned models. Each model is then refitted on train + validation
   and evaluated **once** on the held-out test set (accuracy, balanced accuracy, macro F1, per-class
   precision/recall/F1, confusion matrices).
6. **Interpretation:** logistic-regression coefficients, out-of-fold random-forest permutation importance, and the
   misclassified test countries.

## Key results

| Model | Best hyperparameters | CV macro F1 (mean ± SD) | Test accuracy | Test balanced acc. | Test macro F1 |
|---|---|---|---|---|---|
| Majority baseline | — | 0.216 ± 0.000 | 0.487 | 0.333 | 0.218 |
| Logistic regression | C = 100 | 0.849 ± 0.094 | 0.795 | **0.708** | **0.708** |
| Decision tree | max_depth = 4, min_samples_leaf = 3 | 0.801 ± 0.071 | 0.667 | 0.512 | 0.497 |
| Random forest | max_depth = 5, min_samples_leaf = 1 | **0.895 ± 0.074** | **0.821** | 0.676 | 0.682 |

- All models are far above the majority baseline. The random forest is best in cross-validation and logistic
  regression is best on the test set; the gap between them is within one CV standard deviation.
- **Most informative features:** electricity access, infant mortality, internet use and the agriculture share of
  GDP (random-forest permutation importance 0.171, 0.096, 0.073 and 0.048). Growth, inflation, unemployment
  and trade contribute almost nothing.
- **Preprocessing:** mean imputation without missingness indicators worked best. PCA shows one dominant
  "development" axis (PC1 explains 38.8% of the variance) but does not improve the final models. Region and
  `DominantSector` change CV macro F1 by at most ±0.01.
- **Errors** occur almost only between neighbouring classes, mostly for countries whose HDI lies close to the
  0.550 threshold.
- **Limitations:** small sample (193 countries, only 5 Low countries in the test set), class imbalance, missingness
  linked to development, and some indicator values from 2021–2022. The models show **predictive association,
  not causation**.
