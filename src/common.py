"""Shared data loading, split and preprocessing for the DS203 HDI project.
Imported by 03a_preprocessing.ipynb and 03b_modelling.ipynb so both use the same split and pipeline."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

SEED = 42
CLASS_NAMES = ["Low", "Medium/High", "Very High"]
NUM = ["gdp_growth", "inflation", "unemployment", "infant_mortality", "electricity_access",
       "internet_users", "agriculture_gdp", "industry_gdp", "trade_gdp"]
CAT = ["region", "DominantSector"]
SKEWED = ["inflation", "trade_gdp"]          # heavily right-skewed (see EDA)


def load_data(path="../data/hdi_wdi_2023.csv"):
    """Return the full table, features X (NUM + CAT only) and target y."""
    df = pd.read_csv(path)
    return df, df[NUM + CAT], df["hdi_class"].to_numpy()


def split_data(X, y, ids):
    """Stratified 60/20/20 split. Returns train, validation and test parts (with country names)."""
    X_trval, X_test, y_trval, y_test, id_trval, id_test = train_test_split(
        X, y, ids, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_trval, y_trval, id_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            X_trval, y_trval, id_train, id_val, id_test)


def signed_log(a):
    """sign(x) * log(1 + |x|): compresses large values, works for negatives (deflation)."""
    return np.sign(a) * np.log1p(np.abs(a))


def make_preprocessor(num=NUM, cat=CAT, impute="mean", indicator=False, log_skewed=True):
    """ColumnTransformer fitted later on TRAINING data only (inside a Pipeline)."""
    skew = [c for c in num if c in SKEWED and log_skewed]
    plain = [c for c in num if c not in skew]
    parts = []
    if skew:
        parts.append(("skewed", Pipeline([
            ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy=impute, add_indicator=indicator)),
            ("scale", StandardScaler())]), skew))
    if plain:
        parts.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy=impute, add_indicator=indicator)),
            ("scale", StandardScaler())]), plain))
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts)