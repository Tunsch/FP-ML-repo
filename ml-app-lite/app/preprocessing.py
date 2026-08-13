import numpy as np
import pandas as pd

from pandas import Series
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Any

from config import ExperimentConfig

LOG_CLIP_EPS = 1e-3

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log10(df.clip(lower=LOG_CLIP_EPS))

def drop_too_incomplete(df: pd.DataFrame, feature_cols: list[str], max_missing: int) -> pd.DataFrame:
    #Zeilen mit zu vielen fehlenden Feature-Werten
    n_missing = df[feature_cols].isna().sum(axis=1)
    keep = n_missing <= max_missing
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"Preprocessing: Dropped {n_dropped} missing from {feature_cols} (> {max_missing}) fehlende Feature-Werte.")
    return df.loc[keep].reset_index(drop=True)

def impute_within_session(X: pd.DataFrame, session: pd.Series) -> pd.DataFrame:
    #Imputation mit Median derselben Spalte innerhalb einer Session
    X = X.copy()
    session_median = X.groupby(session).transform("median")
    X = X.fillna(session_median)

    still_missing = X.isna()
    if still_missing.to_numpy().any():
        n_cells = int(still_missing.to_numpy().any().sum())
        print(f"Preprocessing warning: {n_cells} bleiben NaN.")

    return X

def preprocess_pipeline( train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         feature_cols: list[str],
                         config: ExperimentConfig) -> tuple[Any, Series[Any], Any, Series[Any]]:
    #1. Zeilen mit zu vielen fehlenden Feature-Values verwerfen
    #Bei Testdaten keine fehlenden oder imputierten Werte erlaubt
    train_df = drop_too_incomplete(train_df, feature_cols, config.impute_max_missing)
    test_df = drop_too_incomplete(test_df, feature_cols, config.impute_max_missing)

    #2. Modellinputs generieren
    y_train = train_df[config.label_column].copy()
    y_test = test_df[config.label_column].copy()

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    #3. Log-Transformation
    if config.log_transform:
        X_train_raw = log_transform(X_train_raw)
        X_test_raw = log_transform(X_test_raw)

    #4. Imputation
    X_train_imputed = impute_within_session(X_train_raw, train_df[config.session_column])
    X_test_imputed = impute_within_session(X_test_raw, test_df[config.session_column])

    #5. Skalierung
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, y_train, y_test





