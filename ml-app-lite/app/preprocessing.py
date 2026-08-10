from typing import Any

import pandas as pd
from pandas import Series
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig

def preprocess_pipeline(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        feature_cols: list[str],
                        config: ExperimentConfig) -> tuple[Any, Series[Any], Any, Series[Any]]:
    #1. xyz
    y_train = train_df[config.label_column].copy()
    y_test = test_df[config.label_column].copy()

    #TESTCODE
    print("DataFrame columns:")
    print(train_df.columns.tolist())

    print("\nNon-feature columns:")
    print(config.non_feature_columns)

    print("\nMissing columns:")
    print(set(config.non_feature_columns) - set(train_df.columns))

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    #2. Skalierung
    scaler = StandardScaler()
    scaler.set_output(transform="pandas") #Ausgabe als Dataframe
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, X_test_scaled, y_train, y_test



