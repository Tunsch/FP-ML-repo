from typing import Any

import pandas as pd
from pandas import Series
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig

def preprocess_pipeline(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        config: ExperimentConfig) -> tuple[Any, Series[Any], Any, Series[Any]]:
    #1. xyz
    y_train = train_df[config.label_column].copy()
    y_test = test_df[config.label_column].copy()

    X_train_raw = train_df.drop(config.non_feature_columns)
    X_test_raw = test_df.drop(config.non_feature_columns)

    #2. Skalierung
    scaler = StandardScaler()
    scaler.set_output(transform="pandas") #Ausgabe als Dataframe
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, y_train, X_test_scaled, y_test



