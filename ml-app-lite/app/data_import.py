from __future__ import annotations

from pathlib import Path
import pandas as pd

from core import train_test_split_by_session
from config import ExperimentConfig


def load_feature_table(data_dir: Path, config: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    #Einlesen aller CSVs aus source_dir.
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {data_dir}.")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    #Spalten mit Datenpunkten
    feature_cols = [c for c in df.columns if c not in config.meta_columns]
    #Info
    print(f"Daten erfolgreich geladen! Gesamte Zeilen: {df.shape[0]}, Spalten: {df.shape[1]}")
    df.head()
    return df, feature_cols


def split_dataframe(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Split mit expliziten (neuen) Testdaten
    if config.test_ratio is not None:
        test_path = Path(config.test_data_dir)
        if not test_path.is_dir():
            test_path = config.source_dir / test_path

        test_df = pd.read_csv(test_path)
        train_df = df.copy()

        train_df["category"] = "training"
        test_df["category"] = "testing"
    #Split mit bestehenden Sessions
    else:
        df_split = train_test_split_by_session(df, test_ratio=config.test_ratio, seed=config.split_seed)
        train_df = df_split[df_split.category == "training"].copy()
        test_df = df_split[df_split.category == "testing"].copy()
    #Leakage-Check
    overlap = set(train_df["session"]) & set(test_df["session"])
    if overlap:
        raise RuntimeError(f"Sessions overlap between training and testing sessions: {overlap}")
    return train_df, test_df