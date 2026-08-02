"""
ml/data.py

Dünner Adapter zwischen den von core.py/cli.py bereits erzeugten
Feature-Vektor-CSVs und der ML-Pipeline. Enthält KEINE Logik zur
Umwandlung von Rohdaten in Vektoren -- das bleibt exklusiv Aufgabe von
core.py. Hier passiert nur:

  1. Feature-CSVs einlesen und feature_cols bestimmen
  2. Train/Test-Split auf Session-Ebene (importiert aus core.py)
  3. X/y-Arrays für ein gegebenes DataFrame extrahieren

Abhängigkeitsrichtung ist bewusst einseitig: ml/ hängt von core.py ab,
core.py weiss nichts von ml/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import train_test_split_by_session
from ml.config import ExperimentConfig

# Spalten, die core.py als Metadaten mitschreibt -- alles andere gilt als
# Feature. Wenn core.py um weitere Metadatenspalten ergänzt wird, hier
# mit aufnehmen.
META_COLUMNS = [
    "vector_id", "label", "session", "category", "n_imputed",
    "sensor_index", "heater_profile_id", "cycle_id",
]


def load_feature_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Liest alle CSVs in data_dir ein und führt sie zusammen. Gibt
    (DataFrame, feature_cols) zurück. Erwartet das von core.py erzeugte
    Format (Spalten wie 'label', 'session', Feature-Spalten)."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSVs in {data_dir} gefunden.")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return df, feature_cols


def split_data(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session-sauberer Split über die core.py-Funktion. Gibt (train_df,
    test_df) zurück und prüft zusätzlich, dass keine Session in beiden
    Mengen landet (Leakage-Schutz)."""
    df_split = train_test_split_by_session(
        df, test_ratio=config.test_ratio, seed=config.seed,
    )

    train_df = df_split[df_split.category == "training"]
    test_df = df_split[df_split.category == "testing"]

    overlap = set(train_df["session"]) & set(test_df["session"])
    if overlap:
        raise RuntimeError(f"Leakage! Sessions in beiden Kategorien: {overlap}")

    return train_df, test_df


def get_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extrahiert X (Features) und y (Label) aus einem DataFrame."""
    X = df[feature_cols].values
    y = df["label"].values
    return X, y
