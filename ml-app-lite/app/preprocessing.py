from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pathlib import Path
from pandas import Series
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional

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
                         config: ExperimentConfig) -> tuple[Any, Any, Series[Any], Series[Any], Series[Any], Series[Any], StandardScaler]:
    #1. Zeilen mit zu vielen fehlenden Feature-Values verwerfen
    #Bei Testdaten keine fehlenden oder imputierten Werte erlaubt
    train_df = drop_too_incomplete(train_df, feature_cols, config.impute_max_missing)
    test_df = drop_too_incomplete(test_df, feature_cols, config.impute_max_missing)

    #2. Modellinputs generieren
    y_train = train_df[config.label_column].copy()
    y_test = test_df[config.label_column].copy()
    groups_train = train_df[config.session_column].copy()
    groups_test = test_df[config.session_column].copy()

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

    #Der gefittete scaler wird zusätzlich zurückgegeben (vorher wurde er nach
    #Gebrauch verworfen) -- er ist die Voraussetzung dafür, dieselbe Skalierung
    #außerhalb dieser Pipeline (Live-Inferenz, Edge-Deployment) zu reproduzieren.
    #Siehe export_scaler_artifact() unten.
    return X_train_scaled, X_test_scaled, y_train, y_test, groups_train, groups_test, scaler


def export_scaler_artifact(scaler: StandardScaler, feature_cols: list[str],
                            config: ExperimentConfig, train_df: pd.DataFrame,
                            out_path: Optional[Path] = None) -> dict:
    """Schreibt ein portables JSON-Artefakt mit allem, was nötig ist, um das
    Preprocessing (Log-Transformation + Skalierung) AUSSERHALB dieser
    Python/joblib-Pipeline zu reproduzieren -- insbesondere für:
      - pc_live_classify.py (liest BME688 live, wendet dieselbe Transformation
        + Skalierung an, sendet fertigen Vektor an den ESP32)
      - eine mögliche spätere C-Portierung eines eigenen (nicht Edge-Impulse-)
        Modells auf dem Mikrocontroller (Skalierung dort direkt aus
        scaler_mean/scaler_scale nachbauen).

    joblib bleibt für die interne ML-Pipeline (RF/SVM/KNN/NN-Objekte) im
    Einsatz -- dieses JSON ist NUR der Preprocessing-Vertrag, sprachunabhängig
    und ohne Pickle-/Versionsbindung.

    log_transform gibt an, ob (genau einmal) log10 angewendet wird -- siehe
    Kommentar zu config.log_transform. Die Edge-/Live-Pipeline
    (pc_preprocess_bridge.py) geht von rohen, unlogarithmierten Gaswiderständen
    vom Mikrocontroller aus und wendet log10 hier repliziert genau einmal an,
    wenn dieses Flag gesetzt ist.
    """
    if config.data_variant != "sensor_level":
        print("WARNUNG: export_scaler_artifact()/pc_preprocess_bridge.py sind für "
              "data_variant='sensor_level' (ein Sensor) ausgelegt. Für "
              "'heater_profile_level' (Sensorpaare) müssten Zyklus- und "
              "Feature-Logik in pc_preprocess_bridge.py angepasst werden.")

    if out_path is None:
        out_path = Path(config.ml_data_dir) / "preprocessing_artifact.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    #sklearn's LabelEncoder (siehe validation.py) sortiert Klassen alphabetisch --
    #dieselbe Sortierung wird hier repliziert, damit label_classes[i] konsistent
    #zum späteren NN-Klassenindex i ist, falls das eigene Modell portiert wird.
    label_classes = sorted(train_df[config.label_column].unique().tolist())

    artifact = {
        "heater_profile": config.heater_profile,
        "data_variant": config.data_variant,
        "feature_cols": list(feature_cols),
        "n_expected_steps": len(feature_cols),
        "log_transform": bool(config.log_transform),
        "log_clip_eps": LOG_CLIP_EPS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "label_classes": label_classes,
        "session_column": config.session_column,
        "label_column": config.label_column,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"Preprocessing-Artefakt geschrieben: {out_path} "
          f"(log10-Stufe, {len(feature_cols)} Features, "
          f"{len(label_classes)} Klassen).")
    return artifact