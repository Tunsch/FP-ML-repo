from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import pandas as pd

from config import ExperimentConfig

def load_csv_dir(data_dir: Path, heater_profile: Optional[str] = None,
                 heater_profile_column: str = "heater_profile_id") -> pd.DataFrame:
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
    print(f"{data_dir}: {len(csv_files)} Datein, {df.shape[0]} Zeilen geladen.")

    if heater_profile is not None:
        n_before = len(df)
        df = df[df[heater_profile_column] == heater_profile].reset_index(drop=True)
        if df.empty:
            available = sorted(pd.concat(
                [pd.read_csv(f, usecols=[heater_profile_column]) for f in csv_files]
            )[heater_profile_column].unique())
            raise ValueError(f"Heizprofil '{heater_profile}' nicht in {data_dir} gefunden. "
                             f"Verfügbar: {available}")
        print(f"{data_dir}: auf Heizprofil '{heater_profile}' gefiltert "
              f"({len(df)}/{n_before} Zeilen).")
    return df

def discover_heater_profiles(data_dir: Path, heater_profile_column: str = "heater_profile_id") -> list[str]:
    #Findet alle Heizprofile
    csv_files = sorted(Path(data_dir).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    profiles: set[str] = set()
    for csv_file in csv_files:
        profiles.update(pd.read_csv(csv_file, usecols=[heater_profile_column])[heater_profile_column].unique())
    return sorted(profiles)

def split_by_session(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    #Aufteilung wird getrennt nach Label berechnet. Nie selbe Session in Training und Testing
    rng = random.Random(config.split_seed)
    sessions_by_label: dict = {}
    cols = [config.session_column, config.label_column]
    for session, label in df[cols].drop_duplicates().itertuples(index=False):
        sessions_by_label.setdefault(label, []).append(session)

    categories: dict = {}
    for label, sessions in sessions_by_label.items():
        sessions = sorted(sessions)
        rng.shuffle(sessions)
        n_test = max(1, round(len(sessions) * config.test_ratio)) if len(sessions) > 1 else 0
        test_sessions = set(sessions[:n_test])
        categories.update({s: "testing" if s in test_sessions else "training" for s in sessions})

    if config.explicit_test_sessions:
        categories.update(config.explicit_test_sessions)

    df = df.copy()
    df["category"] = df[config.session_column].map(categories).fillna("training")
    return df

def split_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    #Split entsprechend config.split_mode, bei "session" je Label stratifiziert
    if config.split_mode == "session":
        df = load_csv_dir(config.source_dir)
        df = split_by_session(df, config)

    elif config.split_mode == "explicit":
        if config.test_data_dir is None:
            raise ValueError("split_mode='explicit' erfordert test_data_dir in der Config.")
        train_df = load_csv_dir(config.source_dir)
        test_df = load_csv_dir(config.test_data_dir)
        train_df["category"] = "training"
        test_df["category"] = "testing"
        df = pd.concat([train_df, test_df], ignore_index=True)

    else:
        raise ValueError(f"split_mode={config.split_mode} is not supported.")

    #Leakage-Check zwischen Training und Testing
    train_sessions = set(df.loc[df.category == "training", config.session_column])
    test_sessions = set(df.loc[df.category == "testing", config.session_column])
    overlap = train_sessions & test_sessions
    if overlap:
        raise RuntimeError(f"Sessions overlap between training and testing sessions: {overlap}")

    feature_cols = [c for c in df.columns if c not in config.non_feature_columns]
    train_df = df[df.category == "training"].reset_index(drop=True)
    test_df = df[df.category == "testing"].reset_index(drop=True)
    print(f"Gesamt: {df.shape[0]} Zeilen, {len(feature_cols)} Features "
          f"| training={len(train_df)} testing={len(test_df)}")

    return train_df, test_df, feature_cols
