from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd 

from core import train_test_split_by_session


@dataclass
class ExperimentConfig:
    data_dir = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-input-data") #anpassen
    test_ratio: float = 0.25
    split_seed: int = 7
    
    #Testdaten explizit festlegen
    custom_test_data: bool = False
    #Extra-CSVs
    custom_test_dir = Optional[Path] = None #anpassen
    #Über Session wählen
    explicit_test_sessions: Optional[list[str]] = None



#Bei Änderung des Input-Formats anpassen
META_COLUMNS = [
    "vector_id", "label", "session", "category", "n_imputed",
    "sensor_index", "heater_profile_id", "cycle_id",
]

def load_feature_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    #Einlesen aller CSVs aus data_dir. 
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {data_dir}.")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    #Spalten mit Datenpunkten
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return df, feature_cols


def split_data(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Session-basierter Split aus core.py oder custom Split, wenn explizite Testdaten verwendet