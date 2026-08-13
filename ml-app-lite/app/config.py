from pathlib import Path
from typing import Optional, List, Literal, Dict
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    source_dir = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/raw_input_data/Bad") #anpassen

    #Art des Splits. Bei explicit test_data_dir angeben
    split_mode: Literal["session", "explicit"] = "session"

    #Für split_mode = "session"
    test_ratio: float = 0.25
    split_seed: int = 7

    #Optionales manuelles festlegen einzelner Sessions: {session_name: "training"/"testing"}
    explicit_test_sessions: Optional[Dict[str, str]] = None

    #Für split_mode = "explicit"
    test_data_dir: Optional[Path] = None #anpassen

    #Preprocessing
    log_transform: bool = True
    #max. Anzahl fehlender Feature-Werte pro Zeile, die imputiert werden. Zeilen mit fehlenden Werten werden verworfen.
    impute_max_missing: int = 2

    #Spaltendefinitionen bei Änderung des Input-Formats anpassen
    label_column: str = "label"
    session_column: str = "session"
    meta_columns: List[str] = field(
        default_factory=lambda: [
            "vector_id",
            "session",
            "category",
            "n_imputed",
            "heater_profile_id",
            "cycle_id_a",
            "cycle_id_b",
        ]
    )

    # Speicherort des präparierten Datensatzes
    ml_data_dir: Path = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/app/ml_data/ml_input_data")

    #Speicherort für Reports
    report_dir: Path = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/app/ml_data/reports")

    #Validierung: Anzahl der Folds für GroupKFold
    cv_folds: int = 5

    #Validierung: Metrik zur Bewertung/Sortierung der Modell-Konfigurationen
    selection_metric: str = "f1_macro"

    random_seed: int = 42

    #Hilfsfunktion zum Zurückgeben aller Spalten die nicht Features sind
    @property
    def non_feature_columns(self) -> List[str]:
        return self.meta_columns + [self.label_column]
