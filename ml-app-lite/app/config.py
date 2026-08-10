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
    ml_data_dir: Path = Path("ml_data/ml_input_data/")

    random_seed: int = 42

    #Hilfsfunktion zum Zurückgeben aller Spalten die nicht Features sind
    @property
    def non_feature_columns(self) -> List[str]:
        return self.meta_columns + [self.label_column]
