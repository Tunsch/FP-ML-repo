from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    source_dir = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-input-data") #anpassen
    test_ratio: float = 0.25
    split_seed: int = 7
    #Spaltendefinitionen ei Änderung des Input-Formats anpassen
    label_column: str = "label"
    meta_columns: List[str] = field(
        default_factory=lambda: ["vector_id", "label", "session", "category", "n_imputed",
    "sensor_index", "heater_profile_id", "cycle_id",])

    #Extra-CSVs als explizite Testdaten
    test_data_dir = Optional[Path] = None #anpassen
    #Über Session wählen
    explicit_test_sessions: Optional[list[str]] = None

    # Speicherort des präparierten Datensatzes
    ml_data_dir = Path("ml-data/")

    random_seed: int = 42

    #Hilfsfunktion zum zurückgeben aller Spalten die nicht Features sind
    @property
    def non_feature_columns(self) -> List[str]:
        return self.meta_columns + [self.label_column]
