from pathlib import Path
from typing import Optional, List, Literal, Dict
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    #Analyse-Pfad: "heater_profile_level" nutzt die bereits sensorpaarweise
    #kombinierten heater_XXX.csv (core.py::make_level2_df, 1 Zeile/Zyklus).
    #"sensor_level" nutzt die unkombinierten sensor_0.csv...sensor_7.csv
    #(2 Zeilen/Zyklus statt 1, mehr Rohsignal, aber gleiche Session-Anzahl --
    #Session bleibt in BEIDEN Pfaden der alleinige Gruppierungsschlüssel für
    #Split/CV, siehe Chat-Diskussion).
    data_variant: Literal["heater_profile_level", "sensor_level"] = "sensor_level"

    source_dir: Path = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/data/raw_input_data/Bad/") #anpassen
    #Wurzelverzeichnis der sensorbasierten Rohdaten, nur für data_variant="sensor_level"
    sensor_source_dir: Optional[Path] = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/data/raw_input_data/Bad/level1_per_sensor/") #anpassen, falls data_variant="sensor_level" genutzt wird

    #Art des Splits. Bei explicit test_data_dir angeben
    split_mode: Literal["session", "explicit"] = "session"

    #Für split_mode = "session"
    test_ratio: float = 0.25
    split_seed: int = 7

    #Optionales manuelles festlegen einzelner Sessions: {session_name: "training"/"testing"}
    explicit_test_sessions: Optional[Dict[str, str]] = None

    #Für split_mode = "explicit"
    test_data_dir: Optional[Path] = None #anpassen

    #Preprocessing: Log10-Transformation der Gaswiderstands-Features.
    #WICHTIG: Dies ist die EINZIGE Stelle, an der log10 angewendet wird.
    #Beim CSV-Export in app.py (Sidebar "3. Vorverarbeitung") MUSS die
    #Checkbox "log10-Transformation der Gaswiderstände" AUSGESCHALTET sein --
    #sonst wird de facto zweimal logarithmiert (einmal in core.py beim
    #Export, einmal hier). Die Edge-/Live-Pipeline (pc_preprocess_bridge.py)
    #repliziert exakt diesen einen Schritt und geht von rohen, unlogarithmierten
    #Gaswiderständen aus.
    log_transform: bool = True

    #Preprocessing: max. Anzahl fehlender Feature-Werte pro Zeile, die imputiert werden.
    #Zeilen mit mehr fehlenden Werten werden verworfen.
    impute_max_missing: int = 2

    #Spaltendefinitionen bei Änderung des Input-Formats anpassen
    label_column: str = "label"
    session_column: str = "session"
    heater_profile_column: str = "heater_profile_id"
    #Enthält Meta-Spalten BEIDER Varianten gleichzeitig (heater_profile_level:
    #cycle_id_a/cycle_id_b; sensor_level: sensor_index/cycle_id) -- Spalten,
    #die im jeweils anderen Format nicht vorkommen, werden bei der
    #feature_cols-Berechnung einfach ignoriert (Ausschluss per Namensabgleich,
    #kein Fehler bei Nichtvorhandensein).
    meta_columns: List[str] = field(
        default_factory=lambda: [
            "vector_id",
            "session",
            "category",
            "n_imputed",
            "heater_profile_id",
            "cycle_id_a",
            "cycle_id_b",
            "sensor_index",
            "cycle_id",
        ]
    )

    #Auswahl des Heizprofils: Eine Session enthält mehrere Heizprofil-CSVs
    #(je Messobjekt/Session-Ordner eine Datei pro Profil). Exploration und
    #ML-Pipeline laufen bewusst GETRENNT pro Profil.
    #None -> alle im Datensatz gefundenen Profile nacheinander verarbeiten.
    #"heater_322" (z.B.) -> nur dieses eine Profil.
    heater_profile: Optional[str] = "heater_413"

    # Speicherort des präparierten Datensatzes
    ml_data_dir: Path = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/data/ml_data/ml_input_data")

    # Speicherort für Reports (Exploration-Plots, Modell-Ergebnisse)
    report_dir: Path = Path("/home/tun/Projects/tuc/ML/FP-ML-repo/ml-app-lite/data/ml_data/reports")

    #Validierung / Modellauswahl (Stufe 1, auf Trainingsdaten via GroupKFold über Session)
    cv_folds: int = 3

    #Metrik zur Bestimmung des "besten" Laufs -- Platzhalter, an euer Ziel anpassen.
    #Muss ein Key aus validation.SCORING sein.
    selection_metric: str = "f1_macro"

    random_seed: int = 42

    #Hilfsfunktion zum Zurückgeben aller Spalten die nicht Features sind
    @property
    def non_feature_columns(self) -> List[str]:
        return self.meta_columns + [self.label_column]

    #Hilfsfunktion: je nach data_variant das richtige Rohdaten-Wurzelverzeichnis
    @property
    def active_source_dir(self) -> Path:
        if self.data_variant == "sensor_level":
            if self.sensor_source_dir is None:
                raise ValueError("data_variant='sensor_level' erfordert sensor_source_dir in der Config.")
            return self.sensor_source_dir
        return self.source_dir
