"""
ml/config.py

Beschreibt ein einzelnes Experiment vollständig und eindeutig. Eine
ExperimentConfig-Instanz ist alles, was ml.pipeline.run_experiment()
braucht -- kein globaler Zustand, keine losen Notebook-Variablen.

Da bewusst auf YAML-Dateien verzichtet wird, entsteht eine Config direkt
im aufrufenden Code (Notebook oder run_experiment.py):

    from ml.config import ExperimentConfig
    config = ExperimentConfig(
        data_dir="data/level2_per_profile",
        model_name="svm",
        model_params={"C": 1, "kernel": "rbf"},
        scaling=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    # -- Daten --
    data_dir: Path                       # Ordner mit den Feature-CSVs (Output von core.py)
    feature_level: str = ""              # informativ, z.B. "Level 2", "Statistical Features"
    feature_set: str = "all_features"    # informativ, welche Merkmale konkret genutzt wurden
    imputation: str = "unknown"          # informativ, wie core.py fehlende Werte behandelt hat

    # -- Split --
    test_ratio: float = 0.25
    seed: int = 7

    # -- Vorverarbeitung --
    scaling: bool = True

    # -- Modell --
    model_name: str = "random_forest"    # Schlüssel in ml.models.registry
    model_params: dict[str, Any] = field(default_factory=dict)

    # -- Dokumentation --
    notes: str = ""

    # -- Tracking --
    results_dir: Path = Path("results")
    result_file: Path = Path("experiment_results.csv")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.result_file = Path(self.result_file)

    def as_dict(self) -> dict[str, Any]:
        """Für Tracking/Serialisierung -- Pfade als Strings, damit JSON-tauglich."""
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["results_dir"] = str(self.results_dir)
        d["result_file"] = str(self.result_file)
        return d
