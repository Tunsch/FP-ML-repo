"""
ml/tracking.py

Alles rund um das Persistieren eines Experiment-Ergebnisses:

  - Experiment-ID vergeben
  - Ergebnisordner results/<experiment_id>/ anlegen (für Artefakte, die
    NICHT dieses Modul erzeugt -- z.B. Plots aus dem Notebook)
  - config.json + report.txt in diesem Ordner ablegen
  - eine Zeile an experiment_results.csv anhängen
  - bestehende Ergebnisse wieder einlesen (für den Modellvergleich)

Dies ist die einzige Stelle, die weiss, WIE ein Ergebnis gespeichert wird.
Der Rest der Pipeline erzeugt nur Daten (ExperimentResult), ohne sich um
Dateiformat oder Ablagepfade zu kümmern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core import sanitize_label
from ml.config import ExperimentConfig
from ml.evaluation import EvaluationResult


@dataclass
class ExperimentResult:
    experiment_id: str
    config: ExperimentConfig
    evaluation: EvaluationResult
    train_time: float
    predict_time: float
    n_train: int
    n_test: int
    n_train_sessions: int
    n_test_sessions: int
    exp_dir: Path
    # für Plots im Notebook mitgegeben, nicht Teil der CSV-Zeile:
    y_test: Any = field(repr=False)
    y_pred: Any = field(repr=False)
    model: Any = field(repr=False)

    def as_csv_row(self) -> dict[str, Any]:
        """Flache Darstellung für experiment_results.csv -- ein
        Dictionary aus Config-Feldern + Metriken, keine Objekte."""
        ev = self.evaluation
        return {
            "Timestamp": pd.Timestamp.now(),
            "Experiment ID": self.experiment_id,
            "Model": self.config.model_name,
            "Parameters": json.dumps(self.model.get_params()),
            "Feature Level": self.config.feature_level,
            "Feature Set": self.config.feature_set,
            "Imputation": self.config.imputation,
            "Scaling": self.config.scaling,
            "Test Ratio": self.config.test_ratio,
            "Seed": self.config.seed,
            "Train Vectors": self.n_train,
            "Test Vectors": self.n_test,
            "Train Sessions": self.n_train_sessions,
            "Test Sessions": self.n_test_sessions,
            "Accuracy": ev.accuracy,
            "Precision": ev.precision,
            "Recall": ev.recall,
            "F1": ev.f1,
            "False Positive Rate": ev.false_positive_rate,
            "Training Time": self.train_time,
            "Prediction Time": self.predict_time,
            "Notes": self.config.notes,
        }


def generate_experiment_id(model_name: str) -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{sanitize_label(model_name)}"


def make_exp_dir(results_dir: Path, experiment_id: str) -> Path:
    exp_dir = Path(results_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_artifacts(result: ExperimentResult) -> None:
    """Legt config.json und report.txt in result.exp_dir ab. Plots
    (Confusion Matrix, Feature Importance) speichert bewusst NICHT dieses
    Modul, sondern das Notebook -- in denselben Ordner (result.exp_dir),
    damit alle Artefakte eines Experiments beisammen bleiben."""
    with open(result.exp_dir / "config.json", "w") as f:
        json.dump(result.config.as_dict(), f, indent=2, ensure_ascii=False)
    with open(result.exp_dir / "report.txt", "w") as f:
        f.write(result.evaluation.classification_report_text)


def append_to_csv(result_file: Path, row: dict[str, Any]) -> None:
    result_file = Path(result_file)
    result_df = pd.DataFrame([row])
    if result_file.exists():
        old = pd.read_csv(result_file)
        result_df = pd.concat([old, result_df], ignore_index=True)
    result_df.to_csv(result_file, index=False)


def load_results(result_file: Path) -> pd.DataFrame:
    """Für den Modellvergleich im Notebook: gesamte Experiment-Historie."""
    return pd.read_csv(Path(result_file))
