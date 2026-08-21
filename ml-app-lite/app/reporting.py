""""Export der Validierungs-/Modellergebnisse (CSV, alle geprüften
Konfigurationen) und Plots der jeweils besten Konfiguration PRO Modelltyp."""

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from config import ExperimentConfig

RESULTS_FILENAME = "validation_results.csv"

def export_results(results_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    #Hängt aktuelle Validierungsergebnisse an die results.csv an und gibt vollständigen Verlauf zurück
    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_FILENAME

    results_df = results_df.copy()
    results_df["timestamp"] = datetime.now().isoformat(timespec="seconds")

    if results_path.exists():
        history = pd.read_csv(results_path)
        history = pd.concat([history, results_df], ignore_index=True)
    else:
        history = results_df
    history.to_csv(results_path, index=False)
    print(f"{len(results_df)} Ergebniszeile(n) angehängt an {results_path} "
          f"(gesamt: {len(history)}.")
    return history

def generate_model_plots(results_df: pd.DataFrame, best_estimators: dict[str, Any],
                      X_train: pd.DataFrame, y_train: pd.Series,
                      groups: pd.Series, config: ExperimentConfig) -> None:
    """Erzeugt für JEDEN Modelltyp in best_estimators einen Konfusionsmatrix-
    Plot bei dessen bester Konfiguration (out-of-fold via GroupKFold), unter
    modellspezifischem Dateinamen ('{model}_confusion_matrix.png').

    best_estimators: dict[model_name] -> UNGEFITTETER Estimator mit den besten
    Parametern (z.B. aus validation.run_validation, oder aus einem eigenen
    NN-Validierungsschritt mit gleichem Rückgabeformat). Diese Funktion kennt
    validation.py absichtlich nicht -- neue Modelltypen (z.B. NN) müssen hier
    nichts ändern, solange sie einen passenden Eintrag in best_estimators
    liefern.

    results_df: die (aktuellen) Validierungsergebnisse, nur genutzt, um den
    Score der besten Konfiguration je Modell für den Plot-Titel anzuzeigen --
    rein informativ, kein Einfluss auf die Auswahl."""
    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=config.cv_folds)
    metric = config.selection_metric

    for name, estimator in best_estimators.items():
        y_pred = cross_val_predict(estimator, X_train, y_train, cv=cv, groups=groups)

        score_str = ""
        if metric in results_df.columns and "is_best_for_model" in results_df.columns:
            best_row = results_df[(results_df["model"] == name) & (results_df["is_best_for_model"])]
            if len(best_row):
                score_str = f"\n{metric}={best_row.iloc[0][metric]:.3f}"

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, ax=ax)
    ax.set_title (f"{name} (Validierung, out-of-fold, beste Konfiguration){score_str}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Plot für {name} gespeichert unter {out_dir}/{name}_confusion_matrix.png")

