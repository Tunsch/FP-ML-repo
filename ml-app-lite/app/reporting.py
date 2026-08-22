"""Export der Validierungs-/Modellergebnisse (CSV, alle geprüften
Konfigurationen) und Plots der jeweils besten Konfiguration PRO Modelltyp."""
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from config import ExperimentConfig
from cv_utils import effective_cv_folds

RESULTS_FILENAME = "validation_results.csv"


def export_results(results_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Hängt die aktuellen Validierungsergebnisse (alle geprüften
    Konfigurationen) an die Ergebnis-CSV an (erzeugt sie beim ersten Aufruf)
    und gibt den vollständigen Verlauf zurück."""
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
          f"(gesamt: {len(history)}).")
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
    cv = StratifiedGroupKFold(n_splits=effective_cv_folds(y_train, groups, config.cv_folds))
    metric = config.selection_metric

    for name, estimator in best_estimators.items():
        y_pred = cross_val_predict(estimator, X_train, y_train, groups=groups, cv=cv)

        score_str = ""
        if metric in results_df.columns and "is_best_for_model" in results_df.columns:
            best_row = results_df[(results_df["model"] == name) & (results_df["is_best_for_model"])]
            if len(best_row):
                score_str = f"\n{metric}={best_row.iloc[0][metric]:.3f}"

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred, ax=ax, colorbar=True)
        #Titel kompakter (keine Wiederholung von "Konfiguration") und in
        #eigener Zeile über der Achse statt zentriert über Achse+Colorbar,
        #damit er bei langen Modellnamen nicht in die Colorbar hineinragt.
        fig.suptitle(f"{name} -- Validierung (out-of-fold){score_str}", fontsize=11)
        #x-Achsenbeschriftungen schräg stellen, damit sie sich bei längeren
        #Klassennamen nicht gegenseitig überlappen.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(rect=[0, 0, 1, 0.94]) #Platz für suptitle freihalten
        fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"Plot für {name} gespeichert unter {out_dir}/{name}_confusion_matrix.png")