"""Export der Validierungs-/Modellergebnisse (CSV, alle geprüften
Konfigurationen) und Plots der jeweils besten Konfiguration PRO Modelltyp.

Für RF/SVM/KNN gilt weiterhin: `generate_model_plots` erzeugt den Plot per
`cross_val_predict` auf UNGEFITTETEN Klonen (siehe validation.py).

Fürs NN gilt das NICHT (siehe validation.py, Abschnitt "WICHTIGE ASYMMETRIE"
/ Option A): das übergebene `nn_artifact["model"]` ist bereits fertig
trainiert (ein 80/20-Split mit Early Stopping, kein Refit). Ein erneutes
`cross_val_predict` würde das Modell mehrfach neu klonen/fitten und damit
genau die Heuristik umgehen, die wir bewusst vermeiden wollten. Deshalb gibt
es dafür die eigene Funktion `generate_nn_plot`, die auf dem bereits
vorhandenen Holdout-Split des NN plottet, sowie `save_nn_artifact`, um das
Modell für die spätere, separate Testauswertung (final_test_evaluation.py)
auf die Platte zu schreiben."""
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from config import ExperimentConfig
from cv_utils import effective_cv_folds

RESULTS_FILENAME = "validation_results.csv"
NN_MODEL_FILENAME = "nn_keras_model.keras"
NN_LABEL_ENCODER_FILENAME = "nn_keras_label_encoder.joblib"


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


def generate_nn_plot(nn_artifact: dict[str, Any], results_df: pd.DataFrame,
                      X_train: pd.DataFrame, y_train: pd.Series,
                      config: ExperimentConfig) -> None:
    """Konfusionsmatrix-Plot fürs NN -- Pendant zu generate_model_plots(),
    aber bewusst OHNE cross_val_predict: nn_artifact["model"] ist bereits
    fertig trainiert (ein sessionweiser 80/20-Split mit Early Stopping,
    siehe validation.py, Option A), daher wird hier direkt auf dessen eigenem
    Holdout-Anteil geplottet (nn_artifact["val_indices"]) statt das teure
    NN-Training mehrfach für eine eigene CV zu wiederholen."""
    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_idx = nn_artifact["val_indices"]
    X_val = X_train.iloc[val_idx]
    y_val_true = y_train.iloc[val_idx]

    y_val_pred_encoded = nn_artifact["model"].predict(X_val)
    y_val_pred = nn_artifact["label_encoder"].inverse_transform(y_val_pred_encoded)

    metric = config.selection_metric
    score_str = ""
    if metric in results_df.columns and "is_best_for_model" in results_df.columns:
        best_row = results_df[(results_df["model"] == "nn_keras") & (results_df["is_best_for_model"])]
        if len(best_row):
            score_str = f"\n{metric}={best_row.iloc[0][metric]:.3f}"

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_predictions(y_val_true, y_val_pred, ax=ax, colorbar=True)
    fig.suptitle(f"nn_keras -- Validierung (eigener Holdout-Split){score_str}", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "nn_keras_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Plot für nn_keras gespeichert unter {out_dir}/nn_keras_confusion_matrix.png "
          f"(Holdout-Split, kein cross_val_predict -- siehe Docstring).")


def save_nn_artifact(nn_artifact: dict[str, Any], config: ExperimentConfig) -> None:
    """Persistiert das bereits fertig trainierte NN-Modell + den zugehörigen
    LabelEncoder, damit final_test_evaluation.py sie in einem späteren,
    separaten Skriptlauf laden kann (das NN wird dort NICHT erneut
    trainiert/refittet, siehe validation.py, Option A)."""
    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / NN_MODEL_FILENAME
    encoder_path = out_dir / NN_LABEL_ENCODER_FILENAME

    nn_artifact["model"].model_.save(model_path)
    joblib.dump(nn_artifact["label_encoder"], encoder_path)
    print(f"NN-Modell gespeichert unter {model_path}, LabelEncoder unter {encoder_path}.")