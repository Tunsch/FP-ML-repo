"""Stufe 1: Modellauswahl + Hyperparameter-Suche per Cross-Validation auf den
Trainingsdaten. Das Testset wird hier bewusst NICHT angefasst.

Pro Modelltyp läuft ein eigenes GridSearchCV mit sessionweiser GroupKFold-CV.
Die jeweils beste Konfiguration jedes Modelltyps wird als eine Zeile in die
Ergebnistabelle geschrieben -- so bleiben RF/SVM/KNN (und später NN, siehe
unten) direkt vergleichbar.
"""
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import ExperimentConfig
from cv_utils import effective_cv_folds

#Platzhalter -- Parametergitter fachlich/literaturbasiert anpassen.
#Jeder Eintrag: estimator (unfitted) + zugehöriges param_grid für GridSearchCV.
#Das NN läuft bewusst separat über train_nn.py (Keras) -- siehe Chat.
CANDIDATE_MODELS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
            "min_samples_leaf": [1, 2, 5, 10],
        },
    },
    "svm": {
        "estimator": SVC(probability=True),
        "param_grid": {
            "C": [0.1, 1, 10, 100, 1000],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    },
}

#Verfügbare Metriken. config.selection_metric muss einer dieser Keys sein.
SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}


def run_validation(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
                    config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Führt für jeden Kandidaten-Modelltyp ein GridSearchCV mit sessionweiser
    GroupKFold-CV durch. Gibt zurück:
    - results_df: EINE ZEILE PRO GEPRÜFTER PARAMETERKOMBINATION (nicht nur die
      beste), mit Spalte 'is_best_for_model' (True für die beste Kombination
      je Modelltyp).
    - best_estimators: dict[model_name] -> UNGEFITTETER Estimator mit den
      besten Parametern (zum Klonen für OOF-Plots oder zum finalen Refit auf
      dem kompletten Trainingsset, siehe reporting.py / Stufe 2).
    """
    cv = StratifiedGroupKFold(n_splits=effective_cv_folds(y_train, groups, config.cv_folds))
    rows: list[dict[str, Any]] = []
    best_estimators: dict[str, Any] = {}

    for name, spec in CANDIDATE_MODELS.items():
        gs = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring=SCORING,
            refit=config.selection_metric, #entscheidet, welche Kombi als "beste" gilt
            cv=cv,
        )
        gs.fit(X_train, y_train, groups=groups)

        n_candidates = len(gs.cv_results_["params"])
        for i in range(n_candidates):
            row: dict[str, Any] = {
                "model": name,
                "params": str(gs.cv_results_["params"][i]),
                "is_best_for_model": (i == gs.best_index_),
            }
            for metric in SCORING:
                row[metric] = gs.cv_results_[f"mean_test_{metric}"][i]
                row[f"{metric}_std"] = gs.cv_results_[f"std_test_{metric}"][i]
            rows.append(row)

        #Unfitted Klon mit besten Params -- NICHT gs.best_estimator_ (der ist
        #schon auf ganz X_train gefittet und für OOF-Plots ungeeignet).
        best_estimators[name] = clone(spec["estimator"]).set_params(**gs.best_params_)

        print(f"{name}: {n_candidates} Kombinationen geprüft, beste Params {gs.best_params_}, " +
              ", ".join(f"{m}={gs.cv_results_[f'mean_test_{m}'][gs.best_index_]:.3f}" for m in SCORING))

    return pd.DataFrame(rows), best_estimators