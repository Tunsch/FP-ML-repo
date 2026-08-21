#Modellauswahl und Hyperparameter-Suche per Cross-Validation auf den Trainingsdaten
from typing import Any
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import ExperimentConfig

#Parameter grid anpassen
#Je Eintrag: Estimator (unfitted) + zugehöriges paran_grid für GridsearchCV

CANDIDATE_MODELS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [1, 2, 3, 4, 5],
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
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        },
    },
}

#Verfügbare Metriken aus der eine per config.selection_metric je das beste Modell bestimmt
SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    }

def run_validation(X_train: pd.DataFrame, y_train: pd.Series,
                   groups: pd.Series, config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    #Führt für jeden Kandidaten-Modelltyp ein GridSearchCV mit sessionweiser GroupKFold-CV durch.
    #Gibt zurück: results_df -> je eine Zeile pro Modelltyp
    #best_estimators: ungefitteter estimator mit den besten Parametern
    #StratifiedGroupKFold, um Klassen in den Folds zu balancen

    cv = StratifiedGroupKFold(n_splits=config.cv_folds)
    rows: list[dict[str, Any]] = []
    best_estimators: dict[str, Any] = {}

    for name, spec in CANDIDATE_MODELS.items():
        gs = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring=SCORING,
            refit=config.selection_metric,
            cv=cv,
        )
        gs.fit(X_train, y_train, groups=groups)

        n_candidates = len(gs.cv_results_["params"])
        for i in range(n_candidates):
            row: dict[str, Any] = {
                "model": name,
                "params": gs.cv_results_["params"][i],
                "is_best_for_model": (i == gs.best_index_),
            }
            for metric in SCORING:
                row[metric] = gs.cv_results_[f"mean_test_{metric}"][i]
                row[f"{metric}_std"] = gs.cv_results_[f"std_test_{metric}"][i]
            rows.append(row)

        #Unfitted Klon mit den besten Parametern
        best_estimators[name] = clone[spec["estimator"]].set_params(**gs.best_params_)

        print(f"{name}: {n_candidates} Kombinationen geprüft, beste Params {gs.best_params_}, " +
              ", ".join(f"{m}={gs.cv_results_[f'mean_test_{m}'][gs.best_index_]:.3f}" for m in SCORING))


    return pd.DataFrame(rows), best_estimators




