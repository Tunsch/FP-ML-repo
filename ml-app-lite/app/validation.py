"""Stufe 1: Modellauswahl + Hyperparameter-Suche per Cross-Validation auf den
Trainingsdaten. Das Testset wird hier bewusst NICHT angefasst.

Pro Modelltyp läuft eine eigene Hyperparameter-Suche mit sessionweiser
GroupKFold-CV. Die jeweils beste Konfiguration jedes Modelltyps wird als eine
Zeile in die Ergebnistabelle geschrieben -- so bleiben RF/SVM/KNN/NN direkt
vergleichbar.

Änderungen gegenüber der Vorversion (siehe Chat):
- RF nutzt RandomizedSearchCV statt GridSearchCV (720 Kombinationen im Grid
  sind für eine vollständige Suche unverhältnismäßig teuer; nach Bergstra &
  Bengio 2012 liefert Random Search bei vielen Hyperparametern meist
  vergleichbar gute Ergebnisse mit deutlich weniger Auswertungen).
  SVM/KNN haben kleine Grids und bleiben bei GridSearchCV (vollständig, leicht
  zu erklären).
- Das neuronale Netz (Keras) läuft nicht mehr in einem separaten Skript mit
  eigener, sequenzieller Tuning-Logik (train_nn.py, jetzt obsolet), sondern
  über den scikeras-Wrapper im GENAU GLEICHEN CV-Rahmen wie RF/SVM/KNN. Das
  vereinheitlicht die Methodik und vermeidet, dass Architektur/Lernrate/
  Dropout nacheinander statt gemeinsam optimiert werden.
- WICHTIGE ASYMMETRIE (bewusst, siehe Docstring von run_validation): Für
  RF/SVM/KNN liefert diese Funktion UNGEFITTETE Estimator-Klone zum späteren
  Refit auf dem gesamten Trainingsset. Für das NN ist das nicht sinnvoll:
  Early Stopping braucht einen Validierungsanteil, und ein Refit auf 100% der
  Daten ohne Validierungssplit erfordert sonst eine Heuristik zur
  Epochenzahl (z.B. "übernimm die beste Epoche aus dem Holdout-Lauf"), die
  wir bewusst NICHT mehr verwenden, weil sie auf einer ungeprüften Annahme
  beruht (die ideale Epochenzahl bei 100% der Daten muss nicht dieselbe sein
  wie bei 80%). Stattdessen ist das im 80/20-Split mit Early Stopping
  trainierte Modell direkt das finale NN-Modell -- kein zusätzlicher
  Refit-Schritt. Das NN sieht dadurch real ~80% statt 100% der
  Trainingsdaten; dieser Kompromiss ist klein und lässt sich im Gegensatz zur
  Heuristik sauber benennen und begründen.

Daraus folgt: `best_estimators` (unfitted, für Refit auf 100%) enthält nur
RF/SVM/KNN. Das NN wird separat als bereits FERTIG TRAINIERTES Modell
zurückgegeben (siehe `nn_artifact`). Code, der bislang alle Modelltypen
gleich behandelt hat (Klonen + Refit auf komplettem Trainingsset, siehe
reporting.py / Stufe 2), muss für das NN-Ergebnis entsprechend angepasst
werden: `nn_artifact["model"]` direkt für die Testset-Evaluation verwenden,
nicht erneut klonen/fitten.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedGroupKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config import ExperimentConfig
from cv_utils import effective_cv_folds

#Platzhalter -- Parametergitter/-verteilungen fachlich/literaturbasiert anpassen.
#search_type "grid" -> GridSearchCV (vollständig, für kleine Grids).
#search_type "random" -> RandomizedSearchCV mit n_iter Ziehungen (für große Grids).
CANDIDATE_MODELS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 500, 750],
            "max_depth": [5, 10, 20, 30, 50],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
            "min_samples_leaf": [1, 2, 3, 4, 5, 10],
        },
        "search_type": "random",
        "n_iter": 50, #Anzahl geprüfter Kombinationen statt aller 720 im Grid
    },
    "svm": {
        "estimator": SVC(probability=True),
        "param_grid": {
            "C": [0.1, 1, 10, 30, 50, 70, 100, 200, 1000],
            "kernel": ["rbf", "linear"],
            "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, "scale", "auto"],
        },
        "search_type": "grid",
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        },
        "search_type": "grid",
    },
}

#Verfügbare Metriken. config.selection_metric muss einer dieser Keys sein.
SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}

#NN-Hyperparameterraum für die scikeras-Suche (ersetzt die alte, sequenzielle
#Architektur->LR->Dropout-Logik aus train_nn.py durch eine gemeinsame Suche).
NN_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "hidden_layers": [(12,), (20,), (32,), (64,), (16, 8), (20, 10), (32, 16), (64, 32), (20, 10, 5)],
    "learning_rate": [0.0001, 0.001, 0.005],
    "dropout": [0.0, 0.2, 0.4, 0.5],
}
NN_N_ITER = 20 #Anzahl geprüfter Kombinationen (statt 9+3+4=16 sequenziellen Läufen vorher -- vergleichbarer Aufwand, aber als gemeinsame statt sequenzielle Suche)
NN_MAX_EPOCHS = 200
NN_EARLY_STOPPING_PATIENCE = 20
NN_BATCH_SIZE = 32
#Anteil der jeweiligen CV-Trainingsfolds, der intern für Early Stopping
#während der Suche abgezweigt wird. Bewusst NICHT sessionweise gruppiert --
#das würde die Suche stark verkomplizieren, und für die reine
#Early-Stopping-Regularisierung während der Suche ist das ein akzeptabler
#Kompromiss: die eigentliche Modellbewertung (welche Kombination "gewinnt")
#erfolgt weiterhin über den sessionweise gruppierten CV-Score, nicht über
#diesen internen Split.
NN_SEARCH_VALIDATION_SPLIT = 0.15


def _build_nn_model(hidden_layers=(32,), learning_rate: float = 1e-3,
                     dropout: float = 0.0, meta=None):
    """Baut ein Keras-Sequential-Modell. Wird von scikeras' KerasClassifier
    aufgerufen; `meta` liefert u.a. n_features_in_ und n_classes_ automatisch
    passend zu den übergebenen Trainingsdaten.
    """
    from tensorflow import keras

    n_features = meta["n_features_in_"]
    n_classes = meta["n_classes_"]

    layers = [keras.layers.Input(shape=(n_features,))]
    for units in hidden_layers:
        layers.append(keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy", #erwartet Integer-Klassenlabels
        metrics=["accuracy"],
    )
    return model


def _run_nn_search(X_train: pd.DataFrame, y_train_encoded: np.ndarray, groups: pd.Series,
                    cv: StratifiedGroupKFold, config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hyperparametersuche fürs NN im selben CV-Rahmen wie RF/SVM/KNN.
    Gibt (results_rows, best_params) zurück. Trainiert hier NUR zu
    Such-/Bewertungszwecken -- das eigentliche finale Modell entsteht separat
    in run_validation() (siehe Docstring/Option A oben).
    """
    from scikeras.wrappers import KerasClassifier
    from tensorflow import keras

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NN_EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
    clf = KerasClassifier(
        model=_build_nn_model,
        hidden_layers=(32,), learning_rate=1e-3, dropout=0.0, #Startwerte, werden von der Suche überschrieben
        epochs=NN_MAX_EPOCHS, batch_size=NN_BATCH_SIZE, verbose=0,
        validation_split=NN_SEARCH_VALIDATION_SPLIT, callbacks=[early_stopping],
    )

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=NN_PARAM_DISTRIBUTIONS,
        n_iter=NN_N_ITER,
        scoring=SCORING,
        refit=False, #wir wollen hier nur die besten Params, nicht den refitteten Estimator (siehe Option A)
        cv=cv,
        random_state=config.random_seed,
    )
    search.fit(X_train, y_train_encoded, groups=groups)

    rows: list[dict[str, Any]] = []
    n_candidates = len(search.cv_results_["params"])
    #Bei refit=False gibt es kein best_index_ -- wir bestimmen ihn selbst
    #über die gewählte Selektionsmetrik, analog zu GridSearchCV(refit=...).
    metric_key = f"mean_test_{config.selection_metric}"
    best_idx = int(np.argmax(search.cv_results_[metric_key]))

    for i in range(n_candidates):
        row: dict[str, Any] = {
            "model": "nn_keras",
            "params": str(search.cv_results_["params"][i]),
            "is_best_for_model": (i == best_idx),
        }
        for metric in SCORING:
            row[metric] = search.cv_results_[f"mean_test_{metric}"][i]
            row[f"{metric}_std"] = search.cv_results_[f"std_test_{metric}"][i]
        rows.append(row)

    best_params = search.cv_results_["params"][best_idx]

    print(f"nn_keras: {n_candidates} Kombinationen geprüft, beste Params {best_params}, " +
          ", ".join(f"{m}={search.cv_results_[f'mean_test_{m}'][best_idx]:.3f}" for m in SCORING))

    return rows, best_params


def _fit_final_nn_model(X_train: pd.DataFrame, y_train_encoded: np.ndarray, groups: pd.Series,
                         best_params: dict[str, Any], config: ExperimentConfig) -> dict[str, Any]:
    """Trainiert das finale NN-Modell mit den besten gefundenen Hyperparametern.

    Option A (siehe Chat/Docstring oben): Es gibt KEIN zusätzliches Refit auf
    100% der Trainingsdaten mit geratener Epochenzahl. Stattdessen ist der
    sessionweise gruppierte 80/20-Split mit echtem Early Stopping bereits das
    finale Modell -- transparent, ohne zusätzliche Heuristik.
    """
    from scikeras.wrappers import KerasClassifier
    from tensorflow import keras

    n_splits = effective_cv_folds(y_train_encoded, groups, config.cv_folds)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
    train_idx, val_idx = next(splitter.split(X_train, y_train_encoded, groups=groups))
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NN_EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
    final_clf = KerasClassifier(
        model=_build_nn_model,
        epochs=NN_MAX_EPOCHS, batch_size=NN_BATCH_SIZE, verbose=2,
        callbacks=[early_stopping],
        **best_params,
    )
    final_clf.fit(X_tr, y_tr, validation_data=(X_val, y_val))

    print(f"Finales NN-Modell: sessionweiser {n_splits}-facher Split verwendet "
          f"(1 Fold als Holdout, ≈{100 / n_splits:.0f}% der Daten), Early Stopping auf val_loss. "
          f"Kein weiterer Refit auf dem vollständigen Trainingsset (siehe Docstring/Option A).")

    return {
        "model": final_clf, #FERTIG TRAINIERT -- nicht erneut klonen/fitten
        "params": best_params,
        "val_indices": val_idx, #zur Nachvollziehbarkeit: welche Zeilen als Holdout dienten
    }


def run_validation(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
                    config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Führt für jeden Kandidaten-Modelltyp eine Hyperparametersuche mit
    sessionweiser GroupKFold-CV durch (RF: RandomizedSearchCV, SVM/KNN:
    GridSearchCV, NN: scikeras + RandomizedSearchCV im selben CV-Rahmen).

    Gibt zurück:
    - results_df: EINE ZEILE PRO GEPRÜFTER PARAMETERKOMBINATION (nicht nur die
      beste), mit Spalte 'is_best_for_model' (True für die beste Kombination
      je Modelltyp). Enthält RF/SVM/KNN/NN einheitlich.
    - best_estimators: dict[model_name] -> UNGEFITTETER Estimator mit den
      besten Parametern, NUR für RF/SVM/KNN (zum Klonen für OOF-Plots oder
      zum finalen Refit auf dem kompletten Trainingsset, siehe reporting.py /
      Stufe 2).
    - nn_artifact: dict mit dem bereits FERTIG TRAINIERTEN NN-Modell
      ("model"), den gewählten Hyperparametern ("params") und den Indizes des
      Holdout-Splits ("val_indices"). Bewusst kein unfitted Estimator zum
      Refit -- siehe Modul-Docstring, Abschnitt "WICHTIGE ASYMMETRIE".
      Downstream-Code muss `nn_artifact["model"]` direkt für die
      Testset-Evaluation verwenden.
    """
    cv = StratifiedGroupKFold(n_splits=effective_cv_folds(y_train, groups, config.cv_folds))
    rows: list[dict[str, Any]] = []
    best_estimators: dict[str, Any] = {}

    for name, spec in CANDIDATE_MODELS.items():
        if spec["search_type"] == "random":
            search = RandomizedSearchCV(
                estimator=spec["estimator"],
                param_distributions=spec["param_grid"],
                n_iter=spec["n_iter"],
                scoring=SCORING,
                refit=config.selection_metric,
                cv=cv,
                random_state=config.random_seed,
            )
        else:
            search = GridSearchCV(
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                scoring=SCORING,
                refit=config.selection_metric, #entscheidet, welche Kombi als "beste" gilt
                cv=cv,
            )
        search.fit(X_train, y_train, groups=groups)

        n_candidates = len(search.cv_results_["params"])
        for i in range(n_candidates):
            row: dict[str, Any] = {
                "model": name,
                "params": str(search.cv_results_["params"][i]),
                "is_best_for_model": (i == search.best_index_),
            }
            for metric in SCORING:
                row[metric] = search.cv_results_[f"mean_test_{metric}"][i]
                row[f"{metric}_std"] = search.cv_results_[f"std_test_{metric}"][i]
            rows.append(row)

        #Unfitted Klon mit besten Params -- NICHT search.best_estimator_ (der
        #ist schon auf ganz X_train gefittet und für OOF-Plots ungeeignet).
        best_estimators[name] = clone(spec["estimator"]).set_params(**search.best_params_)

        print(f"{name}: {n_candidates} Kombinationen geprüft, beste Params {search.best_params_}, " +
              ", ".join(f"{m}={search.cv_results_[f'mean_test_{m}'][search.best_index_]:.3f}" for m in SCORING))

    #NN separat: eigene Fit-Konvention (Integer-Labels, Keras-Callbacks) und
    #eigene Behandlung des finalen Modells (siehe Option A oben).
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    nn_rows, nn_best_params = _run_nn_search(X_train, y_train_encoded, groups, cv, config)
    rows.extend(nn_rows)

    nn_artifact = _fit_final_nn_model(X_train, y_train_encoded, groups, nn_best_params, config)
    nn_artifact["label_encoder"] = label_encoder

    return pd.DataFrame(rows), best_estimators, nn_artifact