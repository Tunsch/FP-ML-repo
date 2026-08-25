from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from config import ExperimentConfig
from cv_utils import effective_cv_folds
from profiles import resolve_profiles, for_profile
from reporting import export_results

# Kleine, bewusst überschaubare Hyperparametersuche.
TUNING_ARCHITECTURES = [ [12], [20], [32], [64], [16, 8], [20, 10], [32, 16], [64, 32], [20, 10, 5]]
TUNING_LEARNING_RATES = [0.0001, 0.001, 0.005]
TUNING_DROPOUTS = [0.0, 0.2, 0.4, 0.5]


def build_model(
    n_features: int,
    n_classes: int,
    hidden_layers: list[int],
    learning_rate: float = 1e-3,
    dropout: float = 0.0,
) -> keras.Model:
    layers = [keras.layers.Input(shape=(n_features,))]
    for units in hidden_layers:
        layers.append(keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",  # erwartet Integer-Klassenlabels
        metrics=["accuracy"],
    )
    return model


def run_for_profile(config: ExperimentConfig) -> None:
    payload_path = Path(config.ml_data_dir) / "prepared_data.joblib"
    payload = joblib.load(payload_path)
    # X_train ist bereits durch preprocess_pipeline log-transformiert, imputiert
    # und skaliert (StandardScaler) -- für NN-Training genau richtig, keine
    # weitere Vorverarbeitung nötig.
    X_train = payload["X_train"]
    y_train_raw = payload["y_train"]
    groups_train = payload["groups_train"]

    # Klassenlabels robust auf 0..n_classes-1 abbilden (unabhängig davon, ob
    # die Originallabels z.B. Strings oder nicht-fortlaufende Integer sind).
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    n_classes = len(label_encoder.classes_)

    # Sessionweiser UND klassenstratifizierter Train/Validation-Split.
    # Dieselbe Methode wie in validation.py/reporting.py (StratifiedGroupKFold),
    # hier aber nur EIN Split statt k-facher CV -- siehe Begründung im Chat:
    # NN-Training ist teuer, volle CV unverhältnismäßig. config.cv_folds steuert
    # (wie überall sonst) den Anteil, wird aber über effective_cv_folds an die
    # tatsächlich verfügbare Session-Anzahl je Klasse angepasst -- sonst würde
    # z.B. bei nur 3 Trainingssessions ein hartkodiertes n_splits=5 abstürzen.
    n_splits = effective_cv_folds(y_train, groups_train, config.cv_folds)
    print(f"Validierungssplit: n_splits={n_splits} (≈{100/n_splits:.0f}% Validierungsanteil)")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
    train_idx, val_idx = next(splitter.split(X_train, y_train, groups=groups_train))
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    def train_and_evaluate(hidden_layers: list[int], learning_rate: float, dropout: float) -> dict:
        model = build_model(
            n_features=X_train.shape[1], n_classes=n_classes,
            hidden_layers=hidden_layers, learning_rate=learning_rate, dropout=dropout,
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200,
                  batch_size=32, callbacks=[early_stopping], verbose=0)
        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
        return {
            "hidden_layers": hidden_layers, "learning_rate": learning_rate, "dropout": dropout,
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_macro": f1_score(y_val, y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        }

    tuning_results = []

    # 1. Architektur
    print("\nTuning 1/3: Architektur")
    for hidden_layers in TUNING_ARCHITECTURES:
        result = train_and_evaluate(hidden_layers, 1e-3, 0.0)
        tuning_results.append(result)
        print(result)
    best_architecture = max(tuning_results, key=lambda r: r["f1_macro"])["hidden_layers"]

    # 2. Learning Rate
    print("\nTuning 2/3: Learning Rate")
    lr_results = []
    for learning_rate in TUNING_LEARNING_RATES:
        result = train_and_evaluate(best_architecture, learning_rate, 0.0)
        tuning_results.append(result)
        lr_results.append(result)
        print(result)
    best_learning_rate = max(lr_results, key=lambda r: r["f1_macro"])["learning_rate"]

    # 3. Dropout
    print("\nTuning 3/3: Dropout")
    dropout_results = []
    for dropout in TUNING_DROPOUTS:
        result = train_and_evaluate(best_architecture, best_learning_rate, dropout)
        tuning_results.append(result)
        dropout_results.append(result)
        print(result)
    best = max(dropout_results, key=lambda r: r["f1_macro"])
    best_params = {"hidden_layers": best["hidden_layers"],
                   "learning_rate": best["learning_rate"], "dropout": best["dropout"]}

    print("\nBeste NN-Konfiguration:")
    print(best_params)

    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_results).to_csv(out_dir / "nn_keras_tuning.csv", index=False)

    # Bestes Modell mit den gewählten Parametern erneut trainieren -- DIESER
    # Lauf dient der VALIDIERUNG (Metriken, Konfusionsmatrix): bewusst noch auf
    # dem 80/20-Split, weil wir für eine ehrliche Konfusionsmatrix/Metrik
    # weiterhin Vorhersagen auf ungesehenen Daten brauchen.
    model = build_model(n_features=X_train.shape[1], n_classes=n_classes, **best_params)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=32,
                         callbacks=[early_stopping], verbose=2)

    y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "f1_macro": f1_score(y_val, y_val_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
    }
    print("Validierungsmetriken (NN, Holdout):", metrics)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        label_encoder.inverse_transform(y_val),
        label_encoder.inverse_transform(y_val_pred), ax=ax,
    )
    ax.set_title("nn_keras (Validierung, Holdout, sessionweise)\n" +
                 ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    fig.tight_layout()
    fig.savefig(out_dir / "nn_keras_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert unter {out_dir}/nn_keras_confusion_matrix.png")

    results_row = pd.DataFrame([{
        "model": "nn_keras", "params": str(best_params),
        "is_best_for_model": True, **metrics,
    }])
    export_results(results_row, config)

    # Finaler Refit auf dem GESAMTEN Trainingsset (analog zu den sklearn-
    # Modellen in final_test_evaluation.py) -- für bessere Vergleichbarkeit.
    # Ohne Validierungsanteil kann EarlyStopping hier nicht mehr auf val_loss
    # überwachen; stattdessen wird die Epochenzahl übernommen, bei der der
    # obige Validierungslauf laut val_loss am besten war (Standardvorgehen:
    # Modellauswahl/Epochenwahl auf dem Holdout, finales Training mit fixer
    # Epochenzahl auf allen verfügbaren Daten).
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nFinaler Refit auf vollständigem Trainingsset ({len(X_train)} Zeilen), "
          f"{best_epoch} Epochen (aus Validierungslauf übernommen, kein EarlyStopping mehr).")

    final_model = build_model(n_features=X_train.shape[1], n_classes=n_classes, **best_params)
    final_model.fit(X_train, y_train, epochs=best_epoch, batch_size=32, verbose=2)

    model_path = out_dir / "nn_keras_model.keras"
    final_model.save(model_path)
    print(f"Finales (auf vollem Trainingsset trainiertes) Modell gespeichert unter {model_path}")
    print("Hinweis: Die oben berichteten Validierungsmetriken/Konfusionsmatrix stammen weiterhin "
          "vom 80/20-Holdout-Modell, NICHT vom hier gespeicherten finalen Modell -- das finale "
          "Modell hat mehr Daten gesehen und wurde daher (wie die sklearn-Modelle) nicht erneut "
          "auf ungesehenen Daten evaluiert.")


def main():
    base_config = ExperimentConfig()
    profiles = resolve_profiles(base_config)

    for profile in profiles:
        print(f"\n{'=' * 60}\nNN-Training Heizprofil: {profile}\n{'=' * 60}")
        config = for_profile(base_config, profile)
        run_for_profile(config)


if __name__ == "__main__":
    main()