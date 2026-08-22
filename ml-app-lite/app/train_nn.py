from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from config import ExperimentConfig
from profiles import resolve_profiles, for_profile
from reporting import export_results

# Architektur -- bewusst klein/einfach gehalten (2 Dense-Hidden-Layer).
HIDDEN_LAYERS = [32, 16]


def build_model(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential(
        [keras.layers.Input(shape=(n_features,))]
        + [keras.layers.Dense(units, activation="relu") for units in HIDDEN_LAYERS]
        + [keras.layers.Dense(n_classes, activation="softmax")]
    )
    model.compile(
        optimizer="adam",
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
    # NN-Training ist teuer, volle CV unverhältnismäßig. n_splits=5 ergibt
    # einen ~20%-Validierungsanteil; wir nehmen nur den ersten der 5 Folds.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    train_idx, val_idx = next(splitter.split(X_train, y_train, groups=groups_train))
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = build_model(n_features=X_train.shape[1], n_classes=n_classes)
    model.summary()

    # Early Stopping statt manueller Epochenwahl -- Standardvorgehen, um
    # Overfitting zu vermeiden, ohne die Epochenzahl selbst tunen zu müssen.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2,
    )

    # Validierungsmetriken, dieselben wie SCORING in validation.py
    y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "f1_macro": f1_score(y_val, y_val_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
    }
    print("Validierungsmetriken (NN, Holdout):", metrics)

    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Konfusionsmatrix im selben Stil wie generate_model_plots (reporting.py),
    # aber mit Originallabels (inverse_transform) für Lesbarkeit im Plot.
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        label_encoder.inverse_transform(y_val),
        label_encoder.inverse_transform(y_val_pred),
        ax=ax,
    )
    ax.set_title("nn_keras (Validierung, Holdout, sessionweise)\n" +
                 ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    fig.tight_layout()
    fig.savefig(out_dir / "nn_keras_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert unter {out_dir}/nn_keras_confusion_matrix.png")

    # Ergebniszeile im selben Spaltenformat wie validation.py, damit sie sich
    # bei Bedarf in derselben CSV mit RF/SVM/KNN vergleichen lässt. ACHTUNG:
    # hier nur EIN Holdout-Split statt k-facher CV -- kein *_std vorhanden,
    # die Schätzung ist dadurch weniger stabil als bei den anderen Modellen.
    results_row = pd.DataFrame([{
        "model": "nn_keras",
        "params": str({"hidden_layers": HIDDEN_LAYERS, "optimizer": "adam"}),
        "is_best_for_model": True,
        **metrics,
    }])
    export_results(results_row, config)

    model_path = out_dir / "nn_keras_model.keras"
    model.save(model_path)
    print(f"Modell gespeichert unter {model_path}")


def main():
    base_config = ExperimentConfig()
    profiles = resolve_profiles(base_config)

    for profile in profiles:
        print(f"\n{'=' * 60}\nNN-Training Heizprofil: {profile}\n{'=' * 60}")
        config = for_profile(base_config, profile)
        run_for_profile(config)


if __name__ == "__main__":
    main()






