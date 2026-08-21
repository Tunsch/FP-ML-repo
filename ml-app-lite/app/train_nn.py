from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from config import ExperimentConfig
from reporting import export_results

#Einfache Architektur (2 Dense-Hidden-Layer), ggf. anpassen
HIDDEN_LAYERS = [32, 16]

def build_model(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_features,))]
        + [keras.layers.Dense(units, activation="relu") for units in HIDDEN_LAYERS]
        + [keras.layers.Dense(n_classes, activation="softmax")]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    config = ExperimentConfig()

    payload_path = Path(config.ml_data_dir) / "prepared_data.joblib"
    payload = joblib.load(payload_path)
    #Input data
    X_train = payload["X_train"]
    y_train_raw = payload["y_train"]
    groups_train = payload["groups_train"]

    #Label kodieren
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    n_classes = len(label_encoder.classes_)

    #Split sessionweise und stratifiziert für ausgeglichene Klassenverteilung über Train/Test
    #ca. 20 % Validierungsanteil mit 1 aus 5 folds
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    train_idx, valid_idx = next(splitter.split(X_train, y_train, groups=groups_train))
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train[train_idx], y_train[valid_idx]

    model = build_model(n_features=X_train.shape[1], n_classes=n_classes)
    model.summary()

    #Early Stopping, um Overfitting zu vermeiden, ohne Epochen selbst zu tunen
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

    #Validierungsmetriken
    y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "f1_macro": f1_score(y_val, y_val_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
    }
    print("Validierungsmetriken (NN, Holdout):", metrics)

    #Export
    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #Konfusionsmatrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        label_encoder.inverse_transform(y_val),
        label_encoder.inverse_transform(y_val_pred),
        ax=ax,
    )
    ax.set_title("nn_keras (Validierung, Holdout, sessionsweise)\n" +
                ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    fig.tight_layout()
    fig.savefig((out_dir / "nn_keras_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert unter {out_dir}/nn_keras_confusion_matrix.png")

    #Ergebniszeile im selben Spaltenformat wie validation.py
    #hier nur EIN Holdout-Split statt k-facher CV, kein *_std vorhanden
    results_row = pd.DataFrame([{
        "model": "nn_keras",
        "params": str({"hidden_layers": HIDDEN_LAYERS, "optimizer": "adam"}),
        "is_best_for_model": True,
        **metrics,
    }])
    export_results(results_row, config)

    model_path = out_dir / "nn_keras_model.keras"
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()





