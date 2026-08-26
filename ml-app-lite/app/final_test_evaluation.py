"""Stufe 2: Finale, EINMALIGE Testauswertung.

Bestimmt das laut Validierung (Stufe 1, validation.py) beste Modell je
Heizprofil (anhand config.selection_metric in validation_results.csv) --
über RF/SVM/KNN/NN hinweg -- und wertet es GENAU EINMAL auf dem bisher
unberührten Testset aus.

WICHTIG: Dieses Skript sollte erst laufen, wenn die Modell-/Hyperparameterwahl
abgeschlossen ist. Wiederholtes Ausführen nach Config-Änderungen und erneuter
Testauswertung unterläuft den Sinn eines unabhängigen Testsets (siehe
Chat-Diskussion Validierung vs. Test) -- braucht dann eigentlich einen neuen,
bisher unberührten Holdout, um wieder eine ehrliche Zahl zu bekommen.

RF/SVM/KNN vs. NN werden hier UNTERSCHIEDLICH behandelt (siehe validation.py,
Abschnitt "WICHTIGE ASYMMETRIE" / Option A):
- RF/SVM/KNN: ungefitteter Klon mit den besten Params aus Stufe 1 wird HIER
  auf dem GESAMTEN Trainingsset neu gefittet (keine erneute CV mehr, die
  Modell-/Hyperparameterwahl ist mit Stufe 1 bereits abgeschlossen).
- NN: wird NICHT erneut trainiert. Das in Stufe 1 bereits fertig trainierte
  Modell (80/20-Split mit Early Stopping, siehe validation.py) wird per
  reporting.save_nn_artifact() als Datei geladen und direkt auf dem Testset
  ausgewertet. Ein Refit auf 100% der Daten würde wieder eine Heuristik zur
  Epochenzahl erfordern -- genau das, was Option A bewusst vermeidet. Das NN
  hat dadurch beim finalen Test real weniger Trainingsdaten gesehen als
  RF/SVM/KNN; dieser Kompromiss ist klein, transparent benannt und dem
  Alternative (ungeprüfte Epochen-Heuristik) vorzuziehen.
"""
import ast
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              ConfusionMatrixDisplay, f1_score)

from config import ExperimentConfig
from profiles import for_profile, resolve_profiles
from reporting import NN_LABEL_ENCODER_FILENAME, NN_MODEL_FILENAME, RESULTS_FILENAME
from validation import CANDIDATE_MODELS

NN_MODEL_NAME = "nn_keras"


def select_best_row(config: ExperimentConfig) -> pd.Series:
    """Liest validation_results.csv und gibt die Zeile mit dem laut
    config.selection_metric besten Ergebnis unter allen als 'beste
    Konfiguration je Modell' markierten Zeilen zurück -- über RF/SVM/KNN/NN
    hinweg, da alle in derselben CSV mit vergleichbaren Metriken stehen."""
    results_path = Path(config.report_dir) / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(f"Keine Validierungsergebnisse unter {results_path} -- "
                                 f"zuerst run_validation.py ausführen.")
    history = pd.read_csv(results_path)

    metric = config.selection_metric
    if metric not in history.columns:
        raise ValueError(f"selection_metric '{metric}' nicht in {results_path} enthalten "
                          f"(verfügbar: {list(history.columns)}).")

    candidates = history[history["is_best_for_model"]] if "is_best_for_model" in history.columns else history
    valid_models = set(CANDIDATE_MODELS.keys()) | {NN_MODEL_NAME}
    candidates = candidates[candidates["model"].isin(valid_models)]
    if candidates.empty:
        raise ValueError(f"Keine gültigen Kandidatenzeilen in {results_path} gefunden.")

    return candidates.loc[candidates[metric].idxmax()]


def build_final_estimator(best_row: pd.Series):
    """Ungefitteter Klon des besten sklearn-Modells mit dessen besten
    Parametern aus der Validierung (Stufe 1). Nur für RF/SVM/KNN -- fürs NN
    siehe load_final_nn_model()."""
    model_name = best_row["model"]
    params = ast.literal_eval(best_row["params"])
    return clone(CANDIDATE_MODELS[model_name]["estimator"]).set_params(**params)


def load_final_nn_model(config: ExperimentConfig):
    """Lädt das in Stufe 1 bereits fertig trainierte NN-Modell + den
    zugehörigen LabelEncoder von der Platte (siehe reporting.save_nn_artifact
    / validation.py, Option A). Kein Refit hier -- im Unterschied zu
    build_final_estimator() für RF/SVM/KNN."""
    from tensorflow import keras

    out_dir = Path(config.report_dir)
    model_path = out_dir / NN_MODEL_FILENAME
    encoder_path = out_dir / NN_LABEL_ENCODER_FILENAME
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Kein gespeichertes NN-Modell unter {model_path} -- "
            f"zuerst run_validation.py ausführen (Stufe 1, speichert es via "
            f"reporting.save_nn_artifact)."
        )
    model = keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


def run_for_profile(config: ExperimentConfig) -> None:
    payload_path = Path(config.ml_data_dir) / "prepared_data.joblib"
    payload = joblib.load(payload_path)
    X_train, y_train = payload["X_train"], payload["y_train"]
    X_test, y_test = payload["X_test"], payload["y_test"]

    best_row = select_best_row(config)
    model_name = best_row["model"]
    print(f"[{config.heater_profile}] Bestes Modell laut Validierung: {model_name} "
          f"(Validierungs-{config.selection_metric}={best_row[config.selection_metric]:.3f}, "
          f"Params={best_row['params']})")

    if model_name == NN_MODEL_NAME:
        #NN: KEIN Refit -- das bereits fertig trainierte Modell aus Stufe 1
        #wird direkt geladen und ausgewertet (siehe Modul-Docstring/Option A).
        model, label_encoder = load_final_nn_model(config)
        y_pred_encoded = model.predict(X_test, verbose=0).argmax(axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        print(f"[{config.heater_profile}] NN wird OHNE erneuten Refit ausgewertet "
              f"(bereits in Stufe 1 auf einem 80%-Split trainiert, siehe Docstring).")
    else:
        #RF/SVM/KNN: finaler Refit auf dem GESAMTEN Trainingsset -- keine
        #erneute CV mehr, die Modell-/Hyperparameterwahl ist mit Stufe 1
        #bereits abgeschlossen.
        estimator = build_final_estimator(best_row)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }
    print(f"[{config.heater_profile}] FINALES TESTERGEBNIS ({model_name}): {metrics}")

    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=True)
    fig.suptitle(f"{model_name} -- FINALER TEST (einmalig)\n" +
                 ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()), fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / "final_test_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"[{config.heater_profile}] Plot gespeichert unter "
          f"{out_dir}/final_test_confusion_matrix.png")

    result_row = pd.DataFrame([{
        "model": model_name,
        "params": best_row["params"],
        "validation_" + config.selection_metric: best_row[config.selection_metric],
        **{f"test_{k}": v for k, v in metrics.items()},
    }])
    result_path = out_dir / "final_test_result.csv"
    result_row.to_csv(result_path, index=False)
    print(f"[{config.heater_profile}] Ergebnis gespeichert unter {result_path}")


def main():
    base_config = ExperimentConfig()
    profiles = resolve_profiles(base_config)

    for profile in profiles:
        print(f"\n{'=' * 60}\nFinale Testauswertung Heizprofil: {profile}\n{'=' * 60}")
        config = for_profile(base_config, profile)
        run_for_profile(config)


if __name__ == "__main__":
    main()