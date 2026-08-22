"""Stufe 2: Finale, EINMALIGE Testauswertung.

Bestimmt das laut Validierung (Stufe 1, validation.py) beste sklearn-Modell
je Heizprofil (anhand config.selection_metric in validation_results.csv),
fittet es auf dem GESAMTEN Trainingsset neu und wertet es GENAU EINMAL auf
dem bisher unberührten Testset aus.

WICHTIG: Dieses Skript sollte erst laufen, wenn die Modell-/Hyperparameterwahl
abgeschlossen ist. Wiederholtes Ausführen nach Config-Änderungen und erneuter
Testauswertung unterläuft den Sinn eines unabhängigen Testsets (siehe
Chat-Diskussion Validierung vs. Test) -- braucht dann eigentlich einen neuen,
bisher unberührten Holdout, um wieder eine ehrliche Zahl zu bekommen.

NN (nn_keras, aus train_nn.py) wird hier bewusst NICHT automatisch mit
ausgewertet -- das Modell ist bereits vorab trainiert (nicht auf dem vollen
Trainingsset) und liegt als .keras-Datei vor. Für einen fairen Vergleich mit
den sklearn-Modellen (die hier auf ganz X_train neu gefittet werden) siehe
Hinweis am Ende dieser Datei / im Chat.
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
from reporting import RESULTS_FILENAME
from validation import CANDIDATE_MODELS


def select_best_row(config: ExperimentConfig) -> pd.Series:
    """Liest validation_results.csv und gibt die Zeile mit dem laut
    config.selection_metric besten Ergebnis unter allen als 'beste
    Konfiguration je Modell' markierten Zeilen zurück."""
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
    #Nur sklearn-Kandidaten aus validation.py -- nn_keras absichtlich außen vor
    #(siehe Docstring oben), damit der automatische Refit nicht versehentlich
    #versucht, ein Keras-Modell wie ein sklearn-Objekt zu behandeln.
    candidates = candidates[candidates["model"].isin(CANDIDATE_MODELS.keys())]
    if candidates.empty:
        raise ValueError(f"Keine sklearn-Kandidatenzeilen in {results_path} gefunden.")

    return candidates.loc[candidates[metric].idxmax()]


def build_final_estimator(best_row: pd.Series):
    """Ungefitteter Klon des besten Modells mit dessen besten Parametern aus
    der Validierung (Stufe 1)."""
    model_name = best_row["model"]
    params = ast.literal_eval(best_row["params"])
    return clone(CANDIDATE_MODELS[model_name]["estimator"]).set_params(**params)


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

    #Finaler Refit auf dem GESAMTEN Trainingsset -- keine erneute CV mehr,
    #die Modell-/Hyperparameterwahl ist mit Stufe 1 bereits abgeschlossen.
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