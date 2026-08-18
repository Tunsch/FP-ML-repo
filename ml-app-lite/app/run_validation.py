#Validierung/Modellauswahl
from pathlib import Path

import joblib

from config import ExperimentConfig
from validation import run_validation
from reporting import export_results, generate_model_plots

def main():
    config = ExperimentConfig()

    payload_path = Path(config.ml_data_dir) / "prepared_data.joblib"
    payload = joblib.load(payload_path)
    X_train = payload["X_train"]
    y_train = payload["y_train"]
    groups_train = payload["groups_train"]
    print(f"Präparierte Daten geladen aus {payload_path} ({len(X_train)} Zeilen.")

    results, best_estimators = run_validation(X_train, y_train, groups_train, config)
    export_results(results, config)
    generate_model_plots(results, best_estimators, X_train, y_train, groups_train, config)

if __name__ == "__main__":
    main()
