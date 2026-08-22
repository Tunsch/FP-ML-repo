#Validierung/Modellauswahl
#Läuft für jedes Profil aus profiles.resolve_profiles() separat, mit dessen eigenem ml_data_dir/report_dir
#(siehe profiles.for_profile).

from pathlib import Path

import joblib

from config import ExperimentConfig
from profiles import resolve_profiles, for_profile
from reporting import export_results, generate_model_plots
from validation import run_validation


def run_for_profile(config: ExperimentConfig) -> None:
    payload_path = Path(config.ml_data_dir) / "prepared_data.joblib"
    payload = joblib.load(payload_path)
    X_train = payload["X_train"]
    y_train = payload["y_train"]
    groups_train = payload["groups_train"]
    print(f"[{config.heater_profile}] Präparierte Daten geladen aus {payload_path} "
          f"({len(X_train)} Zeilen).")

    results, best_estimators = run_validation(X_train, y_train, groups_train, config)
    export_results(results, config)
    generate_model_plots(results, best_estimators, X_train, y_train, groups_train, config)


def main():
    base_config = ExperimentConfig()
    profiles = resolve_profiles(base_config)

    for profile in profiles:
        print(f"\n{'=' * 60}\nValidierung Heizprofil: {profile}\n{'=' * 60}")
        config = for_profile(base_config, profile)
        run_for_profile(config)


if __name__ == "__main__":
    main()
