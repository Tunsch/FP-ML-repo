import joblib
from pathlib import Path
from config import ExperimentConfig
from data_import import split_dataset
from exploration import run_exploration
from preprocessing import preprocess_pipeline
from profiles import resolve_profiles, for_profile

def run_for_profile(config: ExperimentConfig) -> None:
    #1. Datenimport (bereits auf config.heater_profile gefiltert) und Split
    train_df, test_df, feature_cols = split_dataset(config)

    #2. Explorative Datenanalyse
    run_exploration(train_df, feature_cols, config)

    #3. Preprocessing
    X_train, X_test, y_train, y_test, groups_train, groups_test = preprocess_pipeline(
        train_df, test_df, feature_cols, config)

    #4. Speichern per joblib
    output_dir = Path(config.ml_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "groups_train": groups_train, #Session je Zeile, für sessionweise CV (z.B. GroupKFold)
        "groups_test": groups_test,
        #"preprocessor": preprocessor,
        "config": config,
    }
    joblib.dump(payload, output_dir / "prepared_data.joblib")
    print(f"[{config.heater_profile}] Prepared data saved at {output_dir}")

    #TESTCODE
    print(f"\n=== [{config.heater_profile}] Spalten im DataFrame ===")
    print(train_df.columns.tolist())

    print(f"\n=== [{config.heater_profile}] Konfigurierte Non-Features ===")
    print(config.non_feature_columns)

    print(f"\n=== [{config.heater_profile}] Fehlende Non-Features ===")
    print(set(config.non_feature_columns) - set(train_df.columns))


def main():
    base_config = ExperimentConfig()
    profiles = resolve_profiles(base_config)

    for profile in profiles:
        print(f"\n{'=' * 60}\nHeizprofil: {profile}\n{'=' * 60}")
        config = for_profile(base_config, profile)
        run_for_profile(config)

if __name__ == "__main__":
    main()

