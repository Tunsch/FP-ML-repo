import joblib
from pathlib import Path
from config import ExperimentConfig
from data_import import split_dataset
from exploration import run_exploration
from preprocessing import preprocess_pipeline

def main():
    #1. Datenimport und trennen in Training- und Test-Dataframe
    config = ExperimentConfig()
    train_df, test_df, feature_cols = split_dataset(config)

    #2. Explorative Datenanalyse
    run_exploration(train_df, feature_cols)

    #3. Preprocessing
    X_train, X_test, y_train, y_test = preprocess_pipeline(train_df, test_df, feature_cols, config)

    #4. Speichern per joblib
    output_dir = Path(config.ml_data_dir)

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        #"preprocessor": preprocessor,
        "config": config,
    }
    joblib.dump(payload, output_dir / "prepared_data.joblib")
    print(f"Prepared data saved at {output_dir}")

    #TESTCODE
    print("\n=== Spalten im DataFrame ===")
    print(train_df.columns.tolist())

    print("\n=== Konfigurierte Non-Features ===")
    print(config.non_feature_columns)

    print("\n=== Fehlende Non-Features ===")
    print(set(config.non_feature_columns) - set(train_df.columns))

if __name__ == "__main__":
    main()

