import joblib
from pathlib import Path
from config import ExperimentConfig
from data_import import load_feature_table, split_dataframe
from exploration import run_exploration
from preprocessing import preprocess_pipeline

def main():
    #1. Datenimport und trennen in Training- und Test-Dataframe
    config = ExperimentConfig()
    raw_data = load_feature_table(config.source_dir)
    train_df, test_df = split_dataframe(raw_data, config)

    #2. Explorative Datenanalyse
    run_exploration(train_df)

    #3. Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(train_df, test_df, config)

    #4. Speichern per joblib
    output_dir = Path(config.ml_data_dir)

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "config": config,
    }
    joblib.dump(payload, output_dir / "prepared_data.joblib")
    print(f"Prepared data saved at {output_dir}")

if __name__ == "__main__":
    main()

