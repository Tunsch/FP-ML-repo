"""
ml/pipeline.py

run_experiment(config) ist der einzige Einstiegspunkt, den Notebook oder
CLI kennen müssen. Diese Funktion -- und nur sie -- kennt die Reihenfolge
data -> preprocessing -> models -> evaluation -> tracking. Jeder einzelne
Schritt ist in seinem eigenen Modul isoliert austauschbar, ohne dass sich
diese Datei ändert (z.B. ein weiterer Preprocessing-Schritt, ein neues
Modell in der Registry, eine zusätzliche Metrik in evaluation.py).
"""

from __future__ import annotations

import time

from ml import data, preprocessing, tracking
from ml.config import ExperimentConfig
from ml.evaluation import evaluate
from ml.models.registry import build_model
from ml.tracking import ExperimentResult


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # 1. Daten laden & splitten
    df, feature_cols = data.load_feature_table(config.data_dir)
    train_df, test_df = data.split_data(df, config)

    X_train, y_train = data.get_xy(train_df, feature_cols)
    X_test, y_test = data.get_xy(test_df, feature_cols)

    # 2. Vorverarbeitung (nur auf Trainingsdaten gefittet)
    scaler = preprocessing.fit_scaler(X_train, config.scaling)
    X_train_m = preprocessing.apply_scaler(scaler, X_train)
    X_test_m = preprocessing.apply_scaler(scaler, X_test)

    # 3. Modell aus der Registry bauen, trainieren, vorhersagen
    model = build_model(config.model_name, config.model_params)

    t0 = time.perf_counter()
    model.fit(X_train_m, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test_m)
    predict_time = time.perf_counter() - t0

    # 4. Auswertung (reine Metriken, kein Plotting)
    ev = evaluate(model, y_test, y_pred, feature_cols)

    # 5. Tracking: ID vergeben, Ordner anlegen, Artefakte + CSV-Zeile schreiben
    experiment_id = tracking.generate_experiment_id(config.model_name)
    exp_dir = tracking.make_exp_dir(config.results_dir, experiment_id)

    result = ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        evaluation=ev,
        train_time=train_time,
        predict_time=predict_time,
        n_train=len(X_train),
        n_test=len(X_test),
        n_train_sessions=train_df["session"].nunique(),
        n_test_sessions=test_df["session"].nunique(),
        exp_dir=exp_dir,
        y_test=y_test,
        y_pred=y_pred,
        model=model,
    )

    tracking.save_artifacts(result)
    tracking.append_to_csv(config.result_file, result.as_csv_row())

    return result
