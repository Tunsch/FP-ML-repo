"""
run_experiment.py

Minimaler CLI-Einstiegspunkt für Experimente ohne Notebook -- z.B. um
über Nacht mehrere Modelle nacheinander laufen zu lassen. Nutzt exakt
dieselbe ml.pipeline.run_experiment()-Funktion wie das Notebook; es gibt
keine zweite, parallele Implementierung der Pipeline-Logik.

Beispiel:

    python run_experiment.py --data-dir data/level2_per_profile \\
        --model random_forest --notes "Baseline"

Für systematische Vergleiche mehrerer Modelle einfach diese Datei als
Vorlage für ein eigenes kleines Batch-Skript nehmen (Liste von Configs,
Schleife über run_experiment()) -- eine feste CLI-Option pro Modell wäre
hier eher hinderlich, siehe Doku.
"""

from __future__ import annotations

import argparse
import json

from ml.config import ExperimentConfig
from ml.pipeline import run_experiment
from ml.models.registry import available_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Einzelnes ML-Experiment ausführen.")
    parser.add_argument("--data-dir", required=True, help="Ordner mit Feature-CSVs (Output von core.py)")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--model-params", default="{}", help="JSON-Dict mit Modell-Hyperparametern")
    parser.add_argument("--feature-level", default="")
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--imputation", default="unknown")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--result-file", default="experiment_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        data_dir=args.data_dir,
        feature_level=args.feature_level,
        feature_set=args.feature_set,
        imputation=args.imputation,
        test_ratio=args.test_ratio,
        seed=args.seed,
        scaling=not args.no_scaling,
        model_name=args.model,
        model_params=json.loads(args.model_params),
        notes=args.notes,
        results_dir=args.results_dir,
        result_file=args.result_file,
    )

    result = run_experiment(config)

    print(f"Experiment-ID: {result.experiment_id}")
    print(f"Accuracy={result.evaluation.accuracy:.3f}  F1={result.evaluation.f1:.3f}")
    print(result.evaluation.classification_report_text)
    print(f"Artefakte: {result.exp_dir}")


if __name__ == "__main__":
    main()
