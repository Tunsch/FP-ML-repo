"""
ml/evaluation.py

Berechnet Metriken aus Vorhersagen -- bewusst OHNE jegliches Plotting.
Visualisierung (Confusion-Matrix-Heatmap, Feature-Importance-Balken, ...)
ist Aufgabe des Notebooks, nicht dieses Moduls. Dadurch bleibt evaluate()
auch in einem Batch-Skript ohne Display nutzbar und lässt sich isoliert
testen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationResult:
    labels: list                       # sortierte Klassenliste, für Confusion-Matrix-Achsen
    confusion_matrix: np.ndarray
    classification_report_text: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float         # NaN bei >2 Klassen
    feature_importances: pd.Series | None  # None, wenn Modell keine unterstützt


def evaluate(model: BaseEstimator, y_test: np.ndarray, y_pred: np.ndarray,
             feature_names: list[str]) -> EvaluationResult:
    """Nimmt ein bereits trainiertes Modell sowie y_test/y_pred entgegen
    (Training passiert bewusst außerhalb, in ml/pipeline.py, damit die
    Trainingszeit dort zentral gemessen wird) und berechnet alle Metriken."""
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0,
    )

    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
    else:
        fpr = float("nan")

    report_text = classification_report(y_test, y_pred, zero_division=0)

    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

    return EvaluationResult(
        labels=labels,
        confusion_matrix=cm,
        classification_report_text=report_text,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        feature_importances=feature_importances,
    )
