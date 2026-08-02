"""
ml/models/registry.py

Zentrale Stelle, die einen Modellnamen (String, kommt aus
ExperimentConfig.model_name) auf ein sklearn-kompatibles Modell
abbildet. Das ist der Baustein, der die Pipeline unabhängig davon macht,
welche Klassifikatoren am Ende tatsächlich zum Einsatz kommen.

Ein neues Modell hinzufügen = eine Factory-Funktion schreiben und mit
@register("name") registrieren. Kein bestehender Code (Pipeline, Config,
Evaluation, Tracking) muss dafür angefasst werden.

    @register("gradient_boosting")
    def _build_gradient_boosting(params: dict):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params)

Die Imports der einzelnen Modellklassen stehen bewusst INNERHALB der
Factory-Funktionen (nicht am Modulanfang) -- so bleibt registry.py schnell
importierbar, auch wenn irgendwann Bibliotheken wie xgboost dazukommen,
die nicht jeder installiert hat, solange dieses Modell dann nicht genutzt wird.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator

_REGISTRY: dict[str, Callable[[dict[str, Any]], BaseEstimator]] = {}


def register(name: str) -> Callable:
    """Decorator: registriert eine Factory-Funktion unter `name`."""
    def decorator(factory_fn: Callable[[dict[str, Any]], BaseEstimator]) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Modell '{name}' ist bereits registriert.")
        _REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


def build_model(name: str, params: dict[str, Any]) -> BaseEstimator:
    """Baut ein Modell über die Registry. Wirft eine sprechende
    Fehlermeldung, wenn der Name unbekannt ist -- inkl. Liste der
    verfügbaren Modelle."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(keine registriert)"
        raise KeyError(f"Unbekanntes Modell '{name}'. Verfügbar: {available}")
    return _REGISTRY[name](params)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------
# Eingebaute Modelle. Params kommen 1:1 aus ExperimentConfig.model_params,
# hier findet keine Validierung statt -- unbekannte/falsche Parameter
# wirft sklearn selbst mit einer klaren Fehlermeldung.
# ---------------------------------------------------------------------

@register("random_forest")
def _build_random_forest(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.ensemble import RandomForestClassifier
    defaults = {"n_estimators": 300, "class_weight": "balanced",
                "random_state": 42, "n_jobs": -1}
    return RandomForestClassifier(**{**defaults, **params})


@register("svm")
def _build_svm(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.svm import SVC
    defaults = {"C": 1, "kernel": "rbf"}
    return SVC(**{**defaults, **params})


@register("knn")
def _build_knn(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.neighbors import KNeighborsClassifier
    defaults = {"n_neighbors": 5}
    return KNeighborsClassifier(**{**defaults, **params})


@register("mlp")
def _build_mlp(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.neural_network import MLPClassifier
    defaults = {"hidden_layer_sizes": (64, 32), "max_iter": 1000, "random_state": 42}
    return MLPClassifier(**{**defaults, **params})


@register("logistic_regression")
def _build_logistic_regression(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.linear_model import LogisticRegression
    defaults = {"max_iter": 1000}
    return LogisticRegression(**{**defaults, **params})
