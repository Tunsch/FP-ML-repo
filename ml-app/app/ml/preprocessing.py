"""
ml/preprocessing.py

Reine fit/transform-Funktionen ohne globalen Zustand. Aktuell nur
Skalierung, aber bewusst als eigenes Modul, damit sich weitere Schritte
(z.B. PCA, Feature-Selektion) ergänzen lassen, ohne ml/pipeline.py oder
andere Module anzufassen.

Der Scaler wird -- wie schon im Notebook -- IMMER nur auf den
Trainingsdaten gefittet, nie auf Test- oder Gesamtdaten.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray, scaling: bool) -> Optional[StandardScaler]:
    """Gibt einen gefitteten StandardScaler zurück, oder None, wenn
    scaling=False (dann wird in apply_scaler() nichts verändert)."""
    if not scaling:
        return None
    return StandardScaler().fit(X_train)


def apply_scaler(scaler: Optional[StandardScaler], X: np.ndarray) -> np.ndarray:
    """Wendet einen zuvor gefitteten Scaler an, oder gibt X unverändert
    zurück, wenn scaler None ist (scaling=False)."""
    if scaler is None:
        return X
    return scaler.transform(X)
