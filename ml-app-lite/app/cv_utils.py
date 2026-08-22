"""Gemeinsame Hilfsfunktion für validation.py und reporting.py: passt
cv_folds an die tatsächlich verfügbare Session-Anzahl je Klasse an.

Hintergrund: StratifiedGroupKFold kann eine Klasse nur dann in jedem Fold
vertreten halten, wenn sie auf mindestens so viele Sessions verteilt ist wie
es Folds gibt (eine Session kann nicht zerteilt werden). Mit mehr Folds als
das kleinste Session-Kontingent kommt es zwangsläufig zu Folds ganz ohne
diese Klasse -- daher die 'y_pred contains classes not in y_true'-Warnung
und ein dadurch verzerrter gemittelter Score.
"""
import pandas as pd


def effective_cv_folds(y: pd.Series, groups: pd.Series, requested_folds: int) -> int:
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    sessions_per_label = groups.groupby(y).nunique()
    min_sessions = int(sessions_per_label.min())

    if min_sessions < 2:
        worst_label = sessions_per_label.idxmin()
        raise ValueError(
            f"Klasse '{worst_label}' kommt nur in {min_sessions} Session(s) vor -- "
            f"sessionweise Cross-Validation braucht mindestens 2 Sessions je Klasse. "
            f"Mehr Sessions für diese Klasse sammeln oder Klasse vorerst ausschließen."
        )

    if requested_folds > min_sessions:
        worst_label = sessions_per_label.idxmin()
        print(f"WARNUNG: cv_folds={requested_folds} > kleinste Session-Anzahl je Klasse "
              f"({min_sessions}, Klasse '{worst_label}'). Reduziere cv_folds auf "
              f"{min_sessions}, damit jede Klasse grundsätzlich in jedem Fold vertreten "
              f"sein kann (Restrisiko bleibt, StratifiedGroupKFold ist ein Best-Effort-"
              f"Algorithmus, keine Garantie).")
        return min_sessions

    return requested_folds