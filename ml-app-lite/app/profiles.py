"""Hilfsfunktionen, um Exploration/ML-Pipeline getrennt pro Heizprofil laufen
zu lassen. Eine Session enthält mehrere Heizprofil-CSVs -- die eigentliche
Filterung passiert in data_import.load_csv_dir (config.heater_profile);
dieses Modul kümmert sich nur darum, WELCHE Profile verarbeitet werden und
dass jedes Profil seine eigenen Ausgabe-Verzeichnisse bekommt.
"""
import dataclasses
from pathlib import Path

from config import ExperimentConfig
from data_import import discover_heater_profiles


def resolve_profiles(config: ExperimentConfig) -> list[str]:
    """Gibt zurück, welche Heizprofile verarbeitet werden sollen:
    - config.heater_profile gesetzt -> genau dieses eine
    - config.heater_profile is None -> alle im Datensatz gefundenen Profile
    """
    if config.heater_profile is not None:
        return [config.heater_profile]
    profiles = discover_heater_profiles(config.source_dir, config.heater_profile_column)
    print(f"Gefundene Heizprofile ({len(profiles)}): {profiles}")
    return profiles


def for_profile(config: ExperimentConfig, profile: str) -> ExperimentConfig:
    """Kopie der Config für genau ein Heizprofil: heater_profile gesetzt,
    ml_data_dir/report_dir in ein Unterverzeichnis je Profil verlegt, damit
    sich mehrere Profile beim Batch-Durchlauf nicht gegenseitig überschreiben."""
    return dataclasses.replace(
        config,
        heater_profile=profile,
        ml_data_dir=Path(config.ml_data_dir) / profile,
        report_dir=Path(config.report_dir) / profile,
    )