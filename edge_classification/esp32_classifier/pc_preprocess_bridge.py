"""
pc_preprocess_bridge.py
------------------------
Der BME688 haengt am ESP32-S3 (nicht am PC). Ablauf:

  ESP32 (Sensor + Heizprofil)  --raw_cycle-->  PC (dieses Skript)
  ESP32                        <--features---  PC
  ESP32 (run_classifier / Random Forest)  --result-->  PC (Anzeige + CSV-Log)

Zwei Modelle, zwei unterschiedliche Erwartungen an die Vorverarbeitung:

  - Edge-Impulse-Modell: bringt seine eigene StandardScaler-Normalisierung
    als DSP-Block mit (siehe model_variables.h, DATA_NORMALIZATION_METHOD_
    STANDARD_SCALER, Mittelwert/Std fest im Modell eingebacken). Es braucht
    daher NUR log10-transformierte, UNSKALIERTE Werte -- die Skalierung
    passiert automatisch on-device in run_classifier().
  - Random Forest / euer eigenes NN: hat KEINE eingebaute Normalisierung.
    Braucht die voll vorverarbeiteten Werte, also log10 + StandardScaler,
    exakt wie preprocess_pipeline() sie beim Training erzeugt (siehe
    preprocessing.py / preprocessing_artifact.json).

Diese Bridge berechnet deshalb BEIDE Varianten aus demselben Rohzyklus und
schickt beide in einer Antwort:
  - "log"    -> Eingabe fuer run_classifier() (Edge-Impulse-Modell)
  - "scaled" -> Eingabe fuer Random Forest / euer eigenes NN

Reihenfolge der 10 Werte muss der Heizstufen-Reihenfolge entsprechen
(gasres_step0 ... gasres_step9) -- das ist automatisch der Fall, weil
esp32_classifier.ino den Rohzyklus schon in dieser Reihenfolge sendet.

1. Empfaengt einen vollstaendigen, rohen Heizzyklus vom ESP32 als JSON-Zeile:
     {"type":"raw_cycle","values":[123.4, 567.8, ...]}   (rohe Ohm-Werte)
2. Wendet GENAU EINMAL log10 an (mit Clipping gegen log10(0)).
3. Skaliert zusaetzlich mit dem beim Training gefitteten StandardScaler
   (scaler_mean/scaler_scale aus preprocessing_artifact.json).
4. Sendet beide Varianten zurueck:
     {"type":"features","log":[5.11, 4.98, ...],"scaled":[0.12,-0.34, ...]}
5. Empfaengt das Ergebnis vom ESP32 (Edge-Impulse-Label + Random-Forest-
   Label), zeigt es an UND schreibt eine Zeile ins CSV-Log (siehe unten).

Es wird NICHT imputiert: unvollstaendige Zyklen soll bereits der ESP32 gar
nicht erst als raw_cycle verschicken (siehe esp32_classifier.ino).

CSV-LOGGING
-----------
Jede vollstaendige Runde (Rohzyklus -> Vorverarbeitung -> Klassifikation)
wird als eine Zeile in CSV_LOG_PATH angehaengt (Datei wird beim ersten Start
mit Header angelegt, danach immer angehaengt -- Historie geht nicht verloren,
auch ueber mehrere Programmstarts hinweg). Spalten: Zeitstempel, die 10
Rohwerte, die 10 log-Werte, die 10 skalierten Werte, Edge-Impulse-Label +
Konfidenz + alle Klassen-Scores (als JSON-String), Random-Forest-Label.
Auch Timeouts/Fehler vom ESP32 werden als Zeile mit Fehlerkennung geloggt,
damit in der Nachverfolgung sichtbar bleibt, dass ein Zyklus verworfen wurde.

KEINE ABHAENGIGKEIT ZUR ML-PIPELINE: Alle Pfade/Einstellungen kommen aus
edge_config.py (im selben Ordner) -- nicht aus config.py/ExperimentConfig
des ML-Trainings-Repos. Das Preprocessing-Artefakt in data/ ist zwar
urspruenglich ein Nebenprodukt des Trainings, aber sobald es einmal in
data/preprocessing_artifact.json liegt, braucht ihr fuer Edge-Tests das
ML-Repo nicht mehr.

Anpassungsbedarf vor dem ersten Lauf (alles in edge_config.py):
- SERIAL_PORT auf euren tatsaechlichen ESP32-Port setzen.
- data/preprocessing_artifact.json muss vorhanden sein (einmalig aus dem
  Training kopiert bzw. direkt dorthin exportiert).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import serial  # pip install pyserial

from edge_config import (
    ARTIFACT_PATH, CSV_LOG_PATH, N_EXPECTED_STEPS, SERIAL_BAUD, SERIAL_PORT,
)

CSV_FIELDNAMES = (
    ["timestamp"]
    + [f"raw_step{i}" for i in range(N_EXPECTED_STEPS)]
    + [f"log_step{i}" for i in range(N_EXPECTED_STEPS)]
    + [f"scaled_step{i}" for i in range(N_EXPECTED_STEPS)]
    + ["ei_label", "ei_confidence", "ei_scores_json", "rf_label", "note"]
)


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Kopiert das Preprocessing-Artefakt "
            f"einmalig aus dem ML-Training (preprocessing_artifact.json) "
            f"nach {path}, oder passt ARTIFACT_PATH in edge_config.py an."
        )
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    required = ["n_expected_steps", "log_transform", "log_clip_eps",
                "scaler_mean", "scaler_scale"]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise ValueError(f"Preprocessing-Artefakt {path} unvollstaendig, fehlt: {missing}")
    return artifact


def preprocess(raw_values: list[float], artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    """Gibt (log_values, scaled_values) zurueck.
    log_values    -> fuer das Edge-Impulse-Modell (skaliert bereits selbst)
    scaled_values -> fuer Random Forest / euer eigenes NN (keine eingebaute
                      Normalisierung)
    """
    values = np.array(raw_values, dtype=float)
    if len(values) != artifact["n_expected_steps"]:
        raise ValueError(
            f"Erwartet {artifact['n_expected_steps']} Werte, erhalten {len(values)}."
        )

    log_values = values
    if artifact["log_transform"]:
        log_values = np.log10(np.clip(values, artifact["log_clip_eps"], None))

    mean = np.array(artifact["scaler_mean"])
    scale = np.array(artifact["scaler_scale"])
    scaled_values = (log_values - mean) / scale

    return log_values, scaled_values


def ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def append_csv_row(path: Path, raw_values, log_values, scaled_values,
                    ei_label=None, ei_confidence=None, ei_scores=None,
                    rf_label=None, note="") -> None:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for i in range(N_EXPECTED_STEPS):
        row[f"raw_step{i}"] = raw_values[i] if raw_values is not None else ""
        row[f"log_step{i}"] = round(float(log_values[i]), 6) if log_values is not None else ""
        row[f"scaled_step{i}"] = round(float(scaled_values[i]), 6) if scaled_values is not None else ""
    row["ei_label"] = ei_label or ""
    row["ei_confidence"] = f"{ei_confidence:.4f}" if ei_confidence is not None else ""
    row["ei_scores_json"] = json.dumps(ei_scores, ensure_ascii=False) if ei_scores is not None else ""
    row["rf_label"] = rf_label or ""
    row["note"] = note
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def main():
    artifact = load_artifact()
    print(f"Artefakt geladen: {artifact['n_expected_steps']} Heizstufen, "
          f"log_transform={artifact['log_transform']}, "
          f"Heizprofil '{artifact.get('heater_profile')}'.")
    print("Sende pro Zyklus BEIDE Varianten: 'log' (Edge Impulse) und "
          "'scaled' (Random Forest / eigenes NN).")

    ensure_csv_header(CSV_LOG_PATH)
    print(f"CSV-Log: {CSV_LOG_PATH.resolve()}")

    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=None)
    print(f"Verbunden mit {SERIAL_PORT}. Warte auf Rohzyklen vom ESP32 ...")

    # Zwischenspeicher fuer den zuletzt verarbeiteten Zyklus, damit beim
    # Eintreffen von "result" (oder eines Fehler-Status) die passenden
    # Roh-/log-/skalierten Werte mit ins CSV geloggt werden koennen.
    last_raw_values = None
    last_log_values = None
    last_scaled_values = None

    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"Ignoriere nicht-JSON-Zeile: {line!r}")
            continue

        msg_type = msg.get("type")

        if msg_type == "raw_cycle":
            try:
                log_values, scaled_values = preprocess(msg["values"], artifact)
            except (KeyError, ValueError) as e:
                print(f"Preprocessing-Fehler: {e}")
                continue

            last_raw_values = msg["values"]
            last_log_values = log_values
            last_scaled_values = scaled_values

            response = {
                "type": "features",
                "log": log_values.round(6).tolist(),
                "scaled": scaled_values.round(6).tolist(),
            }
            ser.write((json.dumps(response) + "\n").encode("utf-8"))

        elif msg_type == "result":
            ei_label = msg.get("label")
            ei_confidence = msg.get("confidence")
            ei_scores = msg.get("scores")
            rf_label = msg.get("rf_label")

            print(f"Edge Impulse: {ei_label} ({ei_confidence:.1%})  |  "
                  f"Random Forest: {rf_label}  |  "
                  f"EI-Scores: {ei_scores}")

            append_csv_row(
                CSV_LOG_PATH, last_raw_values, last_log_values, last_scaled_values,
                ei_label=ei_label, ei_confidence=ei_confidence,
                ei_scores=ei_scores, rf_label=rf_label,
            )

        elif msg_type == "status":
            print(f"ESP32-Status: {msg}")
            status = msg.get("status", "")
            if status in ("features_timeout", "classifier_failed",
                          "bme688_init_failed", "config_mismatch"):
                # Fehlerhafter/verworfener Zyklus -- trotzdem loggen, damit in
                # der Historie sichtbar bleibt, dass hier etwas schiefging.
                append_csv_row(
                    CSV_LOG_PATH, last_raw_values, last_log_values, last_scaled_values,
                    note=f"ERROR:{status}",
                )

        else:
            print(f"Unbekannter Nachrichtentyp, ignoriert: {msg}")


if __name__ == "__main__":
    main()