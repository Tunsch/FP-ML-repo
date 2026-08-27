"""
edge_config.py
---------------
Minimale, eigenständige Konfiguration fürs Edge-Testing (ESP32 +
pc_preprocess_bridge.py) -- bewusst OHNE Abhängigkeit von der ML-
Trainings-Pipeline (config.py / ExperimentConfig aus dem ML-Repo). Diese
Datei ist alles, was pc_preprocess_bridge.py braucht.

Das Preprocessing-Artefakt (data/preprocessing_artifact.json) entsteht
einmalig als Nebenprodukt des Trainings (preprocessing.export_scaler_artifact
im ML-Repo) und wird von dort in den data/-Ordner hier kopiert bzw. direkt
dorthin exportiert (out_path-Argument). Danach braucht ihr für Edge-Tests
das ML-Repo nicht mehr anzufassen -- dieser Ordner ist eigenständig.
"""
from pathlib import Path

# Serielle Verbindung zum ESP32-S3
SERIAL_PORT = "/dev/ttyACM0"   # Windows z.B. "COM5", macOS z.B. "/dev/cu.usbserial-XXXX"
SERIAL_BAUD = 115200

# Datenordner: Preprocessing-Artefakt (einmalig aus dem Training kopiert) +
# laufendes CSV-Log der Testsessions. Wird bei Bedarf automatisch angelegt.
DATA_DIR = Path(__file__).parent / "data"
ARTIFACT_PATH = DATA_DIR / "preprocessing_artifact.json"
CSV_LOG_PATH = DATA_DIR / "classification_log.csv"

# Heizprofil hat immer 10 Stufen -- fixer Wert (wie im .ino)
N_EXPECTED_STEPS = 10
