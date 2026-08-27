import csv
import json
from datetime import datetime
from pathlib import Path
import serial
from edge_config import SERIAL_PORT, SERIAL_BAUD, DATA_DIR

LOG_FILE = DATA_DIR / "bsec_classification_log.csv"

CLASS_LABELS = {
    1: "Zähne putzen",
    2: "Rasur",
    3: "Raumspray",
    4: "Deospray"
}

CSV_FIELDS = [
    "timestamp", "predicted_class_id", "label", "confidence",
    "class_1_prob", "class_2_prob", "class_3_prob", "class_4_prob",
    "raw_gas_res", "gas_index", "temp", "humidity"
]

def ensure_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def main():
    ensure_csv(LOG_FILE)
    print(f"Verbinde zu {SERIAL_PORT} @ {SERIAL_BAUD} Baud...")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)

    print(f"Logging aktiv: {LOG_FILE.resolve()}\n")

    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "bsec_result":
            pred_id = msg.get("predicted_class")
            conf = msg.get("confidence", 0.0)
            label = CLASS_LABELS.get(pred_id, f"Unbekannt ({pred_id})")
            estimates = msg.get("estimates", {})
            raw = msg.get("raw", {})

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Klassifikation: {label:<15} | Konfidenz: {conf:.1%} | Index: {raw.get('gas_index')}")

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "predicted_class_id": pred_id or "",
                "label": label,
                "confidence": round(conf, 4) if conf is not None else "",
                "class_1_prob": estimates.get("class_1", ""),
                "class_2_prob": estimates.get("class_2", ""),
                "class_3_prob": estimates.get("class_3", ""),
                "class_4_prob": estimates.get("class_4", ""),
                "raw_gas_res": raw.get("gas_resistance", ""),
                "gas_index": raw.get("gas_index", ""),
                "temp": raw.get("temp", ""),
                "humidity": raw.get("humidity", "")
            }

            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)

        elif msg_type == "status":
            print(f"--> ESP32 Status: {msg}")

if __name__ == "__main__":
    main()