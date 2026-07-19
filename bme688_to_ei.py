#!/usr/bin/env python3
"""
bme688_to_ei.py

Wandelt eine BME688-Rohdaten-Session-CSV (8 Sensoren, bis zu 4 Heizprofile,
je Heizprofil mehrere Heizstufen pro Scanning Cycle) in einzelne
Edge-Impulse-taugliche Feature-Vektor-CSVs um.

Erwartete Eingabe-Spalten (wie im Beispiel-Export):
    sensor_index, sensor_id, timestamp_since_poweron, real_time_clock,
    temperature, pressure, relative_humidity, resistance_gassensor,
    heater_profile_step_index, scanning_enabled, scanning_cycle_index,
    label_tag, error_code, heater_profile_id, label_name

Es werden nur die Gaswiderstandswerte (resistance_gassensor) verwendet.
Da resistance_gassensor über mehrere Größenordnungen streut (typ. 1e3 - 1e7 Ohm,
v.a. bedingt durch den ersten "Purge/Burn-in"-Schritt jedes Zyklus), wird
standardmäßig log10 angewendet (--no-log zum Abschalten).

Es werden drei Aggregationsstufen erzeugt (Level wählbar über --level):
  1 = pro Sensor pro Scanning Cycle           -> 1 Vektor mit n_steps Features
  2 = pro Heizprofil pro Scanning Cycle        -> Vektor aus den 2 Sensoren,
                                                   die dasselbe Profil fahren
                                                   (je nach --level2-mode
                                                   concat oder mean)
  3 = über alle Sensoren (synchronisierter     -> Vektor aus allen 8 Sensoren,
      "Super-Zyklus")                             ausgerichtet am jeweils
                                                   letzten vollständigen
                                                   Zyklus jedes Sensors

Output: pro erzeugtem Vektor eine CSV-Datei im Edge-Impulse "single reading"
Format (1 Header-Zeile, 1 Datenzeile, kein Timestamp), mit dem Dateinamen-
Präfix "<label>." damit der Studio-/CLI-Uploader das Label automatisch
erkennt: <label>.<name>.csv

Beispiel:
    python3 bme688_to_ei.py session1.csv --outdir out --level 1 2 3
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "sensor_index", "sensor_id", "timestamp_since_poweron",
    "resistance_gassensor", "heater_profile_step_index",
    "heater_profile_id", "label_name",
]


def sanitize_label(label: str) -> str:
    """Edge-Impulse-Labels: keine Leerzeichen/Sonderzeichen, damit das
    Dateinamen-Präfix-Schema <label>.<name>.csv sauber funktioniert."""
    label = label.strip()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_\-]", "", label)
    return label


def load_session(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {csv_path}: {missing}")
    return df


def detect_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt pro sensor_index eine fortlaufende cycle_id. Die im Export
    vorhandene scanning_cycle_index-Spalte ist in der Praxis oft nicht
    verlässlich befüllt, daher wird der Zyklus stattdessen über den
    Wrap-Around des heater_profile_step_index erkannt (Schrittindex
    springt beim Zyklusstart wieder auf einen kleineren Wert zurück)."""
    df = df.sort_values(["sensor_index", "timestamp_since_poweron"]).reset_index(drop=True)
    out = []
    for sensor_index, g in df.groupby("sensor_index", sort=False):
        g = g.sort_values("timestamp_since_poweron").reset_index(drop=True)
        wrap = g["heater_profile_step_index"].diff() <= 0
        wrap.iloc[0] = True  # erste Zeile startet Zyklus 0
        g["cycle_id"] = wrap.cumsum() - 1
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_step_vectors(df: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """Baut pro (sensor_index, cycle_id) einen vollständigen Gaswiderstands-
    Vektor über alle Heizstufen. Unvollständige Zyklen (z.B. am Anfang/Ende
    der Aufnahme oder durch Datenlücken) werden verworfen."""
    rows = []
    for (sensor_index, cycle_id), g in df.groupby(["sensor_index", "cycle_id"]):
        steps = sorted(g["heater_profile_step_index"].unique())
        expected = list(range(g["heater_profile_step_index"].max() + 1))
        # nur vollständige, doppelfreie Zyklen verwenden
        if len(g) != len(expected) or steps != expected:
            continue
        g = g.sort_values("heater_profile_step_index")
        values = g["resistance_gassensor"].to_numpy(dtype=float)
        if log_transform:
            values = np.log10(np.clip(values, 1e-3, None))
        row = {
            "sensor_index": sensor_index,
            "sensor_id": g["sensor_id"].iloc[0],
            "heater_profile_id": g["heater_profile_id"].iloc[0],
            "cycle_id": cycle_id,
            "start_ts": g["timestamp_since_poweron"].iloc[0],
            "end_ts": g["timestamp_since_poweron"].iloc[-1],
            "label": g["label_name"].iloc[0],
        }
        for i, v in enumerate(values):
            row[f"gasres_step{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def write_sample(out_dir: Path, label: str, name: str, feature_row: dict, feature_cols: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    label = sanitize_label(label)
    fname = f"{label}.{name}.csv"
    path = out_dir / fname
    header = ",".join(feature_cols)
    values = ",".join(f"{feature_row[c]:.6f}" for c in feature_cols)
    path.write_text(header + "\n" + values + "\n")


def level1(vectors: pd.DataFrame, out_dir: Path, session_tag: str = ""):
    """Ein Feature-Vektor pro Sensor pro Scanning Cycle."""
    step_cols = [c for c in vectors.columns if c.startswith("gasres_step")]
    n = 0
    for _, row in vectors.iterrows():
        name = f"{session_tag}_s{row.sensor_index}_{row.heater_profile_id}_c{row.cycle_id}"
        write_sample(out_dir, row.label, name, row.to_dict(), step_cols)
        n += 1
    print(f"[Level 1] {n} Samples nach {out_dir}")


def level2(vectors: pd.DataFrame, out_dir: Path, mode: str = "concat", session_tag: str = ""):
    """Ein Feature-Vektor pro Heizprofil pro Scanning Cycle, gebildet aus
    den zwei Sensoren, die dasselbe Profil fahren. Die Zyklen der beiden
    Sensoren werden per nächstliegendem Startzeitpunkt gematcht."""
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    n = 0
    for profile, g in vectors.groupby("heater_profile_id"):
        sensors = sorted(g["sensor_index"].unique())
        if len(sensors) != 2:
            print(f"[Level 2] Warnung: Profil {profile} hat {len(sensors)} statt 2 Sensoren, übersprungen")
            continue
        a = g[g.sensor_index == sensors[0]].sort_values("start_ts").reset_index(drop=True)
        b = g[g.sensor_index == sensors[1]].sort_values("start_ts").reset_index(drop=True)
        for _, ra in a.iterrows():
            # nächstgelegenen Zyklus von Sensor b finden (per Startzeitpunkt)
            idx = (b["start_ts"] - ra.start_ts).abs().idxmin()
            rb = b.loc[idx]
            if abs(rb.start_ts - ra.start_ts) > 5000:  # > 5s Versatz: kein valides Paar
                continue
            feat = {}
            if mode == "concat":
                cols = []
                for c in step_cols:
                    feat[f"sA_{c}"] = ra[c]
                    feat[f"sB_{c}"] = rb[c]
                    cols += [f"sA_{c}", f"sB_{c}"]
            else:  # mean
                cols = step_cols
                for c in step_cols:
                    feat[c] = (ra[c] + rb[c]) / 2.0
            name = f"{session_tag}_{profile}_c{ra.cycle_id}-{rb.cycle_id}"
            write_sample(out_dir, ra.label, name, feat, cols)
            n += 1
    print(f"[Level 2] {n} Samples nach {out_dir}")


def level3(vectors: pd.DataFrame, out_dir: Path, session_tag: str = ""):
    """Ein Feature-Vektor über alle 8 Sensoren, ausgerichtet an einem
    'Super-Zyklus'. Als Anker dient jeweils der zuletzt abgeschlossene
    Zyklus des langsamsten Sensors; von allen anderen Sensoren wird der
    zu diesem Zeitpunkt letzte vollständig abgeschlossene Zyklus verwendet
    (so wie es auch im späteren Live-Betrieb aussehen würde: man nimmt
    von jedem Sensor den aktuellsten fertigen Messwert)."""
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    sensors = sorted(vectors["sensor_index"].unique())
    # langsamster Sensor = größte mittlere Zyklusdauer
    durations = vectors.groupby("sensor_index").apply(
        lambda g: (g["end_ts"] - g["start_ts"]).mean())
    anchor_sensor = durations.idxmax()
    anchor = vectors[vectors.sensor_index == anchor_sensor].sort_values("start_ts")

    n = 0
    for _, arow in anchor.iterrows():
        feat = {}
        cols = []
        ok = True
        for s in sensors:
            sub = vectors[(vectors.sensor_index == s) & (vectors.end_ts <= arow.end_ts)]
            if sub.empty:
                ok = False
                break
            latest = sub.loc[sub["end_ts"].idxmax()]
            for c in step_cols:
                key = f"s{s}_{c}"
                feat[key] = latest[c]
                cols.append(key)
        if not ok:
            continue
        name = f"{session_tag}_super_c{arow.cycle_id}"
        write_sample(out_dir, arow.label, name, feat, cols)
        n += 1
    print(f"[Level 3] {n} Samples nach {out_dir} (Anker: sensor_index={anchor_sensor})")


def process_file(csv_path: Path, outdir: Path, levels: list, level2_mode: str, log_transform: bool):
    """Verarbeitet eine einzelne Session-CSV und schreibt die gewählten
    Levels in gemeinsame Unterordner unter outdir. session_tag (abgeleitet
    vom Dateinamen) sorgt dafür, dass Samples verschiedener Sessions sich
    im selben Output-Ordner nicht überschreiben."""
    session_tag = sanitize_label(csv_path.stem)
    try:
        df = load_session(csv_path)
    except ValueError as e:
        print(f"[Übersprungen] {csv_path.name}: {e}", file=sys.stderr)
        return 0
    df = detect_cycles(df)
    vectors = build_step_vectors(df, log_transform=log_transform)
    if vectors.empty:
        print(f"[Übersprungen] {csv_path.name}: keine vollständigen Zyklen gefunden.", file=sys.stderr)
        return 0

    total = 0
    if 1 in levels:
        level1(vectors, outdir / "level1_per_sensor", session_tag=session_tag)
    if 2 in levels:
        level2(vectors, outdir / "level2_per_profile", mode=level2_mode, session_tag=session_tag)
    if 3 in levels:
        level3(vectors, outdir / "level3_all_sensors", session_tag=session_tag)
    return len(vectors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_path", type=Path,
                     help="Pfad zu einer einzelnen Session-CSV ODER zu einem Ordner mit mehreren Session-CSVs")
    ap.add_argument("--outdir", type=Path, default=Path("ei_samples"), help="Ausgabeverzeichnis (gemeinsam für alle Sessions)")
    ap.add_argument("--pattern", default="*.csv", help="Datei-Muster bei Ordner-Input (Standard: *.csv)")
    ap.add_argument("--recursive", action="store_true", help="Bei Ordner-Input auch Unterordner durchsuchen")
    ap.add_argument("--level", nargs="+", type=int, default=[1], choices=[1, 2, 3],
                     help="Welche Aggregationsstufen erzeugt werden sollen")
    ap.add_argument("--level2-mode", choices=["concat", "mean"], default="concat")
    ap.add_argument("--no-log", action="store_true", help="log10-Transformation der Gaswiderstände abschalten")
    args = ap.parse_args()

    if args.input_path.is_dir():
        pattern = f"**/{args.pattern}" if args.recursive else args.pattern
        csv_files = sorted(args.input_path.glob(pattern))
        if not csv_files:
            print(f"Keine Dateien passend zu '{args.pattern}' in {args.input_path} gefunden.", file=sys.stderr)
            sys.exit(1)
    else:
        csv_files = [args.input_path]

    print(f"Verarbeite {len(csv_files)} Datei(en) -> {args.outdir}")
    total_vectors = 0
    for csv_path in csv_files:
        print(f"--- {csv_path.name} ---")
        total_vectors += process_file(
            csv_path, args.outdir, args.level, args.level2_mode, log_transform=not args.no_log
        )
    print(f"\nFertig. {total_vectors} vollständige Zyklen über alle Dateien verarbeitet.")


if __name__ == "__main__":
    main()
