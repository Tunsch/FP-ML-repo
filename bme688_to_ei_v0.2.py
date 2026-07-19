#!/usr/bin/env python3
"""
bme688_to_ei.py

Wandelt BME688-Rohdaten-Session-CSVs (8 Sensoren, bis zu 4 Heizprofile,
je Heizprofil mehrere Heizstufen pro Scanning Cycle) in Edge-Impulse-taugliche
Gaswiderstands-Feature-Vektoren um.

Erwartete Eingabe-Spalten (wie im Beispiel-Export):
    sensor_index, sensor_id, timestamp_since_poweron, real_time_clock,
    temperature, pressure, relative_humidity, resistance_gassensor,
    heater_profile_step_index, scanning_enabled, scanning_cycle_index,
    label_tag, error_code, heater_profile_id, label_name

Es werden nur die Gaswiderstandswerte (resistance_gassensor) verwendet und
standardmäßig log10-transformiert (--no-log zum Abschalten).

AGGREGATIONSSTUFEN (--level):
  1 = pro Sensor pro Scanning Cycle
  2 = pro Heizprofil pro Scanning Cycle (2 Sensoren je Profil kombiniert)
  3 = über alle Sensoren (synchronisierter "Super-Zyklus")

ZUSAMMENFASSUNG DES OUTPUTS (--combine):
  vector  = 1 Datei pro Vektor  (Edge-Impulse "single reading" Format,
            <label>.<name>.csv) -- wie bisher
  session = 1 Datei pro Mess-Session (mehrere Zeilen = mehrere Vektoren
            dieser Session)
  label   = 1 Datei pro Klasse/Label (alle Sessions dieser Klasse
            zusammengefasst, "session"-Spalte bleibt erhalten)
  all     = 1 Datei insgesamt (alle Klassen, alle Sessions)

TRAIN/TEST-TRENNUNG:
  Egal welche --combine-Stufe: der Output wird immer schon in
  training/ und testing/ Unterordner (bzw. Dateien) getrennt, und zwar
  IMMER pro ganzer Mess-Session -- eine Session landet nie anteilig in
  beiden Töpfen. Die Zuordnung erfolgt automatisch (seeded, --split-seed)
  im Verhältnis --test-ratio (Default 0.2) je Label, kann aber über
  --session-split eine JSON-Datei {"session_tag": "training"|"testing"}
  manuell vorgegeben/überschrieben werden.

Beispiel:
    python3 bme688_to_ei.py sessions/ --outdir out --level 1 2 3 --combine session
"""

import argparse
import json
import random
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
    """Ergänzt pro sensor_index eine fortlaufende cycle_id über den
    Wrap-Around des heater_profile_step_index (robuster als die im Export
    oft nicht verlässlich befüllte scanning_cycle_index-Spalte)."""
    df = df.sort_values(["sensor_index", "timestamp_since_poweron"]).reset_index(drop=True)
    out = []
    for sensor_index, g in df.groupby("sensor_index", sort=False):
        g = g.sort_values("timestamp_since_poweron").reset_index(drop=True)
        wrap = g["heater_profile_step_index"].diff() <= 0
        wrap.iloc[0] = True
        g["cycle_id"] = wrap.cumsum() - 1
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_step_vectors(df: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """Baut pro (sensor_index, cycle_id) einen vollständigen Gaswiderstands-
    Vektor über alle Heizstufen. Unvollständige Zyklen werden verworfen."""
    rows = []
    for (sensor_index, cycle_id), g in df.groupby(["sensor_index", "cycle_id"]):
        steps = sorted(g["heater_profile_step_index"].unique())
        expected = list(range(g["heater_profile_step_index"].max() + 1))
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


# ---------------------------------------------------------------------
# Level-Builder: bauen jeweils ein DataFrame mit den Spalten
#   label, <metadata-Spalten...>, <feature-Spalten...>
# und geben zusaetzlich die Liste der Feature-Spalten zurueck.
# ---------------------------------------------------------------------

def make_level1_df(vectors: pd.DataFrame):
    step_cols = [c for c in vectors.columns if c.startswith("gasres_step")]
    df = vectors[["label", "sensor_index", "heater_profile_id", "cycle_id"] + step_cols].copy()
    return df, step_cols


def make_level2_df(vectors: pd.DataFrame, mode: str = "concat"):
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    rows = []
    for profile, g in vectors.groupby("heater_profile_id"):
        sensors = sorted(g["sensor_index"].unique())
        if len(sensors) != 2:
            print(f"[Level 2] Warnung: Profil {profile} hat {len(sensors)} statt 2 Sensoren, übersprungen")
            continue
        a = g[g.sensor_index == sensors[0]].sort_values("start_ts").reset_index(drop=True)
        b = g[g.sensor_index == sensors[1]].sort_values("start_ts").reset_index(drop=True)
        for _, ra in a.iterrows():
            idx = (b["start_ts"] - ra.start_ts).abs().idxmin()
            rb = b.loc[idx]
            if abs(rb.start_ts - ra.start_ts) > 5000:
                continue
            row = {"label": ra.label, "heater_profile_id": profile,
                   "cycle_id_a": ra.cycle_id, "cycle_id_b": rb.cycle_id}
            if mode == "concat":
                for c in step_cols:
                    row[f"sA_{c}"] = ra[c]
                    row[f"sB_{c}"] = rb[c]
            else:
                for c in step_cols:
                    row[c] = (ra[c] + rb[c]) / 2.0
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in
                    ("label", "heater_profile_id", "cycle_id_a", "cycle_id_b")]
    return df, feature_cols


def make_level3_df(vectors: pd.DataFrame):
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    sensors = sorted(vectors["sensor_index"].unique())
    durations = vectors.groupby("sensor_index").apply(lambda g: (g["end_ts"] - g["start_ts"]).mean())
    anchor_sensor = durations.idxmax()
    anchor = vectors[vectors.sensor_index == anchor_sensor].sort_values("start_ts")

    rows = []
    for _, arow in anchor.iterrows():
        row = {"label": arow.label, "cycle_id": arow.cycle_id}
        ok = True
        for s in sensors:
            sub = vectors[(vectors.sensor_index == s) & (vectors.end_ts <= arow.end_ts)]
            if sub.empty:
                ok = False
                break
            latest = sub.loc[sub["end_ts"].idxmax()]
            for c in step_cols:
                row[f"s{s}_{c}"] = latest[c]
        if ok:
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("label", "cycle_id")]
    return df, feature_cols, anchor_sensor


LEVEL_DIRS = {1: "level1_per_sensor", 2: "level2_per_profile", 3: "level3_all_sensors"}


# ---------------------------------------------------------------------
# Session -> Training/Testing Zuordnung
# ---------------------------------------------------------------------

def assign_session_categories(session_labels: dict, test_ratio: float, seed: int,
                               override_path: Path | None) -> dict:
    """session_labels: {session_tag: label}. Liefert {session_tag: 'training'|'testing'}.
    Trennung erfolgt IMMER auf Ebene ganzer Sessions, nie innerhalb einer Session,
    damit Training/Test garantiert unabhängige Messungen enthalten."""
    override = {}
    if override_path is not None:
        override = json.loads(Path(override_path).read_text())

    by_label = {}
    for session, label in session_labels.items():
        by_label.setdefault(label, []).append(session)

    rng = random.Random(seed)
    categories = {}
    for label, sessions in by_label.items():
        sessions = sorted(sessions)
        rng.shuffle(sessions)
        n_test = max(1, round(len(sessions) * test_ratio)) if len(sessions) > 1 else 0
        test_sessions = set(sessions[:n_test])
        for s in sessions:
            categories[s] = "testing" if s in test_sessions else "training"
        if len(sessions) == 1:
            print(f"[Split] Warnung: Label '{label}' hat nur 1 Session ({sessions[0]}) "
                  f"-> komplett 'training', kein leakage-freier Test möglich für dieses Label.")

    for s, cat in override.items():
        if cat not in ("training", "testing"):
            print(f"[Split] Warnung: ungültiger Wert '{cat}' für Session '{s}' in --session-split, ignoriert.")
            continue
        categories[s] = cat

    return categories


# ---------------------------------------------------------------------
# Output schreiben je nach --combine
# ---------------------------------------------------------------------

def write_vector_files(df: pd.DataFrame, feature_cols: list, out_dir: Path, session_tag: str):
    n = 0
    for _, row in df.iterrows():
        cat = row["category"]
        label = sanitize_label(row["label"])
        extra = "_".join(str(row[c]) for c in df.columns
                          if c not in ("label", "category", "session") and c not in feature_cols)
        name = f"{session_tag}_{extra}" if extra else session_tag
        d = out_dir / cat
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{label}.{name}_{n}.csv"
        header = ",".join(feature_cols)
        values = ",".join(f"{row[c]:.6f}" for c in feature_cols)
        path.write_text(header + "\n" + values + "\n")
        n += 1
    return n


def write_table(df: pd.DataFrame, feature_cols: list, path: Path, extra_cols: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["label", "session"] + extra_cols + feature_cols
    df.to_csv(path, columns=cols, index=False, float_format="%.6f")


def emit(level: int, df: pd.DataFrame, feature_cols: list, session_tag: str, label: str,
          categories: dict, outdir: Path, combine: str, accum: dict):
    if df.empty:
        return 0
    df = df.copy()
    df["label"] = label
    df["session"] = session_tag
    df["category"] = categories.get(session_tag, "training")
    level_dir = outdir / LEVEL_DIRS[level]
    extra_cols = [c for c in df.columns if c not in ("label", "session", "category") and c not in feature_cols]

    if combine == "vector":
        n = write_vector_files(df, feature_cols, level_dir, session_tag)
    elif combine == "session":
        cat = df["category"].iloc[0]
        path = level_dir / cat / f"{sanitize_label(label)}__{session_tag}.csv"
        write_table(df, feature_cols, path, extra_cols)
        n = len(df)
    else:  # 'label' oder 'all' -> im Speicher sammeln, am Ende gemeinsam schreiben
        key = (level, label if combine == "label" else "__all__")
        accum.setdefault(key, {"rows": [], "feature_cols": feature_cols, "extra_cols": extra_cols})
        accum[key]["rows"].append(df)
        n = len(df)
    return n


def flush_accum(accum: dict, outdir: Path, combine: str):
    for (level, key), payload in accum.items():
        full = pd.concat(payload["rows"], ignore_index=True)
        level_dir = outdir / LEVEL_DIRS[level]
        name = "all" if key == "__all__" else sanitize_label(key)
        for cat, g in full.groupby("category"):
            path = level_dir / cat / f"{name}.csv"
            write_table(g, payload["feature_cols"], path, payload["extra_cols"])
            print(f"  -> {path} ({len(g)} Zeilen)")


def process_file(csv_path: Path, levels: list, level2_mode: str, log_transform: bool):
    session_tag = sanitize_label(csv_path.stem)
    try:
        df = load_session(csv_path)
    except ValueError as e:
        print(f"[Übersprungen] {csv_path.name}: {e}", file=sys.stderr)
        return None
    df = detect_cycles(df)
    vectors = build_step_vectors(df, log_transform=log_transform)
    if vectors.empty:
        print(f"[Übersprungen] {csv_path.name}: keine vollständigen Zyklen gefunden.", file=sys.stderr)
        return None

    label = vectors["label"].iloc[0]
    built = {}
    if 1 in levels:
        built[1] = make_level1_df(vectors)
    if 2 in levels:
        built[2] = make_level2_df(vectors, mode=level2_mode)
    if 3 in levels:
        d3, f3, anchor = make_level3_df(vectors)
        built[3] = (d3, f3)
        print(f"  [Level 3] Anker-Sensor (langsamstes Profil): {anchor}")
    return session_tag, label, built


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_path", type=Path,
                     help="Pfad zu einer einzelnen Session-CSV ODER zu einem Ordner mit mehreren Session-CSVs")
    ap.add_argument("--outdir", type=Path, default=Path("ei_samples"))
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--level", nargs="+", type=int, default=[1], choices=[1, 2, 3])
    ap.add_argument("--level2-mode", choices=["concat", "mean"], default="concat")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--combine", choices=["vector", "session", "label", "all"], default="vector",
                     help="Wie der Output zusammengefasst wird (Default: vector = 1 Datei je Vektor, wie bisher)")
    ap.add_argument("--test-ratio", type=float, default=0.2,
                     help="Anteil der Sessions je Label, die in die Testmenge wandern (Default 0.2)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--session-split", type=Path, default=None,
                     help="Optionale JSON-Datei {\"session_tag\": \"training\"|\"testing\"} zur manuellen Vorgabe/Override")
    args = ap.parse_args()

    if args.input_path.is_dir():
        pattern = f"**/{args.pattern}" if args.recursive else args.pattern
        csv_files = sorted(args.input_path.glob(pattern))
        if not csv_files:
            print(f"Keine Dateien passend zu '{args.pattern}' in {args.input_path} gefunden.", file=sys.stderr)
            sys.exit(1)
    else:
        csv_files = [args.input_path]

    print(f"Verarbeite {len(csv_files)} Datei(en) -> {args.outdir} (combine={args.combine})")

    # Erster Durchlauf: alles einlesen und Level-DataFrames bauen
    per_file = {}
    session_labels = {}
    for csv_path in csv_files:
        print(f"--- {csv_path.name} ---")
        result = process_file(csv_path, args.level, args.level2_mode, log_transform=not args.no_log)
        if result is None:
            continue
        session_tag, label, built = result
        per_file[session_tag] = built
        session_labels[session_tag] = label

    if not per_file:
        print("Keine verwertbaren Daten gefunden.", file=sys.stderr)
        sys.exit(1)

    categories = assign_session_categories(session_labels, args.test_ratio, args.split_seed, args.session_split)
    print("\nSession -> Kategorie:")
    for s, c in sorted(categories.items()):
        print(f"  {s} ({session_labels[s]}): {c}")

    accum = {}
    total = {1: 0, 2: 0, 3: 0}
    for session_tag, built in per_file.items():
        label = session_labels[session_tag]
        for level, (df, feature_cols) in built.items():
            n = emit(level, df, feature_cols, session_tag, label, categories,
                      args.outdir, args.combine, accum)
            total[level] += n

    if args.combine in ("label", "all"):
        print("\nSchreibe zusammengefasste Dateien:")
        flush_accum(accum, args.outdir, args.combine)

    print("\nFertig. Vektoren je Level:")
    for level in args.level:
        print(f"  Level {level}: {total[level]}")


if __name__ == "__main__":
    main()
