"""
core.py

Kernlogik zur Umwandlung von BME688-Rohdaten-Sessions in Edge-Impulse-taugliche
Gaswiderstands-Feature-Vektoren. Enthält keine CLI-/UI-Logik, wird sowohl von
cli.py (Kommandozeile) als auch von app.py (Streamlit) importiert.

WICHTIGES PRINZIP: Der Train/Test-Split ist hier bewusst OPTIONAL und von der
eigentlichen Vektor-Erzeugung entkoppelt. Jede erzeugte Zeile/jeder Vektor
trägt in jedem Ausgabeformat immer die Spalte 'session' (bei --combine vector
steckt sie im Dateinamen). Das ist der Schlüssel, um den Split später frei
einzustellen -- z.B. im Jupyter Notebook mit train_test_split_by_session()
weiter unten, die genau diese Spalte nutzt.
"""

from __future__ import annotations

import io
import json
import random
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "sensor_index", "sensor_id", "timestamp_since_poweron",
    "resistance_gassensor", "heater_profile_step_index", "label_name",
]
# heater_profile_id ist NICHT hart erforderlich: manche älteren Rohexporte
# enthalten diese Spalte nicht. Sie wird nur für Level 2 (Sensorpaar je
# Heizprofil) gebraucht -- fehlt sie, wird ein Platzhalter je Sensor gesetzt
# und Level 1/3 funktionieren unverändert, Level 2 liefert für diese Datei
# dann keine Paare (Warnung statt Absturz).

LEVEL_DIRS = {1: "level1_per_sensor", 2: "level2_per_profile", 3: "level3_all_sensors"}
LEVEL_NAMES = {1: "pro Sensor", 2: "pro Heizprofil", 3: "über alle Sensoren"}


def sanitize_label(label: str) -> str:
    label = str(label).strip()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_\-]", "", label)
    return label


# ---------------------------------------------------------------------
# Einlesen & Zyklen erkennen
# ---------------------------------------------------------------------

def peek_label(file_like_or_path) -> Optional[str]:
    """Liest nur die label_name-Spalte der ersten Zeile -- schnelle Vorschau,
    ohne die ganze Datei einzulesen. Gibt None zurück, wenn die Spalte fehlt
    oder die Datei nicht lesbar ist."""
    try:
        df = pd.read_csv(file_like_or_path, usecols=["label_name"], nrows=1)
        if hasattr(file_like_or_path, "seek"):
            file_like_or_path.seek(0)
        return str(df["label_name"].iloc[0]).strip()
    except Exception:
        if hasattr(file_like_or_path, "seek"):
            file_like_or_path.seek(0)
        return None


def load_session(file_like_or_path, name_hint: Optional[str] = None) -> tuple[pd.DataFrame, list]:
    """Liest eine Session-CSV. file_like_or_path kann ein Pfad ODER ein
    file-like Objekt sein (z.B. ein von Streamlit hochgeladenes File).
    Gibt (DataFrame, Liste von Warnungen) zurück."""
    df = pd.read_csv(file_like_or_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten{f' in {name_hint}' if name_hint else ''}: {missing}")

    warnings = []
    if "heater_profile_id" not in df.columns:
        df = df.copy()
        df["heater_profile_id"] = "unknown_s" + df["sensor_index"].astype(str)
        warnings.append(
            f"{name_hint or 'Datei'}: Spalte 'heater_profile_id' fehlt im Rohexport -- "
            "durch einen Platzhalter je Sensor ersetzt. Level 1 und Level 3 sind davon "
            "nicht betroffen, Level 2 (Sensorpaar je Heizprofil) liefert für diese Datei "
            "keine Paare, da nicht bekannt ist, welche 2 Sensoren dasselbe Profil nutzen."
        )
    return df, warnings


def detect_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt pro sensor_index eine fortlaufende cycle_id über den
    Wrap-Around des heater_profile_step_index."""
    df = df.sort_values(["sensor_index", "timestamp_since_poweron"]).reset_index(drop=True)
    out = []
    for sensor_index, g in df.groupby("sensor_index", sort=False):
        g = g.sort_values("timestamp_since_poweron").reset_index(drop=True)
        wrap = g["heater_profile_step_index"].diff() <= 0
        wrap.iloc[0] = True
        g["cycle_id"] = wrap.cumsum() - 1
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _canonical_step_count(df: pd.DataFrame) -> dict:
    result = {}
    for sensor_index, g in df.groupby("sensor_index"):
        n_per_cycle = g.groupby("cycle_id")["heater_profile_step_index"].apply(lambda s: s.nunique())
        result[sensor_index] = int(n_per_cycle.mode().iloc[0])
    return result


def report_incomplete_cycles(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Diagnose-Report zur Zyklen-Vollständigkeit je Sensor.
    Gibt (Tabelle, formatierter Text) zurück, damit sowohl CLI (print) als
    auch Streamlit (st.text/st.dataframe) es direkt nutzen können."""
    step_counts = _canonical_step_count(df)
    rows = []
    for (sensor_index, cycle_id), g in df.groupby(["sensor_index", "cycle_id"]):
        n_expected = step_counts[sensor_index]
        present = set(g["heater_profile_step_index"].unique())
        rows.append({"sensor_index": sensor_index, "cycle_id": cycle_id,
                      "n_expected": n_expected, "n_missing": n_expected - len(present),
                      "n_duplicates": len(g) - len(present)})
    rep = pd.DataFrame(rows)

    lines = ["Zyklen-Vollständigkeit je Sensor:"]
    for sensor_index, g in rep.groupby("sensor_index"):
        total = len(g)
        complete = (g["n_missing"] == 0).sum()
        dist = g.loc[g["n_missing"] > 0, "n_missing"].value_counts().sort_index()
        dist_str = ", ".join(f"{k} fehlend: {v}x" for k, v in dist.items()) or "-"
        dup = (g["n_duplicates"] > 0).sum()
        lines.append(f"  Sensor {sensor_index}: {complete}/{total} vollständig "
                      f"({dist_str}), Zyklen mit doppelten Messwerten: {dup}")
    return rep, "\n".join(lines)


# ---------------------------------------------------------------------
# Vektor-Erzeugung inkl. optionaler zeitlicher Imputation
# ---------------------------------------------------------------------

def build_step_vectors(df: pd.DataFrame, log_transform: bool = True,
                        impute_max_missing: int = 0, impute_max_gap: int = 3) -> tuple[pd.DataFrame, str]:
    """Siehe Docstring der Vorgängerversion: baut pro (sensor_index, cycle_id)
    einen Gaswiderstands-Vektor. Fehlende Stufen werden -- falls aktiviert --
    zeitlich aus dem gleichen Schritt benachbarter Zyklen desselben Sensors
    interpoliert, nie aus benachbarten Stufen desselben Zyklus."""
    step_counts = _canonical_step_count(df)

    raw = (df.groupby(["sensor_index", "cycle_id", "heater_profile_step_index"])["resistance_gassensor"]
             .mean())
    if log_transform:
        raw = np.log10(raw.clip(lower=1e-3))
    meta = (df.groupby(["sensor_index", "cycle_id"])
              .agg(sensor_id=("sensor_id", "first"),
                   heater_profile_id=("heater_profile_id", "first"),
                   start_ts=("timestamp_since_poweron", "min"),
                   end_ts=("timestamp_since_poweron", "max"),
                   label=("label_name", "first")))

    rows = []
    n_rescued = 0
    for (sensor_index, cycle_id), meta_row in meta.iterrows():
        n_expected = step_counts[sensor_index]
        values, n_imputed = {}, 0
        for step in range(n_expected):
            key = (sensor_index, cycle_id, step)
            if key in raw.index:
                values[step] = raw.loc[key]
                continue
            if impute_max_missing <= 0:
                values[step] = None
                continue
            prev_val = prev_dist = None
            for d in range(1, impute_max_gap + 1):
                k = (sensor_index, cycle_id - d, step)
                if k in raw.index:
                    prev_val, prev_dist = raw.loc[k], d
                    break
            next_val = next_dist = None
            for d in range(1, impute_max_gap + 1):
                k = (sensor_index, cycle_id + d, step)
                if k in raw.index:
                    next_val, next_dist = raw.loc[k], d
                    break
            if prev_val is not None and next_val is not None:
                w = prev_dist / (prev_dist + next_dist)
                values[step] = prev_val + (next_val - prev_val) * w
                n_imputed += 1
            elif prev_val is not None:
                values[step] = prev_val
                n_imputed += 1
            elif next_val is not None:
                values[step] = next_val
                n_imputed += 1
            else:
                values[step] = None

        if any(v is None for v in values.values()) or n_imputed > impute_max_missing:
            continue
        if n_imputed > 0:
            n_rescued += 1

        row = {
            "sensor_index": sensor_index, "sensor_id": meta_row.sensor_id,
            "heater_profile_id": meta_row.heater_profile_id, "cycle_id": cycle_id,
            "start_ts": meta_row.start_ts, "end_ts": meta_row.end_ts,
            "label": meta_row.label, "n_imputed": n_imputed,
        }
        for step, v in values.items():
            row[f"gasres_step{step}"] = v
        rows.append(row)

    msg = (f"{n_rescued} zusätzliche Zyklen durch Imputation gerettet "
           f"(max. {impute_max_missing} fehlende Stufe(n), Suchfenster {impute_max_gap} Zyklen)."
           if impute_max_missing > 0 else "Imputation deaktiviert.")
    return pd.DataFrame(rows), msg


# ---------------------------------------------------------------------
# Level-Builder
# ---------------------------------------------------------------------

def make_level1_df(vectors: pd.DataFrame):
    step_cols = [c for c in vectors.columns if c.startswith("gasres_step")]
    df = vectors[["label", "sensor_index", "heater_profile_id", "cycle_id", "n_imputed"] + step_cols].copy()
    return df, step_cols


def make_level2_df(vectors: pd.DataFrame, mode: str = "concat"):
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    rows, warnings = [], []
    for profile, g in vectors.groupby("heater_profile_id"):
        sensors = sorted(g["sensor_index"].unique())
        if len(sensors) != 2:
            warnings.append(f"Profil {profile} hat {len(sensors)} statt 2 Sensoren, übersprungen")
            continue
        a = g[g.sensor_index == sensors[0]].sort_values("start_ts").reset_index(drop=True)
        b = g[g.sensor_index == sensors[1]].sort_values("start_ts").reset_index(drop=True)
        for _, ra in a.iterrows():
            idx = (b["start_ts"] - ra.start_ts).abs().idxmin()
            rb = b.loc[idx]
            if abs(rb.start_ts - ra.start_ts) > 5000:
                continue
            row = {"label": ra.label, "heater_profile_id": profile,
                   "cycle_id_a": ra.cycle_id, "cycle_id_b": rb.cycle_id,
                   "n_imputed": int(ra.n_imputed) + int(rb.n_imputed)}
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
                    ("label", "heater_profile_id", "cycle_id_a", "cycle_id_b", "n_imputed")]
    return df, feature_cols, warnings


def make_level3_df(vectors: pd.DataFrame):
    step_cols = sorted([c for c in vectors.columns if c.startswith("gasres_step")],
                        key=lambda c: int(c.replace("gasres_step", "")))
    sensors = sorted(vectors["sensor_index"].unique())
    durations = vectors.groupby("sensor_index").apply(lambda g: (g["end_ts"] - g["start_ts"]).mean())
    anchor_sensor = durations.idxmax()
    anchor = vectors[vectors.sensor_index == anchor_sensor].sort_values("start_ts")

    rows = []
    for _, arow in anchor.iterrows():
        row = {"label": arow.label, "cycle_id": arow.cycle_id, "n_imputed": 0}
        ok = True
        for s in sensors:
            sub = vectors[(vectors.sensor_index == s) & (vectors.end_ts <= arow.end_ts)]
            if sub.empty:
                ok = False
                break
            latest = sub.loc[sub["end_ts"].idxmax()]
            row["n_imputed"] += int(latest["n_imputed"])
            for c in step_cols:
                row[f"s{s}_{c}"] = latest[c]
        if ok:
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("label", "cycle_id", "n_imputed")]
    return df, feature_cols, anchor_sensor


# ---------------------------------------------------------------------
# Eine Session komplett verarbeiten
# ---------------------------------------------------------------------

def process_file(session_tag: str, raw_df: pd.DataFrame, levels: list, level2_mode: str,
                  log_transform: bool, impute_max_missing: int, impute_max_gap: int,
                  label_map: Optional[dict] = None):
    """Verarbeitet eine Session (bereits als DataFrame eingelesen) und baut
    die gewünschten Level-DataFrames. Gibt ein dict mit allen Zwischen-
    ergebnissen inkl. Logtexten zurück (für CLI-print bzw. Streamlit-Anzeige).

    label_map: optionales dict {roher Label-String: kanonischer Label-String}.
    Wird genutzt, um Schreibvarianten desselben Labels zusammenzuführen
    (z.B. wenn sich die Namenskonvention der Aufnahme-App über die Zeit
    geändert hat -- "Rasur" / "HP Exp 3 Rasur" / "HP Exp 3 Bad Rasur" sollen
    z.B. dieselbe Klasse sein). Führende/nachgestellte Leerzeichen im
    Rohlabel werden dabei immer automatisch entfernt, unabhängig von
    label_map."""
    logs = []
    df = detect_cycles(raw_df)
    _, report_text = report_incomplete_cycles(df)
    logs.append(report_text)

    vectors, impute_msg = build_step_vectors(df, log_transform=log_transform,
                                              impute_max_missing=impute_max_missing,
                                              impute_max_gap=impute_max_gap)
    logs.append(impute_msg)

    if vectors.empty:
        return {"session_tag": session_tag, "label": None, "built": {}, "logs": logs, "ok": False}

    raw_label = str(vectors["label"].iloc[0]).strip()
    label = raw_label
    if label_map and raw_label in label_map:
        label = label_map[raw_label]
    if label != raw_label:
        logs.append(f"[Label] '{raw_label}' -> '{label}' (per Zuordnungstabelle zusammengeführt)")

    built = {}
    if 1 in levels:
        built[1] = make_level1_df(vectors)[:2]
    if 2 in levels:
        d2, f2, warn2 = make_level2_df(vectors, mode=level2_mode)
        built[2] = (d2, f2)
        logs += [f"[Level 2] {w}" for w in warn2]
    if 3 in levels:
        d3, f3, anchor = make_level3_df(vectors)
        built[3] = (d3, f3)
        logs.append(f"[Level 3] Anker-Sensor (langsamstes Profil): {anchor}")

    # eindeutige, stabile ID je Vektor -- nützlich zur Rückverfolgung, egal
    # welcher combine-Modus später genutzt wird
    for level, (d, cols) in built.items():
        d = d.copy()
        d.insert(0, "vector_id", [f"{session_tag}__L{level}__{i}" for i in range(len(d))])
        built[level] = (d, cols)

    return {"session_tag": session_tag, "label": label, "built": built, "logs": logs, "ok": True}


# ---------------------------------------------------------------------
# Split-Zuordnung -- eigenständig nutzbar (Skript-Pipeline UND Jupyter!)
# ---------------------------------------------------------------------

def train_test_split_by_session(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42,
                                 session_col: str = "session", label_col: str = "label",
                                 session_split: Optional[dict] = None) -> pd.DataFrame:
    """Nimmt eine Tabelle mit (mindestens) den Spalten session_col und
    label_col und ergänzt eine 'category'-Spalte ('training'/'testing').
    Die Aufteilung erfolgt IMMER auf Ebene ganzer Sessions (nie werden
    Zeilen derselben Session auf beide Seiten verteilt), getrennt je Label
    berechnet, damit das Verhältnis über die Klassen stabil bleibt.

    session_split: optionales dict {session_name: "training"|"testing"} zur
    manuellen Vorgabe/Override einzelner Sessions.

    Das ist exakt dieselbe Funktion, die auch app.py/cli.py optional nutzen
    -- ihr könnt sie 1:1 in einem Jupyter Notebook importieren:

        from core import train_test_split_by_session
        df = pd.read_csv("level1_per_sensor.csv")
        df = train_test_split_by_session(df, test_ratio=0.25, seed=7)
        train = df[df.category == "training"]
        test  = df[df.category == "testing"]
    """
    df = df.copy()
    by_label = {}
    for session, label in df[[session_col, label_col]].drop_duplicates().itertuples(index=False):
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

    if session_split:
        for s, cat in session_split.items():
            if cat in ("training", "testing"):
                categories[s] = cat

    df["category"] = df[session_col].map(categories).fillna("training")

    # Sicherheitsnetz: imputierte Vektoren gehören nicht in die Testmenge
    if "n_imputed" in df.columns:
        drop_mask = (df["category"] == "testing") & (df["n_imputed"] > 0)
        df = df[~drop_mask].copy()

    return df


# ---------------------------------------------------------------------
# Output-Zusammenstellung: liefert {relativer_pfad: csv_text}
# ---------------------------------------------------------------------

def build_outputs(per_file: list, levels: list, combine: str,
                   apply_split: bool, test_ratio: float, seed: int,
                   session_split: Optional[dict] = None) -> dict:
    """per_file: Liste von process_file()-Ergebnissen.
    Gibt {relativer_dateipfad (str): csv_inhalt (str)} zurück -- unabhängig
    davon, ob das Ergebnis danach auf Platte geschrieben oder in ein ZIP für
    den Streamlit-Download gepackt wird."""
    outputs = {}

    for level in levels:
        rows_all = []
        for res in per_file:
            if not res["ok"] or level not in res["built"]:
                continue
            d, feature_cols = res["built"][level]
            d = d.copy()
            d["label"] = res["label"]
            d["session"] = res["session_tag"]
            rows_all.append((d, feature_cols))

        if not rows_all:
            continue
        full = pd.concat([d for d, _ in rows_all], ignore_index=True)
        feature_cols = rows_all[0][1]

        if apply_split:
            full = train_test_split_by_session(full, test_ratio=test_ratio, seed=seed,
                                                session_split=session_split)
        else:
            full["category"] = "unsplit"

        meta_cols = [c for c in full.columns if c not in feature_cols
                     and c not in ("vector_id", "label", "session", "category")]

        level_dir = LEVEL_DIRS[level]
        cat_subdir = (lambda cat: f"{cat}/") if apply_split else (lambda cat: "")

        if combine == "vector":
            for _, row in full.iterrows():
                label = sanitize_label(row["label"])
                path = f"{level_dir}/{cat_subdir(row['category'])}{label}.{row['vector_id']}.csv"
                header = ",".join(feature_cols)
                values = ",".join(f"{row[c]:.6f}" for c in feature_cols)
                outputs[path] = header + "\n" + values + "\n"
        elif combine == "session":
            for (session, cat), g in full.groupby(["session", "category"]):
                label = sanitize_label(g["label"].iloc[0])
                path = f"{level_dir}/{cat_subdir(cat)}{label}__{session}.csv"
                cols = ["vector_id", "label", "session", "category"] + meta_cols + feature_cols
                outputs[path] = g[cols].to_csv(index=False, float_format="%.6f")
        elif combine == "label":
            for (label, cat), g in full.groupby(["label", "category"]):
                path = f"{level_dir}/{cat_subdir(cat)}{sanitize_label(label)}.csv"
                cols = ["vector_id", "label", "session", "category"] + meta_cols + feature_cols
                outputs[path] = g[cols].to_csv(index=False, float_format="%.6f")
        elif combine == "hierarchy":
            # Für Level 1: Aufteilung nach einzelnen Sensoren
            if level == 1 and "sensor_index" in full.columns:
                for (lbl, sess, s_idx, cat), g in full.groupby(["label", "session", "sensor_index", "category"]):
                    clean_lbl = sanitize_label(lbl)
                    clean_sess = sanitize_label(sess)
                    # Erzeugt: Messobjekt / Messsession / sensor_0.csv
                    path = f"{level_dir}/{cat_subdir(cat)}{clean_lbl}/{clean_sess}/sensor_{s_idx}.csv"
                    cols = ["vector_id", "label", "session", "category"] + meta_cols + feature_cols
                    outputs[path] = g[cols].to_csv(index=False, float_format="%.6f")

            # Für Level 2: Aufteilung nach Heizprofil (Sensoren-Paare)
            elif "heater_profile_id" in full.columns:
                for (lbl, sess, hp, cat), g in full.groupby(["label", "session", "heater_profile_id", "category"]):
                    clean_lbl = sanitize_label(lbl)
                    clean_sess = sanitize_label(sess)
                    clean_hp = sanitize_label(hp)
                    # Erzeugt: Messobjekt / Messsession / heater_322.csv
                    path = f"{level_dir}/{cat_subdir(cat)}{clean_lbl}/{clean_sess}/{clean_hp}.csv"
                    cols = ["vector_id", "label", "session", "category"] + meta_cols + feature_cols
                    outputs[path] = g[cols].to_csv(index=False, float_format="%.6f")

            # Fallback für Level 3 (alle Sensoren zusammen)
            else:
                for (lbl, sess, cat), g in full.groupby(["label", "session", "category"]):
                    clean_lbl = sanitize_label(lbl)
                    clean_sess = sanitize_label(sess)
                    # Erzeugt: Messobjekt / Messsession / all_profiles.csv
                    path = f"{level_dir}/{cat_subdir(cat)}{clean_lbl}/{clean_sess}/all_profiles.csv"
                    cols = ["vector_id", "label", "session", "category"] + meta_cols + feature_cols
                    outputs[path] = g[cols].to_csv(index=False, float_format="%.6f")

    return outputs


def truncate_session_by_time(df: pd.DataFrame, max_minutes: float) -> pd.DataFrame:
    """
    Kürzt eine Session ab, ohne Zyklen zu zerschneiden.
    Ein Zyklus bleibt komplett erhalten, wenn sein Startzeitpunkt
    vor dem Limit liegt.
    """
    df_cycles = detect_cycles(df)
    max_ms = max_minutes * 60 * 1000

    valid_cycles = []
    for (sensor, cycle), group in df_cycles.groupby(["sensor_index", "cycle_id"]):
        if group["timestamp_since_poweron"].min() <= max_ms:
            valid_cycles.append(group)

    if not valid_cycles:
        return pd.DataFrame()

    # cycle_id (von detect_cycles) wieder entfernen für sauberen Export
    return pd.concat(valid_cycles, ignore_index=True).drop(columns=["cycle_id"], errors="ignore")
