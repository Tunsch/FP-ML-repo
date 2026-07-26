"""
app.py -- Streamlit-Oberfläche für die BME688 -> Edge-Impulse Konvertierung.

Start lokal:   streamlit run app.py
Start im Container: siehe Dockerfile (läuft identisch).
"""

import io
import json
import zipfile

import pandas as pd
import streamlit as st

from core import LEVEL_NAMES, build_outputs, load_session, peek_label, process_file

st.set_page_config(page_title="BME688 -> Edge Impulse", layout="wide")
st.title("BME688 → Edge Impulse Konverter")
st.caption("Rohdaten-Sessions hochladen, Aggregationsstufe & Format wählen, fertigen Output herunterladen.")

# ----------------------------------------------------------------------
# Sidebar: alle Optionen
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("1. Daten")
    uploaded_files = st.file_uploader(
        "Session-CSVs (eine oder mehrere)", type="csv", accept_multiple_files=True
    )

    st.header("2. Aggregationsstufe")
    levels = st.multiselect(
        "Level", options=[1, 2, 3],
        default=[1],
        format_func=lambda l: f"Level {l} – {LEVEL_NAMES[l]}",
    )
    level2_mode = "concat"
    if 2 in levels:
        level2_mode = st.radio("Level 2: Sensorpaar kombinieren als", ["concat", "mean"], horizontal=True)

    st.header("3. Vorverarbeitung")
    log_transform = st.checkbox("log10-Transformation der Gaswiderstände", value=True)
    impute_on = st.checkbox("Fehlende Heizstufen imputieren", value=False)
    impute_max_missing = st.slider("max. fehlende Stufen je Zyklus", 0, 5, 2, disabled=not impute_on)
    impute_max_gap = st.slider("Suchfenster (Zyklen)", 1, 10, 3, disabled=not impute_on)
    if not impute_on:
        impute_max_missing = 0

    st.header("4. Output-Zusammenfassung")
    combine = st.radio(
        "Combine-Modus", ["vector", "session", "label", "all"],
        format_func=lambda c: {
            "vector": "vector – 1 Datei je Vektor",
            "session": "session – 1 Datei je Mess-Session",
            "label": "label – 1 Datei je Klasse",
            "all": "all – 1 Gesamtdatei",
        }[c],
        index=0,
    )

    st.header("5. Train/Test-Split")
    apply_split = st.checkbox(
        "Split jetzt schon festlegen", value=False,
        help="Wenn deaktiviert (empfohlen, falls ihr den Split später z.B. in einem "
             "Jupyter Notebook machen wollt): jede Zeile behält die 'session'-Spalte "
             "(bzw. bei combine=vector den Session-Namen im Dateinamen) zur späteren "
             "Zuordnung.",
    )
    test_ratio, split_seed, session_split = 0.2, 42, None
    if apply_split:
        test_ratio = st.slider("Test-Anteil je Label", 0.0, 0.5, 0.2, 0.05)
        split_seed = st.number_input("Zufalls-Seed", value=42, step=1)
        override_file = st.file_uploader(
            "Optional: manuelle Session-Zuordnung (JSON)", type="json", key="split_json"
        )
        if override_file is not None:
            session_split = json.loads(override_file.read())

# ----------------------------------------------------------------------
# Hauptbereich
# ----------------------------------------------------------------------
if not uploaded_files:
    st.session_state.pop("result", None)
    st.info("Lade in der Seitenleiste eine oder mehrere Session-CSVs hoch, um zu starten.")
    st.markdown(
        "**Erwartete Spalten:** `sensor_index, sensor_id, timestamp_since_poweron, "
        "resistance_gassensor, heater_profile_step_index, label_name` "
        "(weitere Spalten sind erlaubt; `heater_profile_id` wird nur für Level 2 gebraucht)."
    )
    st.stop()

# --- 6. Label-Zuordnung: Rohlabels vorab anzeigen und ggf. zusammenführen ---
st.subheader("Erkannte Labels")
st.caption(
    "Falls sich die Namenskonvention eurer Aufnahme-App über die Zeit geändert hat "
    "(z.B. 'Rasur' vs. 'HP Exp 3 Bad Rasur'), tragt hier denselben Ziel-Label-Text ein, "
    "um Varianten zur selben Klasse zusammenzuführen."
)
raw_labels = []
for f in uploaded_files:
    lbl = peek_label(f)
    raw_labels.append(lbl if lbl is not None else "(nicht lesbar)")
label_table = pd.DataFrame({
    "Datei": [f.name for f in uploaded_files],
    "Roh-Label": raw_labels,
    "Ziel-Label": raw_labels,
})
edited = st.data_editor(
    label_table, key="label_editor", hide_index=True, use_container_width=True,
    disabled=["Datei", "Roh-Label"],
)
label_map = dict(zip(edited["Roh-Label"], edited["Ziel-Label"]))
n_merged = sum(1 for r, z in label_map.items() if r != z)
if n_merged:
    st.caption(f"{n_merged} Label-Variante(n) werden zusammengeführt.")

run = st.button("Verarbeiten", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Verarbeitung -- Ergebnis wird in st.session_state abgelegt, damit es
# NICHT verloren geht, wenn Streamlit durch eine Folge-Interaktion (z.B.
# Klick auf den Download-Button) das Skript erneut von oben ausführt.
# ----------------------------------------------------------------------
if run:
    if not levels:
        st.error("Bitte mindestens ein Level auswählen.")
        st.stop()

    try:
        per_file = []
        log_area = st.expander("Verarbeitungs-Log", expanded=False)
        progress = st.progress(0.0)

        for i, f in enumerate(uploaded_files):
            session_tag = f.name.rsplit(".", 1)[0]
            try:
                raw, load_warnings = load_session(f, name_hint=f.name)
            except ValueError as e:
                st.warning(f"{f.name}: {e}")
                continue
            res = process_file(session_tag, raw, levels, level2_mode,
                                log_transform=log_transform,
                                impute_max_missing=impute_max_missing,
                                impute_max_gap=impute_max_gap,
                                label_map=label_map)
            with log_area:
                st.markdown(f"**{f.name}**")
                for w in load_warnings:
                    st.warning(w)
                for line in res["logs"]:
                    st.text(line)
            if not res["ok"]:
                st.warning(f"{f.name}: keine verwertbaren Zyklen gefunden, übersprungen.")
                continue
            per_file.append(res)
            progress.progress((i + 1) / len(uploaded_files))

        if not per_file:
            st.error("Keine verwertbaren Daten in den hochgeladenen Dateien gefunden.")
            st.stop()

        outputs = build_outputs(per_file, levels, combine, apply_split=apply_split,
                                 test_ratio=test_ratio, seed=split_seed, session_split=session_split)

        overview = pd.DataFrame(
            [{"session": r["session_tag"], "label": r["label"]} for r in per_file]
        )
        if apply_split:
            from core import train_test_split_by_session
            overview = train_test_split_by_session(
                overview.assign(n_imputed=0), test_ratio=test_ratio, seed=split_seed,
                session_split=session_split
            )[["session", "label", "category"]]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_path, content in outputs.items():
                zf.writestr(rel_path, content)
        buf.seek(0)

        example_path = next(iter(outputs))
        example_content = outputs[example_path]

        # Ergebnis persistent ablegen -- überlebt jeden nachfolgenden Rerun
        st.session_state["result"] = {
            "n_files": len(outputs),
            "n_sessions": len(per_file),
            "overview": overview,
            "example_path": example_path,
            "example_content": example_content,
            "zip_bytes": buf.getvalue(),
            "apply_split": apply_split,
        }
    except Exception as e:
        st.error("Bei der Verarbeitung ist ein Fehler aufgetreten:")
        st.exception(e)
        st.stop()

# ----------------------------------------------------------------------
# Ergebnis anzeigen -- unabhängig davon, ob dieser Rerun durch den
# "Verarbeiten"-Klick oder durch eine andere Interaktion (z.B. den
# Download-Button selbst) ausgelöst wurde.
# ----------------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]
    st.success(f"{r['n_files']} Datei(en) erzeugt aus {r['n_sessions']} Session(s).")

    st.subheader("Sessions in diesem Lauf")
    st.dataframe(r["overview"], use_container_width=True, hide_index=True)

    st.subheader("Vorschau")
    st.caption(r["example_path"])
    if r["example_content"].count("\n") > 1:
        st.dataframe(pd.read_csv(io.StringIO(r["example_content"])).head(20), use_container_width=True)
    else:
        st.code(r["example_content"])

    st.download_button(
        "Ergebnis als ZIP herunterladen", data=r["zip_bytes"],
        file_name="ei_samples.zip", mime="application/zip",
        use_container_width=True, key="download_zip",
    )

    if not r["apply_split"]:
        st.info(
            "Kein Split angewendet. Jede Zeile trägt die Spalte **session** "
            "(bei combine=vector: im Dateinamen) zur späteren Zuordnung. "
            "Im Notebook z.B.:\n\n"
            "```python\n"
            "from core import train_test_split_by_session\n"
            "df = pd.read_csv('level1_per_sensor/all.csv')\n"
            "df = train_test_split_by_session(df, test_ratio=0.25, seed=7)\n"
            "train = df[df.category == 'training']\n"
            "test  = df[df.category == 'testing']\n"
            "```"
        )
elif not run:
    st.info(f"{len(uploaded_files)} Datei(en) hochgeladen. Optionen in der Seitenleiste einstellen und "
            "auf **Verarbeiten** klicken.")
