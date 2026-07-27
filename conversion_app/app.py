import streamlit as st
import json
import pandas as pd
import re
import io
import os
from datetime import datetime
import zipfile

# Pfad für die dauerhafte Speicherung im Docker-Container
HISTORY_FILE = "/app/data/labels_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_history(seed, date, labels, filename):
    history = load_history()
    konvertiert_am = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history[seed] = {
        "Datum/Uhrzeit (Messung)": date,
        "Konvertiert am": konvertiert_am,
        "Zugeordnete Labels": labels,
        "Ursprungsdatei": filename
    }
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


st.set_page_config(page_title="BME688 ML Toolbox", page_icon="💨", layout="centered")

tab1, tab2, tab3 = st.tabs(["Datei-Konverter (Smart)", "Externe CSVs verketten", "Label Übersicht"])

# ==========================================
# TAB 1: SMART-KONVERTER WITH SENSOR_CONFIG MAPPING
# ==========================================
with tab1:
    st.title("BME688 Daten-Konverter")
    st.write("Ziehe eine oder mehrere Dateien (`.bmerawdata` und optional `.bmelabelinfo`) gleichzeitig hierher.")

    # Filter-Einstellungen
    st.subheader("⚙️ Filter-Einstellungen für den Mehrfachimport")
    mindestdauer_sek = st.slider(
        "Mindestdauer der Messung berücksichtigen (in Sekunden):",
        min_value=0, max_value=600, value=90, step=5
    )

    uploaded_files = st.file_uploader(
        "Dateien hier ablegen (Mehrfachauswahl oder Ordner-Inhalt herziehen)",
        type=["bmerawdata", "bmelabelinfo"],
        accept_multiple_files=True
    )

    if uploaded_files:
        raw_dict = {}
        label_dict = {}
        fehlerhafte_dateien = []

        # Sortiere Dateien nach Typ und extrahiere den Seed
        for f in uploaded_files:
            filename = f.name
            match = re.search(r'_([a-zA-Z0-9]{16})_', filename)
            if match:
                seed = match.group(1)
                if filename.endswith(".bmerawdata"):
                    raw_dict[seed] = f
                elif filename.endswith(".bmelabelinfo"):
                    label_dict[seed] = f
            else:
                fehlerhafte_dateien.append(f"{filename} (Kein Seed im Namen)")

        # Zuordnungen ermitteln
        matching_seeds = set(raw_dict.keys()).intersection(set(label_dict.keys()))
        alleinstehende_raw_seeds = set(raw_dict.keys()) - set(label_dict.keys())

        verwaiste_label_seeds = set(label_dict.keys()) - set(raw_dict.keys())
        for vs in verwaiste_label_seeds:
            fehlerhafte_dateien.append(f"{label_dict[vs].name} (Zugehörige .bmerawdata fehlt)")

        erfolgreich_liste = []
        ignoriert_liste = []

        # 1. Standard-Paare prüfen
        for seed in matching_seeds:
            try:
                raw_file = raw_dict[seed]
                raw_file.seek(0)
                raw_json = json.load(raw_file)

                df_temp = pd.DataFrame(raw_json["rawDataBody"]["dataBlock"],
                                       columns=[col["key"] for col in raw_json["rawDataBody"]["dataColumns"]])
                total_time_sec = (df_temp['timestamp_since_poweron'].iloc[-1] - df_temp['timestamp_since_poweron'].iloc[
                    0]) / 1000.0 if 'timestamp_since_poweron' in df_temp.columns else 0.0

                if total_time_sec < mindestdauer_sek:
                    ignoriert_liste.append((seed, total_time_sec, "Normaler Import"))
                else:
                    erfolgreich_liste.append({
                        "seed": seed, "raw_json": raw_json, "df": df_temp,
                        "filename": raw_file.name, "mode": "standard", "label_json": label_dict[seed]
                    })
            except Exception as e:
                fehlerhafte_dateien.append(f"Session {seed}: {str(e)}")

        # 2. Alleinstehende Raws prüfen
        for seed in alleinstehende_raw_seeds:
            try:
                raw_file = raw_dict[seed]
                raw_file.seek(0)
                raw_json = json.load(raw_file)

                df_temp = pd.DataFrame(raw_json["rawDataBody"]["dataBlock"],
                                       columns=[col["key"] for col in raw_json["rawDataBody"]["dataColumns"]])
                total_time_sec = (df_temp['timestamp_since_poweron'].iloc[-1] - df_temp['timestamp_since_poweron'].iloc[
                    0]) / 1000.0 if 'timestamp_since_poweron' in df_temp.columns else 0.0

                if total_time_sec < mindestdauer_sek:
                    ignoriert_liste.append((seed, total_time_sec, "Manueller Override"))
                else:
                    erfolgreich_liste.append({
                        "seed": seed, "raw_json": raw_json, "df": df_temp,
                        "filename": raw_file.name, "mode": "manual_override", "label_json": None
                    })
            except Exception as e:
                fehlerhafte_dateien.append(f"Override-Session {seed}: {str(e)}")

        # --- GESAMTSTATUS ANZEIGEN ---
        st.subheader("📊 Gesamtstatus des Imports")
        st.info(
            f"Anzahl verarbeitbarer Sessions: **{len(erfolgreich_liste)}** | Durch Zeitfilter ignoriert: **{len(ignoriert_liste)}**")

        if fehlerhafte_dateien:
            st.error("⚠️ Folgende Zuordnungsfehler liegen vor:")
            for fehler in fehlerhafte_dateien:
                st.write(f"- {fehler}")

        for seed, dauer, typ in ignoriert_liste:
            st.warning(f"⏭️ Session `{seed}` übersprungen ({typ} - Dauer: {dauer:.1f}s unter Limit).")

        # --- DATEI-VERARBEITUNG & BUNDELING ---
        zip_buffer_dict = {}
        dataframes_for_direct_merge = []

        if erfolgreich_liste:
            st.markdown("---")
            st.subheader("📦 Verfügbare Messungen zur Konvertierung")

            for item in erfolgreich_liste:
                seed = item["seed"]
                raw_json = item["raw_json"]
                df_temp = item["df"].copy()
                orig_filename = item["filename"]

                # NEU & KORRIGIERT: Extraktion über das 'sensorConfigurations' Array
                sensor_configs = raw_json.get("configBody", {}).get("sensorConfigurations", [])

                # Wir bauen ein exaktes Wörterbuch: { sensorIndex: "heater_profile_name" }
                profile_mapping = {}
                for config in sensor_configs:
                    s_idx = config.get("sensorIndex")
                    h_prof = config.get("heaterProfile")
                    if s_idx is not None and h_prof:
                        profile_mapping[int(s_idx)] = str(h_prof)

                # Mapping auf das Dataframe anwenden
                if 'sensor_index' in df_temp.columns and profile_mapping:
                    df_temp['heater_profile_id'] = df_temp['sensor_index'].map(profile_mapping).fillna(
                        "unknown_profile")
                else:
                    df_temp['heater_profile_id'] = "unknown_profile"

                if item["mode"] == "standard":
                    label_file = item["label_json"]
                    label_file.seek(0)
                    label_json = json.load(label_file)
                    label_mapping = {i["labelTag"]: i["labelName"] for i in label_json["labelInformation"]}
                    df_temp['label_name'] = df_temp['label_tag'].map(label_mapping).fillna("Unbekannt")

                    einzigartige_labels = df_temp['label_name'].unique()
                    andere_labels_vorhanden = any(l for l in einzigartige_labels if l != "Initial" and l != "Unbekannt")
                    df_filtered = df_temp[
                        df_temp['label_name'] != "Initial"].copy() if andere_labels_vorhanden else df_temp.copy()

                    uhrzeit_raw = raw_json["rawDataHeader"]["dateCreated_ISO"]
                    uhrzeit_clean = uhrzeit_raw.split('+')[0].replace(':', '-').replace('T', '_')
                    aktive_labels = [l for l in df_filtered['label_name'].unique() if
                                     l != "Initial" and l != "Unbekannt"]
                    label_str = "_".join(aktive_labels) if aktive_labels else "NurInitial"
                    label_str = re.sub(r'[\\/*?:"<>| ]', '_', label_str)

                    output_filename = f"BME688_{uhrzeit_clean}_{seed}_{label_str}.csv"

                    with st.expander(f"🔹 Session {seed} ({label_str})", expanded=False):
                        # Zeige zur Kontrolle die gefundenen Paar-Zuweisungen im UI an
                        gefundene_profile_str = ", ".join(
                            [f"Sensor {k} ➔ {v}" for k, v in sorted(profile_mapping.items())])
                        st.caption(f"**Profil-Mapping:** {gefundene_profile_str}")
                        st.dataframe(df_filtered.head(3))

                    ausgewaehlt = st.checkbox(f"Berücksichtigen ({output_filename})", value=True, key=f"chk_{seed}")
                    if ausgewaehlt:
                        save_history(seed, uhrzeit_raw, aktive_labels if aktive_labels else ["Initial"], orig_filename)

                        csv_buffer = io.StringIO()
                        df_filtered.to_csv(csv_buffer, index=False)
                        zip_buffer_dict[output_filename] = csv_buffer.getvalue()
                        dataframes_for_direct_merge.append(df_filtered)

                elif item["mode"] == "manual_override":
                    with st.expander(f"⚠️ Session {seed} (KEINE LABEL-INFO GEFUNDEN)", expanded=True):
                        st.warning(f"Für die Datei `{os.path.basename(orig_filename)}` fehlt die Label-Information.")
                        manual_label = st.text_input("Bitte gib das gewünschte Label manuell ein:",
                                                     value="HP_Exp_neutrale_Luft_Küche", key=f"txt_{seed}")

                        df_temp['label_name'] = manual_label
                        df_filtered = df_temp.copy()

                        uhrzeit_raw = raw_json["rawDataHeader"]["dateCreated_ISO"]
                        uhrzeit_clean = uhrzeit_raw.split('+')[0].replace(':', '-').replace('T', '_')
                        label_str_clean = re.sub(r'[\\/*?:"<>| ]', '_', manual_label)

                        output_filename = f"BME688_{uhrzeit_clean}_{seed}_{label_str_clean}.csv"
                        gefundene_profile_str = ", ".join([f"S{k}➔{v}" for k, v in sorted(profile_mapping.items())])
                        st.caption(f"**Automatisch verknüpfte Profile:** {gefundene_profile_str}")

                    ausgewaehlt = st.checkbox(f"Berücksichtigen ({output_filename})", value=True, key=f"chk_{seed}")
                    if ausgewaehlt:
                        save_history(seed, uhrzeit_raw, [manual_label], orig_filename)

                        csv_buffer = io.StringIO()
                        df_filtered.to_csv(csv_buffer, index=False)
                        zip_buffer_dict[output_filename] = csv_buffer.getvalue()
                        dataframes_for_direct_merge.append(df_filtered)

            # --- DOWNLOAD-ZENTRALE MIT INTEGRATION ---
            if zip_buffer_dict:
                st.markdown("---")
                st.subheader("📥 Download-Zentrale")

                # Option A: Unverketteter ZIP-Download
                zip_io = io.BytesIO()
                with zipfile.ZipFile(zip_io, "w", zipfile.ZIP_DEFLATED) as archive:
                    for filename, csv_content in zip_buffer_dict.items():
                        archive.writestr(filename, csv_content)
                zip_io.seek(0)

                st.download_button(
                    label=f"📥 Unverkettete Einzel-CSVs als ZIP herunterladen ({len(zip_buffer_dict)} Dateien)",
                    data=zip_io.getvalue(),
                    file_name=f"BME688_Einzeldateien_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                )

                st.markdown(" ")

                # Option B: DIREKT-MERGE IN DER APP
                st.markdown("**⚡ Direkt-Verschmelzung (In-App Merge):**")
                direct_merge_active = st.checkbox("Ausgewählte Sessions direkt zu einer großen ML-CSV verketten")

                if direct_merge_active and dataframes_for_direct_merge:
                    direct_target = st.text_input(
                        "Einheitliches ML-Target für dieses kombinierte Paket (z.B. 'Kaffee', 'Deo'):", value="Kaffee",
                        key="direct_target_input")

                    if st.button("Direkt-Merge ausführen & downloaden"):
                        try:
                            processed_dfs = []
                            for df_sub in dataframes_for_direct_merge:
                                df_copy = df_sub.copy()
                                df_copy['target'] = direct_target
                                processed_dfs.append(df_copy)

                            direct_combined_df = pd.concat(processed_dfs, ignore_index=True)

                            direct_buffer = io.StringIO()
                            direct_combined_df.to_csv(direct_buffer, index=False)

                            st.success(
                                f"✓ {len(processed_dfs)} Sessions im RAM verkettet! Spalte 'target' wurde mit '{direct_target}' befüllt. ({len(direct_combined_df)} Zeilen gesamt)")

                            st.download_button(
                                label="📥 Fertige Gesamt-CSV herunterladen",
                                data=direct_buffer.getvalue(),
                                file_name=f"BME688_DIRECT_MERGED_Target_{direct_target}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Fehler beim Direkt-Merge: {str(e)}")

# ==========================================
# TAB 2: EXTERNE CSV VERKETTUNG
# ==========================================
with tab2:
    st.title("Externe CSV-Dateien verketten (Merge)")
    st.write("Falls du bereits heruntergeladene CSVs von älteren Sessions nachträglich verschmelzen möchtest:")

    global_target = st.text_input("Einheitliches ML-Target für diese Auswahl festlegen:", value="Kaffee",
                                  key="external_target")
    uploaded_csv_files = st.file_uploader("Wähle konvertierte CSV-Dateien aus", type=["csv"],
                                          accept_multiple_files=True)

    if uploaded_csv_files and st.button("Externe Dateien jetzt verketten"):
        try:
            dataframe_list = []
            for file in uploaded_csv_files:
                temp_df = pd.read_csv(file)
                temp_df['target'] = global_target
                dataframe_list.append(temp_df)

            combined_df = pd.concat(dataframe_list, ignore_index=True)
            st.success(f"✓ Erfolgreich verkettet!")

            merged_filename = f"BME688_EXT_MERGED_Target_{global_target}.csv"
            merged_buffer = io.StringIO()
            combined_df.to_csv(merged_buffer, index=False)

            st.download_button("📥 Verkettete Gesamt-CSV herunterladen", data=merged_buffer.getvalue(),
                               file_name=merged_filename, mime="text/csv")
        except Exception as e:
            st.error(f"Fehler: {str(e)}")

# ==========================================
# TAB 3: LABEL ÜBERSICHT
# ==========================================
with tab3:
    st.title("Label-Verzeichnis & Historie")
    history_data = load_history()
    if history_data:
        table_rows = []
        for seed, info in history_data.items():
            table_rows.append({
                "Sitzungs-Seed": seed,
                "Konvertiert am": info.get("Konvertiert am", "Keine Angabe"),
                "Erstellungsdatum Messung (ISO)": info.get("Datum/Uhrzeit (Messung)", "Keine Angabe"),
                "Gefundene Labels": ", ".join(info["Zugeordnete Labels"]),
                "Letzte Quelldatei": info["Ursprungsdatei"]
            })
        df_history = pd.DataFrame(table_rows).sort_values(by="Konvertiert am",
                                                          ascending=False) if history_data else pd.DataFrame()
        st.dataframe(df_history, use_container_width=True)
        if st.button("Historie zurücksetzen"):
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.info("Noch keine konvertierten Dateien in der Historie erfasst.")