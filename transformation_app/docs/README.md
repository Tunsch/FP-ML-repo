# BME688 → Edge Impulse Konverter

Wandelt BME688-Rohdaten-Sessions (8 Sensoren, mehrere Heizprofile mit
unterschiedlich langen Zyklen) in Gaswiderstands-Feature-Vektoren um – zur
Nutzung sowohl **in Edge Impulse** als auch **in eigenen Jupyter-Analysen**.

Für die inhaltliche Begründung der einzelnen Verarbeitungsschritte (warum
Log10-Transformation, warum session-basierter Split, etc.) siehe
**`KONZEPT_ERKLAERUNG.md`**. Dieses Dokument hier ist die reine
Bedienungsanleitung.

---

## Inhalt dieses Repos

| Datei | Zweck |
|---|---|
| `core.py` | Kernlogik (keine CLI-/UI-Abhängigkeiten) – auch direkt in Jupyter importierbar |
| `cli.py` | Kommandozeilen-Werkzeug |
| `app.py` | Streamlit-Weboberfläche |
| `ml_notebook_draft.ipynb` | Beispiel-Notebook für eine erste Klassifikations-Baseline |
| `requirements.txt` | Python-Abhängigkeiten |
| `Dockerfile`, `docker-compose.yml` | Container-Setup zum Hosten der Streamlit-App |
| `KONZEPT_ERKLAERUNG.md` | Ausführliche inhaltliche Erklärung aller Verarbeitungsschritte |

---

## Installation

### Lokal (für CLI und/oder Streamlit)

```bash
pip install -r requirements.txt
```

### Per Docker (nur für die Streamlit-App)

```bash
docker compose up -d --build
```
Läuft danach unter `http://<server>:8502` (Port in `docker-compose.yml`
frei wählbar, siehe Abschnitt "Port ändern" unten).

**Hinweis:** Streamlit bringt keine Authentifizierung mit. Wenn der Server
von außen erreichbar ist, den Container hinter einen Reverse Proxy mit
Basic Auth (z. B. Caddy/nginx) legen oder auf ein VPN beschränken.

### Port ändern

Nur in `docker-compose.yml` die linke Seite von `ports:` anpassen, z. B.
`"8502:8501"` → `"9000:8501"`. Der Container lauscht intern weiterhin auf
8501, es ändert sich nur, unter welchem Port ihr von außen zugreift –
Dockerfile/`app.py` müssen dafür nicht angefasst werden.

Ohne Compose direkt mit `docker run`:
```bash
docker build -t bme688-ei-converter .
docker run -d --name bme688-ei-converter -p 9000:8501 bme688-ei-converter
```

---

## Workflow A: Daten für Edge Impulse exportieren

### Erwartete Eingabe

Eine oder mehrere Session-CSVs mit (mindestens) diesen Spalten:
```
sensor_index, sensor_id, timestamp_since_poweron, resistance_gassensor,
heater_profile_step_index, heater_profile_id, label_name
```
Eine Datei = eine Mess-Session = eine durchgängige Klasse.

### Per Streamlit (empfohlen für den Einstieg)

1. `streamlit run app.py` bzw. Container starten, Browser öffnen.
2. In der Sidebar alle Session-CSVs hochladen (Mehrfachauswahl).
3. **Level** wählen: mindestens Level 1 für den Start (siehe
   `KONZEPT_ERKLAERUNG.md` Abschnitt 7 für die Unterschiede).
4. **Vorverarbeitung**: log10 an lassen (Default); Imputation nur aktivieren,
   wenn der angezeigte Diagnose-Report viele Zyklen mit 1–2 fehlenden Stufen
   zeigt.
5. **Combine-Modus**: `vector`, wenn ihr direkt Edge-Impulse-taugliche
   Einzeldateien wollt.
6. **Train/Test-Split**: für den direkten Edge-Impulse-Export könnt ihr ihn
   hier gleich mit festlegen (Checkbox aktivieren) – oder unten in Workflow B
   erst im Notebook.
7. **Verarbeiten** klicken, Vorschau prüfen, ZIP herunterladen.

### Per Kommandozeile

```bash
python3 cli.py sessions/ --outdir out --level 1 --combine vector \
    --apply-split --test-ratio 0.2
```

Alle Optionen: `python3 cli.py --help`

### Ergebnis in Edge Impulse hochladen

Bei `--combine vector` liegt der Output bereits fertig sortiert in
`out/level1_per_sensor/training/` bzw. `.../testing/` vor, jede Datei im
Edge-Impulse-"Single-Reading"-Format (ein Header, eine Datenzeile, kein
Timestamp, Dateiname `<Label>.<Name>.csv`):

- **Studio-UI:** Data Acquisition → Upload → jeweils den kompletten
  `training/`-Ordner mit Kategorie „Training“ hochladen, danach den
  `testing/`-Ordner mit Kategorie „Testing“.
- **CLI:** `edge-impulse-uploader --category training out/level1_per_sensor/training/*.csv`
  (analog für testing).

Bei `--combine session/label/all` sind die Dateien mehrzeilig – dafür im
Studio den **CSV Wizard** nutzen (Data Acquisition → Upload → CSV): dort
`label` als Label-Spalte zuweisen, `session`/`sensor_index`/`heater_profile_id`/
`cycle_id`/`n_imputed` als Metadaten (nicht als Feature!) markieren, und
`training/`- bzw. `testing/`-Dateien mit der jeweils passenden Kategorie
hochladen.

---

## Workflow B: Eigene Analyse in Jupyter

Für eigene Analysen ist `--combine label` oder `--combine all` **ohne**
`--apply-split` meist am praktischsten: wenige, übersichtliche Dateien, bei
denen der Split noch offen ist.

```bash
python3 cli.py sessions/ --outdir out --level 1 --combine label
```
(kein `--apply-split` → Ergebnis liegt unter `out/level1_per_sensor/all.csv`
bzw. je Klasse, jeweils mit vollständigen Metadaten-Spalten)

### Split im Notebook festlegen

`core.py` muss im selben Ordner wie euer Notebook liegen (oder über
`sys.path` erreichbar sein).

```python
import pandas as pd
from core import train_test_split_by_session

df = pd.read_csv("out/level1_per_sensor/all.csv")
df = train_test_split_by_session(df, test_ratio=0.25, seed=7)

train = df[df.category == "training"]
test  = df[df.category == "testing"]
```

Was diese Funktion für euch übernimmt (Details in `KONZEPT_ERKLAERUNG.md`
Abschnitt 9):
- Teilt immer auf Ebene ganzer Sessions auf – nie landen Zeilen derselben
  Session in Training *und* Test.
- Berechnet den Split getrennt je Label, damit `test_ratio` klassenübergreifend
  stabil bleibt.
- Entfernt automatisch imputierte Vektoren (`n_imputed > 0`) aus der
  Testmenge.
- Optional: `session_split={"session_a": "training", ...}` für eine manuelle
  Vorgabe/Override einzelner Sessions.

Ihr könnt beliebig oft mit unterschiedlichen `test_ratio`/`seed`-Werten
experimentieren, ohne die Rohdaten erneut zu konvertieren.

### Fertiges Beispiel-Notebook

`ml_notebook_draft.ipynb` enthält eine komplette Baseline-Pipeline:
Daten laden → session-sauberer Split (inkl. Leakage-Check per Assert) →
Skalierung (nur auf Training gefittet) → Random-Forest-Klassifikator →
`classification_report` + Konfusionsmatrix → `StratifiedGroupKFold`-
Cross-Validation über Sessions für eine stabilere Genauigkeitsschätzung.
Einfach `DATA_DIR` in Zelle 2 auf euren Export-Ordner anpassen und
durchlaufen lassen.

---

## Parameter-Referenz (CLI & Streamlit)

| Parameter | Default | Bedeutung |
|---|---|---|
| `--level` | `1` | Aggregationsstufe(n): 1 = pro Sensor, 2 = pro Heizprofil, 3 = alle Sensoren |
| `--level2-mode` | `concat` | Level 2: Sensorpaar als `concat` (20 Features) oder `mean` (10 Features) |
| `--no-log` | aus | log10-Transformation abschalten |
| `--impute-max-missing` | `0` (aus) | max. Anzahl fehlender Heizstufen je Zyklus, die zeitlich imputiert werden |
| `--impute-max-gap` | `3` | Suchfenster (Zyklen) für Imputations-Nachbarwerte |
| `--combine` | `vector` | `vector` / `session` / `label` / `all` – siehe Abschnitt 10 in `KONZEPT_ERKLAERUNG.md` |
| `--apply-split` | aus | Train/Test-Split jetzt festlegen statt später |
| `--test-ratio` | `0.2` | Anteil der Sessions je Label, die in die Testmenge wandern |
| `--split-seed` | `42` | Zufalls-Seed für die Split-Zuordnung |
| `--session-split` | – | JSON-Datei `{"session_tag": "training"/"testing"}` zur manuellen Vorgabe |
| `--recursive` | aus | Bei Ordner-Input auch Unterordner durchsuchen |
| `--pattern` | `*.csv` | Datei-Muster bei Ordner-Input |

---

## Troubleshooting / FAQ

**„Fehlende Spalten“-Fehler beim Einlesen** – Die CSV muss mindestens die in
Abschnitt "Erwartete Eingabe" genannten Spalten enthalten. Zusätzliche
Spalten (z. B. `temperature`, `pressure`) stören nicht, werden aber aktuell
nicht verwendet.

**Viele Zyklen werden verworfen** – Diagnose-Report prüfen (läuft
automatisch mit, in der App im Expander "Verarbeitungs-Log"): Sind es
wenige fehlende Stufen pro Zyklus (1–2) → Imputation aktivieren. Sind es
viele fehlende Stufen (4+) → meist Rand-Effekt am Session-Anfang/-Ende,
Imputation hilft hier nicht.

**„Label X hat nur 1 Session“-Warnung beim Split** – Mit nur einer Session
je Klasse ist kein leakage-freier Test möglich; diese Session wird komplett
dem Training zugeordnet. Nehmt für jede Klasse mindestens 2, besser 4+
Sessions auf, damit ein aussagekräftiger Test-Split (und die
`StratifiedGroupKFold`-Cross-Validation im Notebook) möglich ist.

**Level 2 „Warnung: Profil X hat N statt 2 Sensoren“** – Wird angezeigt,
wenn nicht genau 2 Sensoren auf dasselbe Heizprofil gemappt sind (z. B. bei
abweichender Hardware-Konfiguration). Dieses Profil wird dann für Level 2
übersprungen; Level 1/3 sind davon nicht betroffen.
