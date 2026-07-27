# BME688 Rohdaten-Konverter – Anleitung

Wandelt die Rohexporte der e-nose (`.bmerawdata` + `.bmelabelinfo`) in
CSV-Dateien um, wie sie die **Transformator-App** (BME688 → Edge Impulse)
als Eingabe erwartet. Beide Apps sind bewusst getrennt: der Konverter
kümmert sich um das Gerätedatenformat, die Transformator-App um
Klassifikations-Feature-Vektoren.

---

## Voraussetzungen

Von der e-nose kommen zwei Dateitypen pro Aufnahme, beide mit demselben
16-stelligen "Seed" im Dateinamen:
- **`.bmerawdata`** – die eigentlichen Sensordaten (Pflicht)
- **`.bmelabelinfo`** – die Label-Zuordnung (optional, siehe "Manueller
  Override" unten)

---

## Tab 1: Datei-Konverter (Smart)

### Ablauf

1. **Mindestdauer-Filter** einstellen (Slider, Default 90 s) – Sessions,
   die kürzer sind, werden automatisch übersprungen (z. B. Testaufnahmen).
2. `.bmerawdata`- und `.bmelabelinfo`-Dateien gemeinsam hochladen
   (Mehrfachauswahl). Die App matched sie automatisch über den Seed im
   Dateinamen.
3. Für jede erkannte Session ein Expander mit Vorschau (erste 3 Zeilen)
   und dem erkannten **Profil-Mapping** (welcher Sensor läuft mit welchem
   Heizprofil) – das steht ganz oben, weil genau diese Information über
   `heater_profile_id` in die CSV geschrieben wird (siehe unten).
4. Checkbox je Session, ob sie in den Export soll (standardmäßig an).

### Zwei Modi

- **Standard** (Raw + Label-Info vorhanden): Label wird automatisch aus
  der `.bmelabelinfo` gemappt. Zeilen mit dem Label `"Initial"` werden
  entfernt, *sofern* die Session auch andere Labels enthält (reiner
  Vorlauf/Leerlauf vor dem eigentlichen Ereignis). Enthält eine Session
  **nur** `"Initial"`, bleibt sie komplett erhalten (Dateiname bekommt
  dann `NurInitial`).
- **Manueller Override** (nur Raw, keine Label-Info gefunden): Ihr gebt
  das Label per Texteingabe selbst ein. Praktisch für Aufnahmen, bei
  denen die Label-Datei verloren ging oder nie erzeugt wurde.

### Woher kommt `heater_profile_id`?

Wird aus `configBody.sensorConfigurations` im `.bmerawdata`-JSON
extrahiert (Mapping Sensor-Index → Heizprofil-Name) und der Tabelle als
Spalte `heater_profile_id` hinzugefügt. Fehlt dieses Feld im Rohexport
(ältere e-nose-Firmware/App-Version) oder ist ein Sensor nicht gemappt,
wird `"unknown_profile"` eingetragen.

**Wichtig für Altdaten:** Frühere Versionen dieser Konverter-App hatten
diese Extraktion noch nicht (siehe Code-Kommentar „NEU & KORRIGIERT" an
der entsprechenden Stelle). CSVs, die mit einer solchen älteren Version
erzeugt wurden, haben die Spalte `heater_profile_id` **gar nicht** im
Header – nicht `"unknown_profile"`, sondern komplett fehlend. Die
Transformator-App erkennt das automatisch und behandelt es mit einem
Platzhalter (Level 1/3 funktionieren normal, Level 2 liefert für diese
Dateien keine Sensorpaare).

### Downloads

- **„📥 Unverkettete Einzel-CSVs als ZIP"** – **dies ist der Export, den
  ihr für die Transformator-App verwenden sollt.** Eine CSV je Session,
  Dateiname `BME688_<Zeitstempel>_<Seed>_<Label(s)>.csv`, Spaltenschema
  identisch zu dem, was die Transformator-App erwartet.
- **„Direkt-Verschmelzung (In-App Merge)"** – verkettet die ausgewählten
  Sessions sofort zu einer einzigen CSV und ersetzt dabei alle Labels
  durch ein einziges, frei eingegebenes `target`. **Nicht für die
  Transformator-App geeignet:** Es gibt danach weder eine `label_name`-
  noch eine Session-Spalte, die Session-Trennung für einen späteren
  Train/Test-Split ist damit unwiederbringlich weg. Nur nutzen, wenn ihr
  die Daten für etwas anderes als die Transformator-App braucht.

---

## Tab 2: Externe CSVs verketten

Nimmt bereits konvertierte CSVs (z. B. aus Tab 1) und verkettet sie zu
einer Gesamtdatei, wobei wie bei der Direkt-Verschmelzung alle
individuellen Labels durch ein einziges `target` ersetzt werden.

**Für die Transformator-App nicht verwenden** – aus demselben Grund wie
oben (keine `label_name`-/Session-Information mehr im Ergebnis). Die
Transformator-App übernimmt Verkettung, Label-Zusammenführung
(Schreibvarianten wie „Rasur"/„HP Exp 3 Rasur" zusammenführen) und
Train/Test-Split ohnehin selbst und session-sauber – dieser Tab ist für
die kombinierte Pipeline redundant.

---

## Tab 3: Label-Übersicht

Zeigt eine Historie aller bisher konvertierten Sessions (Seed, Zeitpunkt
der Konvertierung, zugeordnete Labels, Ursprungsdatei) – gespeichert in
`/app/data/labels_history.json` im Container.

**Achtung bei Docker:** Diese Datei liegt aktuell in keinem gemounteten
Volume. Ohne Volume geht die komplette Historie bei jedem Neubau des
Containers (`docker build`/Recreate) verloren. Empfehlung, falls euch die
Historie wichtig ist:

```yaml
# docker-compose.yml
services:
  bme688-konverter:
    build: .
    ports:
      - "8503:8501"   # Port frei wählbar, nicht mit der Transformator-App kollidieren lassen
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

---

## Empfohlener Gesamt-Workflow

1. `.bmerawdata` + `.bmelabelinfo` in dieser App hochladen (Tab 1).
2. Sessions prüfen/auswählen, **„Unverkettete Einzel-CSVs als ZIP"**
   herunterladen.
3. ZIP entpacken, alle Einzel-CSVs in die **Transformator-App**
   hochladen (Streamlit-UI oder `cli.py` mit einem Ordner).
4. Dort: Level wählen, ggf. Label-Varianten in der Zuordnungstabelle
   zusammenführen, Combine-Modus wählen, Split jetzt oder später (Notebook)
   festlegen.

---

## Bekannter Absturz-Risikofaktor (Empfehlung, optional)

Diese App nutzt `st.dataframe(...)` zur Session-Vorschau (Tab 1). In der
Transformator-App hatten wir einen reproduzierbaren **Segfault in
`libarrow.so`** (der internen Serialisierungsbibliothek hinter
`st.dataframe`/`st.data_editor`), verursacht durch eine zu neue,
CPU-inkompatible PyArrow-Version, die pandas ungepinnt mitzieht. Da diese
Konverter-App dieselbe Kombination (`streamlit` + `pandas` ohne
`pyarrow`-Pin) verwendet, besteht dasselbe Risiko potenziell auch hier.
Vorsorglich empfohlen, in `requirements.txt` zu ergänzen:

```
streamlit>=1.35.0
pandas>=2.0.0
pyarrow==14.0.2
```

und im Dockerfile vor der Paketinstallation:

```dockerfile
ENV ARROW_USE_SIMD=0
```

Sagt Bescheid, falls ich diese beiden Änderungen direkt für euch
einpflegen soll.
