# BME688 → Edge Impulse Konverter

Wandelt BME688-Rohdaten-Sessions (8 Sensoren, mehrere Heizprofile mit
unterschiedlich langen Zyklen) in Edge-Impulse-taugliche Gaswiderstands-
Feature-Vektoren um. Verfügbar als Kommandozeilen-Tool (`cli.py`) und als
Streamlit-Weboberfläche (`app.py`).

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```
Danach im Browser: http://localhost:8501

## Per Docker hosten

```bash
docker compose up -d --build
```
Läuft danach unter `http://<server>:8501`.

**Hinweis:** Streamlit bringt keine Authentifizierung mit. Wenn der Server
von außen erreichbar ist, den Container hinter einen Reverse Proxy mit
Basic Auth (z. B. Caddy/nginx) legen oder auf ein VPN beschränken.

## Kommandozeile

```bash
python3 cli.py sessions/ --outdir out --level 1 3 --combine label
```
Alle Optionen: `python3 cli.py --help`

## Train/Test-Split: jetzt oder später?

Standardmäßig wird **kein** Split angewendet (`--apply-split` bzw. die
Checkbox in der App bleiben aus). Jede erzeugte Zeile trägt trotzdem
immer die Spalte `session` (bei `combine=vector` steckt der Session-Name
stattdessen im Dateinamen über die `vector_id`) -- das reicht als
eindeutiger Schlüssel, um den Split beliebig später vorzunehmen, ohne dass
Trainings- und Testdaten sich eine Session teilen.

Im Jupyter Notebook z. B.:

```python
import pandas as pd
from core import train_test_split_by_session

df = pd.read_csv("out/level1_per_sensor/all.csv")
df = train_test_split_by_session(df, test_ratio=0.25, seed=7)

train = df[df.category == "training"]
test  = df[df.category == "testing"]
```

Die Funktion `train_test_split_by_session`:
- teilt immer auf Ebene ganzer Sessions auf (nie werden Zeilen derselben
  Session auf beide Seiten verteilt),
- berechnet den Split getrennt je Label, damit das Verhältnis über die
  Klassen stabil bleibt,
- entfernt automatisch imputierte Vektoren (`n_imputed > 0`) aus der
  Testmenge, da Testdaten ausschließlich echte Messungen enthalten sollten,
- akzeptiert optional `session_split={"session_a": "training", ...}` für
  eine manuelle Vorgabe/Override einzelner Sessions.

Damit könnt ihr im Notebook beliebig oft mit unterschiedlichen
`test_ratio`/`seed`-Werten experimentieren, ohne die Rohdaten erneut zu
konvertieren.

## Dateien

| Datei | Zweck |
|---|---|
| `core.py` | Kernlogik (Zyklen erkennen, Vektoren bauen, Imputation, Split, Output) -- keine CLI/UI-Abhängigkeiten |
| `cli.py` | Kommandozeilen-Wrapper um `core.py` |
| `app.py` | Streamlit-Oberfläche um `core.py` |
| `requirements.txt` | Python-Abhängigkeiten |
| `Dockerfile`, `docker-compose.yml` | Container-Setup zum Hosten |
