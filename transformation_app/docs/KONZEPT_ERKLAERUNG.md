# Konzept-Erklärung: BME688 → Edge Impulse Konverter

Dieses Dokument erklärt **inhaltlich**, was der Code tut und – vor allem –
**warum** er es so tut. Für die reine Bedienungsanleitung siehe `README.md`.

Referenziert wird der Code in `core.py`, der die gesamte Verarbeitungslogik
enthält (`cli.py` und `app.py` sind nur dünne Wrapper darum).

---

## 1. Das Ausgangsproblem

Ein BME688 ist ein Gassensor, dessen Sensorelement (Metalloxid-Heizplatte)
während eines "Heizprofils" mehrere Temperaturstufen nacheinander durchläuft.
Der Gaswiderstand wird an jeder Stufe einmal gemessen – die "Zeitreihe"
innerhalb eines Zyklus besteht also aus so vielen Punkten wie es Heizstufen
gibt, nicht aus einer kontinuierlichen Abtastung.

Euer Aufbau hat das zusätzlich verschärft:
- **8 Sensoren**, paarweise auf **4 Heizprofile** verteilt (2 Sensoren je Profil).
- Die 4 Profile haben **unterschiedliche Zyklusdauern** (in eurem Beispiel
  ca. 18–28 s), weil die einzelnen Heizstufen unterschiedlich lange dauern.

Edge Impulse erwartet bei seinem klassischen Zeitreihen-Import (CSV/JSON)
aber ein **einziges, konstantes Abtastintervall über alle Achsen einer
Datei hinweg** – das lässt sich mit 8 Sensoren an 4 verschieden getakteten
Profilen nicht direkt abbilden. Die gesamte Pipeline in diesem Repo löst
genau dieses Problem: Sie wandelt die heterogenen Rohdaten in **flache
Feature-Vektoren** um, bei denen "Zeit" durch "Heizstufen-Index" ersetzt
wird – dafür ist kein gemeinsames Zeitraster mehr nötig.

---

## 2. Zyklus-Erkennung (`detect_cycles`)

**Was passiert:** Für jeden Sensor wird die Zeile-für-Zeile-Abfolge des
`heater_profile_step_index` betrachtet. Immer wenn der Wert nicht weiter
ansteigt (z. B. von 9 zurück auf 0), beginnt ein neuer Zyklus.

**Warum nicht einfach die Spalte `scanning_cycle_index` aus dem Rohexport
nutzen?** Weil sie sich in der Praxis als unzuverlässig erwiesen hat – in
eurem Beispielexport stand dort über die gesamte Datei hinweg konstant der
Wert `1`. Der Wrap-Around des Stufenindex ist dagegen ein Signal, das aus
den tatsächlichen Messwerten selbst abgeleitet wird und deshalb robust ist,
unabhängig davon, ob Metadaten-Zähler im Logging korrekt mitgeführt wurden.

---

## 3. Kanonische Stufenzahl (`_canonical_step_count`)

**Was passiert:** Für jeden Sensor wird ermittelt, wie viele *unterschiedliche*
Heizstufen pro Zyklus typischerweise vorkommen (der **Modus**, also der
häufigste Wert über alle erkannten Zyklen).

**Warum nicht einfach `max(step_index) + 1` pro Zyklus nehmen?** Das war der
Ansatz in einer früheren Version und hatte einen blinden Fleck: Fehlt genau
die *letzte* Stufe eines Zyklus (z. B. Stufe 9 von 10), dann ist der
beobachtete Maximalwert in diesem Zyklus 8, und `max()+1` ergibt fälschlich
9 – der fehlende Wert würde also gar nicht als Lücke erkannt. Der Modus über
viele Zyklen hinweg ist robust gegen so einen Einzelfall, weil die meisten
Zyklen in einer Session vollständig sind und damit die tatsächliche
Profillänge korrekt bestimmen.

---

## 4. Diagnose-Report (`report_incomplete_cycles`)

**Was passiert:** Vor jeder Vektor-Erzeugung wird gezählt, wie viele Zyklen
je Sensor vollständig sind, wie viele Stufen im Schnitt fehlen und wie oft
doppelte Messwerte für dieselbe Stufe vorkommen.

**Warum das wichtig ist:** Ohne diesen Report würdet ihr nie erfahren, *wie
viele* Daten durch unvollständige Zyklen verloren gehen und *warum* – z. B.
ob es sich um vereinzelte Ausreißer handelt (dann lohnt sich Imputation) oder
um systematische Rand-Effekte (Aufnahme beginnt/endet mitten in einem
Heizzyklus – das lässt sich nicht sinnvoll imputieren, sondern ist einfach
der Preis für den Sessionanfang/-ende). Genau das habt ihr an eurem
Beispielfile auch gesehen: Die meisten unvollständigen Zyklen dort hatten
4–8 fehlende Stufen (Rand-Effekt), nur wenige genau 1–2 (Imputationskandidaten).

---

## 5. Log10-Transformation der Gaswiderstände

**Was passiert:** `resistance_gassensor` wird standardmäßig mit `log10`
transformiert, bevor irgendetwas anderes damit gemacht wird (auch die
Imputation arbeitet bereits auf den log-transformierten Werten).

**Warum:** MOX-Gassensoren wie der BME688 liefern Widerstandswerte, die über
mehrere Größenordnungen streuen – in eurem Beispiel von wenigen Tausend Ohm
bis über 10 Millionen Ohm, insbesondere bedingt durch die erste
Reinigungs-/Burn-in-Heizstufe jedes Zyklus. Ohne Log-Transformation würde
dieser eine Wert bei jedem distanzbasierten Verfahren (z. B. Skalierung,
k-NN, neuronale Netze mit Standard-Initialisierung) die anderen, eigentlich
aussagekräftigeren Stufen dominieren. Log10 komprimiert diese Spannweite auf
eine handhabbare Größenordnung und macht cycle-zu-cycle-Veränderungen an
jeder Stufe vergleichbar. Abschaltbar über `log_transform=False`, falls ihr
aus gutem Grund mit Rohwerten arbeiten wollt.

---

## 6. Zeitliche Imputation fehlender Heizstufen

**Was passiert:** Fehlt bei einem Zyklus eine einzelne Heizstufe (bis zu
`impute_max_missing` Stufen), wird ihr Wert aus **demselben Stufenindex des
zeitlich nächsten vorherigen und/oder folgenden Zyklus desselben Sensors**
linear interpoliert (gewichtet nach Zyklus-Abstand). Ist nur eine
Nachbarseite innerhalb von `impute_max_gap` Zyklen verfügbar, wird deren Wert
übernommen (Carry-Forward/-Backward).

**Warum genau so und nicht anders:**
- **Zeitliche statt stufenübergreifende Interpolation:** Jede Heizstufe
  entspricht einem eigenen Temperatur-Arbeitspunkt mit einer eigenen,
  physikalisch bedingten Größenordnung des Gaswiderstands. Eine fehlende
  Stufe aus den *benachbarten Stufen desselben Zyklus* zu schätzen, würde
  zwei nicht vergleichbare physikalische Messungen vermischen. Der
  Gaswiderstand *derselben* Stufe ändert sich dagegen von Zyklus zu Zyklus
  (bei euren ~20 s Zykluszeit) meist nur graduell – außer genau während des
  Beginns eines zu erkennenden Ereignisses, was die Hauptgrenze dieses
  Ansatzes ist (siehe unten).
- **Obergrenze `impute_max_missing`:** Je mehr Stufen fehlen, desto weniger
  Information bleibt tatsächlich gemessen, und desto mehr wird der Vektor
  zu einer Erfindung. Ab einer gewissen Schwelle (Default: aus, empfohlen
  max. 1–2 von z. B. 10 Stufen) überwiegt das Risiko, ein rein synthetisches
  Muster zu lernen.
- **Suchfenster `impute_max_gap`:** Verhindert, dass über sehr große
  zeitliche Lücken hinweg interpoliert wird (z. B. über eine
  Session-Unterbrechung hinweg), wo die Annahme "ändert sich nur graduell"
  nicht mehr haltbar ist.
- **`n_imputed`-Spalte:** Jeder Vektor trägt fortan mit, wie viele seiner
  Werte imputiert wurden – nichts passiert "unsichtbar".
- **Sicherheitsnetz in `train_test_split_by_session`:** Imputierte Vektoren
  werden automatisch aus der Testmenge entfernt, unabhängig vom gewählten
  Schwellenwert. Der Grund: Ein Testset soll die reale Modellleistung auf
  echten Sensordaten widerspiegeln, nicht auf teilweise erfundenen Werten.
  Imputation ist also ausschließlich ein Mittel, um mehr *Trainings*daten zu
  gewinnen, nie um die Evaluation zu beeinflussen.

**Wann ist Imputation sinnvoll?** Schaut euch den Diagnose-Report an: Wenn
viele Zyklen mit *genau 1–2* fehlenden Stufen auftreten (typisch bei
sporadischen Funk-/Übertragungsaussetzern), lohnt sich Imputation. Wenn die
fehlenden Stufen sich auf wenige Zyklen mit *vielen* fehlenden Stufen
konzentrieren (typisch bei Session-Rand-Effekten), bringt Imputation wenig,
weil diese Zyklen ohnehin über der Schwelle liegen und verworfen werden.

---

## 7. Drei Aggregationsstufen (Level 1/2/3)

Alle drei Level nutzen ausschließlich den Gaswiderstand (wie gewünscht),
unterscheiden sich aber darin, *wie viele physische Sensoren* pro erzeugtem
Vektor zusammengefasst werden:

| Level | Was wird kombiniert | Warum das interessant ist |
|---|---|---|
| **1 – pro Sensor** | Ein Zyklus, ein Sensor | Feinste Granularität, meiste Trainingsbeispiele, keinerlei Synchronisationsproblem zwischen Sensoren. Guter Startpunkt. |
| **2 – pro Heizprofil** | Die 2 Sensoren desselben Heizprofils, deren Zyklen per nächstliegendem Startzeitpunkt gematcht werden | Testet, ob die (redundante) zweite Sensorinstanz zusätzliche, nicht-redundante Information liefert (z. B. leicht unterschiedliche Platzierung/Exposition) |
| **3 – alle Sensoren** | Alle 8 Sensoren, ausgerichtet an einem "Super-Zyklus" | Nutzt die volle Sensor-Vielfalt (unterschiedliche Heizprofile = unterschiedliche Gas-Selektivität), am nächsten an einem realistischen Live-Deployment |

**Warum bei Level 3 der "Anker" der langsamste Sensor ist:** Bei Level 3
muss zu einem gemeinsamen Zeitpunkt von *jedem* der 8 Sensoren ein
vollständiger Zyklus vorliegen. Der Sensor mit der längsten Zyklusdauer gibt
also unweigerlich den Takt vor – schneller laufende Sensoren haben zu jedem
Zeitpunkt des Ankers bereits mehrere abgeschlossene Zyklen zur Auswahl, von
denen jeweils der zuletzt abgeschlossene verwendet wird. Das bildet exakt
nach, wie es im späteren Live-Betrieb aussehen würde: Man nimmt von jedem
Sensor den aktuellsten fertigen Messwert, auch wenn der nicht taufrisch ist.

**Warum bei Level 2 ein Zeit-Matching statt einfachem Reihenfolge-Pairing:**
Die zwei Sensoren desselben Profils laufen zwar mit derselben
Profil-Zykluszeit, aber nicht zwangsläufig exakt synchron (leicht versetzter
Start, minimale Drift). Ein Matching nach zeitlicher Nähe (mit 5 s
Toleranzgrenze, danach wird das Paar verworfen) ist robuster als eine reine
Positions-Zuordnung (1. Zyklus von A mit 1. Zyklus von B), die bei Drift mit
der Zeit auseinanderlaufen würde.

---

## 8. Warum diese Metadaten-Spalten (`vector_id`, `session`, `label`, `n_imputed`)

Jeder erzeugte Vektor trägt vier Informationen mit, die für nichts an der
eigentlichen Klassifikation gebraucht werden, aber für **Nachvollziehbarkeit
und korrekte Auswertung unverzichtbar** sind:

- **`session`**: identifiziert die Ursprungsdatei. Der wichtigste Schlüssel
  überhaupt, weil er die Grundlage für einen leakage-freien Train/Test-Split
  ist (siehe Abschnitt 9).
- **`label`**: die Klasse, unverändert aus `label_name` übernommen.
- **`vector_id`**: ein eindeutiger, stabiler Bezeichner
  (`<session>__L<level>__<laufende Nummer>`), damit sich ein einzelner
  Vektor auch nach mehrfachem Kombinieren/Filtern/Exportieren immer wieder
  demselben Ursprungszyklus zuordnen lässt.
- **`n_imputed`**: siehe Abschnitt 6 – Transparenz über künstlich aufgefüllte
  Werte.

Diese Spalten werden bei `--combine vector` bewusst *nicht* mit in die
CSV-Datei geschrieben (Edge Impulse würde sie sonst als Features
missverstehen), sondern stecken stattdessen im Dateinamen bzw. der
Ordnerstruktur. Bei den anderen Combine-Modi (`session`, `label`, `all`)
bleiben sie als echte Spalten erhalten, weil dort ohnehin eine
Nachbearbeitung (CSV-Wizard-Import, eigene Analyse) vorgesehen ist.

---

## 9. Der Train/Test-Split (`train_test_split_by_session`)

**Kernregel:** Eine komplette Session gehört immer entweder zu Training
*oder* zu Test, nie zu beiden.

**Warum das entscheidend ist:** Aufeinanderfolgende Zyklen einer Session
sind stark korreliert – gleiche Umgebungsbedingungen, gleiche Sensordrift,
gleiches Hintergrundrauschen. Ein zufälliger Split auf Ebene einzelner
Zeilen/Vektoren würde dazu führen, dass Vektoren derselben Session sowohl im
Training als auch im Test landen. Ein Modell könnte dann teilweise nur die
*Session* wiedererkennen statt das eigentliche Gasereignis – die gemessene
Testgenauigkeit wäre dadurch geschönt und würde die tatsächliche Leistung
auf neuen, unbekannten Messungen überschätzen (Data Leakage).

**Warum die Aufteilung getrennt je Label berechnet wird:** Ohne das könnte
es passieren, dass z. B. alle Sessions einer bestimmten Klasse zufällig ins
Training rutschen und die Testmenge diese Klasse gar nicht mehr enthält.
Die Berechnung je Label sorgt dafür, dass das Split-Verhältnis
(`test_ratio`) über alle Klassen hinweg ungefähr stabil bleibt.

**Warum das eine eigenständige Funktion ist, die nicht zwangsläufig beim
Export läuft:** Ursprünglich war der Split fester Teil der Export-Pipeline.
Das hat sich als zu unflexibel herausgestellt, sobald Auswertung/Analyse in
einem Jupyter Notebook stattfinden soll, wo man mit `test_ratio`, `seed`
oder einer manuellen Session-Zuordnung experimentieren möchte, ohne jedes
Mal die Rohdaten neu zu konvertieren. Deshalb: Export (`build_outputs`) und
Split (`train_test_split_by_session`) sind entkoppelt; Ersteres kann
Letzteres optional aufrufen (`apply_split=True`), oder ihr ruft die
Split-Funktion später selbst auf denselben, unveränderten Export-Daten auf
– identisches Verhalten, nur zu einem späteren Zeitpunkt.

---

## 10. Die vier Combine-Modi und ihre Ziel-Workflows

| Modus | Struktur | Für welchen Workflow gedacht |
|---|---|---|
| `vector` | 1 Datei je Vektor, keine Metadaten-Spalten (stecken im Dateinamen) | Direkter Upload zu Edge Impulse im "Single-Reading"-Format – kein CSV-Wizard nötig, nur Ordner hochladen |
| `session` | 1 Datei je Session, mehrzeilig, mit `session`/`label`/`n_imputed` als Spalten | Edge Impulse CSV-Wizard (Label-/Metadaten-Spalten zuweisen) *oder* Jupyter-Analyse je Session |
| `label` | 1 Datei je Klasse, alle Sessions dieser Klasse zusammen | Guter Kompromiss für Jupyter-Analyse: wenige, übersichtliche Dateien, `session`-Spalte bleibt zur Split-Bildung erhalten |
| `all` | 1 Gesamtdatei | Einfachster Einstieg für eine schnelle Jupyter-Analyse über alle Klassen hinweg |

Die Wahl hat also nichts mit unterschiedlicher *Information* zu tun (überall
stecken dieselben Vektoren drin), sondern nur damit, wie ihr die Daten
anschließend weiterverarbeiten wollt.

---

## 11. Architektur: `core.py` / `cli.py` / `app.py`

`core.py` enthält ausschließlich reine Datenverarbeitungsfunktionen ohne
Kenntnis von Kommandozeile oder Weboberfläche. `cli.py` und `app.py`
importieren dieselben Funktionen und unterscheiden sich nur darin, wie sie
Eingaben entgegennehmen (Dateipfade vs. hochgeladene Dateien) und Ergebnisse
ausgeben (Dateien auf Platte vs. ZIP-Download/Vorschau im Browser). Diese
Trennung hat zwei Vorteile: Erstens verhält sich die Konvertierungslogik in
CLI und Web-UI garantiert identisch (kein doppelt gepflegter Code), zweitens
lässt sich `core.py` genau deshalb auch direkt in einem Jupyter Notebook
importieren (siehe `ml_notebook_draft.ipynb`), ohne Streamlit oder argparse
mitschleppen zu müssen.
