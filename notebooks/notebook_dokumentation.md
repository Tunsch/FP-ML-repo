# Dokumentation: BME688-Experiment-Framework (`ml_notebook_revised.ipynb`)

Diese Datei erklärt Aufbau, Logik und Verwendung des überarbeiteten
Notebooks. Sie ergänzt die Markdown-Zellen im Notebook selbst und dient
als Referenz, wenn ihr das Framework später erweitert.

## 1. Grundidee

Das ursprüngliche Notebook trainierte genau einen Random Forest gegen
genau einen Feature-Datensatz. Für den geplanten Vergleich mehrerer
Modelle (Random Forest, SVM, KNN, MLP, ...) über mehrere Feature-Level
und Vorverarbeitungsvarianten hinweg wurde das Notebook zu einem
**wiederverwendbaren Experiment-Framework** umgebaut:

- Eine **zentrale Konfigurationszelle** beschreibt jedes Experiment
  eindeutig.
- Eine **einzige Funktion** (`evaluate_model`) übernimmt Training,
  Auswertung, Visualisierung und Protokollierung für jedes beliebige
  sklearn-kompatible Modell.
- Jedes Experiment landet automatisch als Zeile in
  `experiment_results.csv` sowie als Plot-/Report-Ordner unter
  `results/<Experiment-ID>/`.

Für ein neues Modell müsst ihr im Regelfall **nichts** am Rahmencode
ändern -- nur ein neues Modellobjekt anlegen und `evaluate_model(...)`
aufrufen.

## 2. Ablauf des Notebooks (Zelle für Zelle)

| Abschnitt | Zweck |
|---|---|
| 1. Daten laden | CSVs aus `DATA_DIR` einlesen, `feature_cols` bestimmen |
| 2. Experiment-Konfiguration | Alle Einstellungen für den aktuellen Lauf an einer Stelle |
| 3. Session-Split | `train_test_split_by_session` aus `core.py`, mit Leakage-Check |
| 4. Skalierung | `StandardScaler`, nur wenn `SCALING = True`, nur auf Trainingsdaten gefittet |
| 5. `evaluate_model()` | Zentrale Trainings-/Auswertungsfunktion (Details unten) |
| 6. Random Forest | Erstes Modell über `evaluate_model()` |
| 7. Weitere Modelle | SVM, KNN, MLP als Vorlage für eigene Modelle |
| 8. Cross-Validation | `StratifiedGroupKFold` über Sessions für ein stabileres Bild |
| 9. Modellvergleich | `experiment_results.csv` laden, nach F1 sortieren |
| 10. Nächste Schritte | Offene Punkte / Ausblick |

## 3. Wie neue Modelle ergänzt werden

Jedes sklearn-kompatible Modell (mit `.fit()` / `.predict()`) lässt sich
so einbinden:

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)

logreg_result = evaluate_model(
    logreg, X_train_model, X_test_model, y_train, y_test, train_df, test_df,
)
```

Wichtig: Wenn das Modell distanz- oder gradientenbasiert ist (SVM, KNN,
MLP, Logistische Regression), sollte `SCALING = True` in Abschnitt 2
gesetzt sein, bevor ihr `X_train_model` / `X_test_model` erzeugt (Zelle
in Abschnitt 4). Für Random Forest ist das Flag praktisch irrelevant.

## 4. Wie neue Features ergänzt werden

Die Feature-Spalten werden automatisch aus allen Spalten von `df`
bestimmt, die nicht in `meta_cols` stehen (Abschnitt 1). Um neue
Merkmale zu ergänzen, reicht es, sie vor dem Laden in die CSV
aufzunehmen (z. B. in `bme688_to_ei.py`) oder nach dem Laden als neue
Spalte in `df` zu berechnen -- `feature_cols` erfasst sie dann
automatisch mit. Tragt in `FEATURE_SET` (Abschnitt 2) einen sprechenden
Namen ein (z. B. `"mean_std"`, `"all_features"`, `"PCA20"`), damit sich
Experimente mit unterschiedlichen Feature-Sets in der CSV auseinanderhalten
lassen.

## 5. Wie neue Metriken ergänzt werden

Alle Kennzahlen werden im `result`-Dictionary innerhalb von
`evaluate_model()` zusammengestellt (Abschnitt 5 im Notebook). Um eine
weitere Metrik zu ergänzen:

1. Metrik berechnen (z. B. mit einer weiteren Funktion aus
   `sklearn.metrics`).
2. Einen zusätzlichen Eintrag im `result`-Dictionary hinzufügen.
3. Optional: die Spalte in Abschnitt 9 (Modellvergleich) mit ausgeben.

Da `result_df` beim Schreiben mit der bestehenden CSV zusammengeführt
wird (`pd.concat`), tauchen für ältere Zeilen ohne die neue Spalte
automatisch `NaN`-Werte auf -- das ist unkritisch, sollte euch aber beim
Sortieren/Filtern bewusst sein.

## 6. Aufbau von `experiment_results.csv`

| Spalte | Bedeutung |
|---|---|
| `Timestamp` | Zeitpunkt des Experiments |
| `Experiment ID` | Eindeutige ID `YYYYMMDD_HHMMSS_<Modellname>`, verweist auf `results/<ID>/` |
| `Model` | Klassenname des Modells (`model.__class__.__name__`) |
| `Parameters` | Alle Hyperparameter als JSON (`model.get_params()`) |
| `Feature Level` | Verwendetes Rohdaten-/Aggregationslevel |
| `Feature Set` | Konkrete verwendete Merkmale |
| `Imputation` | Umgang mit fehlenden Werten |
| `Scaling` | Ob `StandardScaler` angewendet wurde |
| `Test Ratio`, `Seed` | Split-Parameter für Reproduzierbarkeit |
| `Train/Test Vectors`, `Train/Test Sessions` | Größe von Training/Test |
| `Accuracy`, `Precision`, `Recall`, `F1` | Standard-Klassifikationsmetriken (gewichtet gemittelt) |
| `False Positive Rate` | Nur bei binärer Klassifikation definiert, sonst `NaN` |
| `Training Time`, `Prediction Time` | Laufzeiten in Sekunden |
| `Notes` | Freitext aus `EXPERIMENT_NOTES` |

Da die Datei bei jedem Aufruf von `evaluate_model()` neu eingelesen,
ergänzt und komplett neu geschrieben wird, wächst sie über beliebig viele
Notebook-Sitzungen hinweg an -- solange ihr sie nicht löscht oder
umbenennt.

## 7. Warum `model.get_params()` verwendet wird

`get_params()` ist bei allen sklearn-Schätzern verfügbar und liefert
**alle** Hyperparameter des Modells, nicht nur die explizit gesetzten --
inklusive der Defaultwerte. Das ist wichtig für Reproduzierbarkeit: Wenn
ihr in sechs Monaten eine CSV-Zeile mit guter F1 seht, lässt sich daraus
das exakte Modell rekonstruieren, ohne dass ihr euch an die damals
verwendeten Defaults erinnern müsst. Die Serialisierung als JSON-String
hält die CSV dabei in einer einzigen Zelle lesbar.

## 8. Wie die False Positive Rate berechnet wird

Die FPR (`fp / (fp + tn)`) ist nur für **binäre** Klassifikation
eindeutig definiert. Bei den vier BME688-Klassen im Beispiel ist sie
aktuell `NaN`. Sobald ihr ein Experiment mit nur zwei Klassen fahrt (z.
B. "Zielgeruch erkannt vs. nicht erkannt"), wird sie automatisch über
`sklearn.metrics.confusion_matrix` berechnet: `tn`, `fp`, `fn`, `tp`
ergeben sich aus dem Ravel der 2x2-Konfusionsmatrix.

## 9. Wie Experimente reproduziert werden

Für ein reproduzierbares Experiment müsst ihr folgende Werte aus der
CSV-Zeile übernehmen:

- `Parameters` (JSON) → als `**json.loads(row["Parameters"])` an die
  Modellklasse übergeben
- `Feature Level`, `Feature Set`, `Imputation`, `Scaling` → in Abschnitt
  2 (Konfiguration) eintragen
- `Test Ratio`, `Seed` → ebenfalls in Abschnitt 2, steuern den
  Session-Split

Damit läuft exakt derselbe Trainings-/Testsplit und dieselbe
Modellkonfiguration erneut.

## 10. Warum diese Kennzahlen gespeichert werden

- **Accuracy / Precision / Recall / F1**: Standard-Vergleichsgrößen für
  Multiklassen-Klassifikation; `average="weighted"` berücksichtigt die
  (hier ungleiche) Klassengrößen.
- **False Positive Rate**: bei binären Anwendungsfällen (z. B.
  "Gasereignis erkannt ja/nein") oft die praktisch relevantere Größe als
  reine Accuracy, gerade bei unbalancierten Klassen.
- **Training/Prediction Time**: relevant für die spätere
  Embedded-Deployment-Entscheidung -- ein Modell mit minimal besserer
  Accuracy, aber deutlich höherer Inferenzzeit, ist für Edge Impulse
  ggf. keine gute Wahl.
- **Train/Test Sessions/Vectors**: macht sichtbar, auf wie wenig
  unabhängigen Sessions ein Ergebnis eigentlich beruht -- wichtig für die
  Einschätzung der statistischen Aussagekraft.

## 11. Wie das Framework später erweitert werden kann

- **`GridSearchCV` / `RandomizedSearchCV`**: Als Splitter
  `StratifiedGroupKFold` (mit `groups=train_df["session"]`) übergeben,
  damit auch die Hyperparametersuche session-sauber bleibt. Das beste
  Modell (`.best_estimator_`) anschließend wie gewohnt an
  `evaluate_model()` übergeben.
- **`sklearn.pipeline.Pipeline`**: `StandardScaler` und Modell in einer
  Pipeline bündeln, damit Vorverarbeitung und Modell gemeinsam
  persistiert/deployt werden können, statt den Scaler separat zu
  verwalten.
- **Cross-Validation statt Einzelsplit für alle Modelle**: Abschnitt 8
  zeigt das Prinzip für den Random Forest; lässt sich analog in
  `evaluate_model()` integrieren (Mittelwert/Standardabweichung über
  Folds statt eines Einzelwerts), falls die Session-Anzahl je Klasse
  wächst.
- **Weitere Feature-Level (2/3)**: `DATA_DIR` und `FEATURE_LEVEL`
  anpassen, restlicher Code bleibt unverändert.

## 12. Bekannte Grenzen

- Bei wenigen Sessions je Klasse (aktuell z. B. nur 2 für "Deospray")
  schwankt ein einzelner Split stark -- die Cross-Validation in Abschnitt
  8 sollte für belastbare Aussagen immer mit herangezogen werden.
- Die FPR ist nur für binäre Probleme aussagekräftig; bei vier Klassen
  bleibt sie `NaN` und sollte nicht überinterpretiert werden.
- `experiment_results.csv` wird bei jedem Lauf komplett neu geschrieben
  (nicht angehängt) -- bei sehr vielen Experimenten (mehrere tausend
  Zeilen) kann das spürbar langsamer werden. Für den aktuellen Umfang
  ist das unkritisch.
