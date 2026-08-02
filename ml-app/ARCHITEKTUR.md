# Architektur: ML-Experiment-App

Diese App trennt die BME688-Vektor-Erzeugung (`core.py`, unverändert) von
der ML-Experimentierlogik (`ml/`). Das Notebook (`notebooks/analysis.ipynb`)
enthält keine Trainings-/Auswertungslogik mehr, sondern ruft nur noch eine
Funktion auf und plottet das Ergebnis.

## Verzeichnisstruktur

```
app/
├── core.py                  # unverändert: Rohdaten -> Feature-Vektor-CSVs
├── run_experiment.py        # CLI-Einstiegspunkt für Batch-Läufe
├── ml/
│   ├── __init__.py
│   ├── config.py            # ExperimentConfig (Dataclass)
│   ├── data.py               # Feature-CSVs laden, Split (nutzt core.py)
│   ├── preprocessing.py      # Skalierung, rein fit/transform
│   ├── models/
│   │   ├── __init__.py
│   │   └── registry.py       # Modell-Registry (Name -> Factory)
│   ├── evaluation.py         # Metriken, KEIN Plotting
│   ├── tracking.py           # Ergebnisse persistieren (CSV, JSON, Ordner)
│   └── pipeline.py           # run_experiment() -- verdrahtet alles
└── notebooks/
    └── analysis.ipynb        # nur noch: run_experiment() + Plots
```

## Grundprinzip

Jedes Modul hat genau eine Verantwortung und kommuniziert nur über
einfache Datenstrukturen (DataFrames, Dataclasses) -- nie über
Seiteneffekte wie "trainiert und plottet nebenbei". Dadurch ist jeder
Baustein isoliert austauschbar und mit `pytest` testbar, unabhängig von
Jupyter.

## Datenfluss

```
core.py (unveraendert)
    | Rohdaten -> Feature-Vektor-CSVs (data/level*_per_*/…)
    v
ExperimentConfig                    (ml/config.py)
    v
ml.pipeline.run_experiment(config)
    |
    |-- ml.data           : CSVs laden, feature_cols bestimmen,
    |                        train_test_split_by_session() aus core.py
    |-- ml.preprocessing  : Scaler nur auf Trainingsdaten fitten
    |-- ml.models.registry: Modellname -> sklearn-Objekt
    |-- (Training/Predict + Zeitmessung direkt in pipeline.py)
    |-- ml.evaluation     : Metriken aus y_test/y_pred berechnen
    v
ExperimentResult                    (ml/tracking.py)
    |-- tracking.save_artifacts()  -> results/<id>/config.json, report.txt
    |-- tracking.append_to_csv()   -> experiment_results.csv
    v
Notebook: Confusion-Matrix-/Feature-Importance-Plots, Modellvergleich
```

## Die Module im Einzelnen

### `ml/config.py` -- `ExperimentConfig`

Eine Dataclass, die ein Experiment vollständig beschreibt: Datenpfad,
Feature-Metadaten (informativ), Split-Parameter, Skalierung, Modellname +
Hyperparameter, Freitext-Notizen, Ablagepfade. Es gibt bewusst **keine**
YAML-Dateien -- eine Config entsteht direkt im Notebook oder in
`run_experiment.py` als normales Python-Objekt.

Wichtig: die Config trägt nur den **Modellnamen** (String), nie ein
Modellobjekt selbst. Das entkoppelt Konfiguration von Implementierung
(siehe Registry unten).

### `ml/data.py` -- dünner Adapter zu `core.py`

Drei Funktionen:

- `load_feature_table(data_dir)`: liest alle CSVs in einem Ordner ein,
  bestimmt `feature_cols` (alles, was nicht in `META_COLUMNS` steht).
- `split_data(df, config)`: ruft `train_test_split_by_session` **aus
  core.py** auf (kein eigener Split-Code) und prüft zusätzlich per
  Assertion, dass keine Session in beiden Mengen landet.
- `get_xy(df, feature_cols)`: extrahiert X/y-Arrays.

Enthält keinerlei Logik zur Umwandlung von Rohdaten in Vektoren -- das
bleibt exklusiv Aufgabe von `core.py`. Die Abhängigkeit läuft nur in eine
Richtung: `ml/` importiert aus `core.py`, nie umgekehrt.

### `ml/preprocessing.py`

`fit_scaler(X_train, scaling)` und `apply_scaler(scaler, X)`. Reine
Funktionen ohne globalen Zustand. Der Scaler wird immer nur auf
Trainingsdaten gefittet. Weitere Vorverarbeitungsschritte (PCA,
Feature-Selektion, ...) würden hier als eigene Funktionen ergänzt, ohne
`pipeline.py` anzufassen.

### `ml/models/registry.py` -- der Kernbaustein für Erweiterbarkeit

Ein Dictionary `{Modellname: Factory-Funktion}`. Neue Modelle werden per
Decorator registriert:

```python
@register("gradient_boosting")
def _build_gradient_boosting(params: dict):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**params)
```

`build_model(name, params)` ist die einzige Stelle, die einen Modellnamen
in ein konkretes sklearn-Objekt übersetzt. Die restliche Pipeline kennt
nie eine konkrete Modellklasse, nur diese Funktion. Das ist die direkte
Antwort auf "ich weiß noch nicht, welche Klassifikatoren ich brauchen
werde": ein neues Modell = eine neue Factory-Funktion, **kein**
bestehender Code wird geändert. Aktuell registriert: `random_forest`,
`svm`, `knn`, `mlp`, `logistic_regression`.

Die Imports der jeweiligen sklearn-Klasse stehen bewusst innerhalb der
Factory-Funktion, nicht am Modulanfang -- falls irgendwann eine Bibliothek
wie `xgboost` dazukommt, die nicht überall installiert ist, bleibt
`registry.py` trotzdem für alle anderen Modelle importierbar.

### `ml/evaluation.py`

`evaluate(model, y_test, y_pred, feature_names)` berechnet Accuracy,
Precision, Recall, F1, False-Positive-Rate (nur bei 2 Klassen definiert),
Confusion Matrix, `classification_report`-Text und -- falls vorhanden --
`feature_importances_`. Gibt ein `EvaluationResult`-Dataclass zurück.

**Bewusst kein Plotting hier.** Das hält das Modul batch-tauglich (auch
ohne Display nutzbar) und einfach mit `pytest` testbar. Visualisierung
passiert ausschließlich im Notebook.

### `ml/tracking.py`

Die einzige Stelle, die weiß, *wie* ein Ergebnis gespeichert wird:

- `generate_experiment_id(model_name)`: `YYYYMMDD_HHMMSS_<Modellname>`
- `make_exp_dir(results_dir, experiment_id)`: legt `results/<id>/` an
- `save_artifacts(result)`: schreibt `config.json` und `report.txt` in
  diesen Ordner (Plots kommen später vom Notebook dazu -- gleicher
  Ordner, damit alle Artefakte eines Experiments beisammenbleiben)
- `append_to_csv(result_file, row)`: hängt eine Zeile an
  `experiment_results.csv` an
- `load_results(result_file)`: liest die komplette Experiment-Historie
  für den Modellvergleich

`ExperimentResult` bündelt Config + Auswertung + Timing + Pfade + das
trainierte Modell selbst (für Notebook-Plots wie Feature-Importance) in
einem Objekt.

### `ml/pipeline.py` -- die Verdrahtung

`run_experiment(config)` ist der **einzige** Ort, der die Reihenfolge
data → preprocessing → models → evaluation → tracking kennt. Jeder Schritt
ist über sein eigenes Modul einzeln testbar und austauschbar, ohne dass
sich diese Datei ändert.

### `run_experiment.py` -- CLI

Dünner Wrapper um `run_experiment()` für Batch-Läufe ohne Notebook (z. B.
mehrere Modelle über Nacht durchlaufen lassen). Nutzt exakt dieselbe
Pipeline-Funktion wie das Notebook -- es gibt keine zweite Implementierung
der Logik.

### `notebooks/analysis.ipynb`

Enthält nur noch: Config bauen → `run_experiment()` aufrufen → Plots
zeichnen (Confusion-Matrix-Heatmap, Feature-Importance-Balken) → am Ende
`load_results()` für den sortierten Modellvergleich. Keine Trainings-
oder Metriklogik mehr im Notebook selbst.

## Neues Modell ergänzen -- Schritt für Schritt

1. In `ml/models/registry.py` eine Factory-Funktion schreiben und mit
   `@register("neuer_name")` registrieren.
2. Im Notebook (oder in `run_experiment.py`) eine `ExperimentConfig` mit
   `model_name="neuer_name"` anlegen.
3. Fertig -- `run_experiment()`, Evaluation, Tracking und Plots
   funktionieren unverändert.

## Neue Metrik ergänzen

In `ml/evaluation.py`: Metrik berechnen, als Feld in `EvaluationResult`
und im Rückgabewert von `evaluate()` ergänzen. `ExperimentResult.as_csv_row()`
in `ml/tracking.py` um die entsprechende Spalte erweitern, falls sie auch
in die CSV soll.

## Neues Feature-Level / neue Datenquelle

`core.py` bleibt unverändert die einzige Quelle für "Rohdaten → Vektor".
Ein neues Level bedeutet nur einen neuen `data_dir` in der
`ExperimentConfig` -- `ml/data.py` liest jeden Ordner mit dem erwarteten
Spaltenformat automatisch ein.

## Warum core.py nicht in ml/ aufgeht

`core.py` beantwortet "wie wird aus rohen Sensordaten ein Feature-Vektor?"
-- eine der ML-Pipeline vorgelagerte, unabhängige Frage. `cli.py` und
`app.py` (Streamlit) hängen von `core.py` ab, ohne je etwas von `ml/` zu
wissen. Würde man core.py-Funktionen nach `ml/` verschieben, liefe diese
Abhängigkeit rückwärts. Stattdessen importiert `ml/data.py` gezielt nur
`train_test_split_by_session` und `ml/tracking.py` nur `sanitize_label`
aus `core.py` -- der Rest bleibt dort, wo er hingehört:

```
core.py  <-  cli.py
core.py  <-  app.py
core.py  <-  ml/data.py, ml/tracking.py  <-  ml/pipeline.py  <-  Notebook / CLI
```

## Testbarkeit

Da keine Funktion in `ml/` Jupyter, Matplotlib-Display oder globalen
Zustand voraussetzt, lässt sich die gesamte Pipeline mit `pytest` und
einem kleinen synthetischen DataFrame testen, z. B.:

```python
def test_run_experiment_random_forest(tmp_path, fake_feature_csv):
    config = ExperimentConfig(
        data_dir=fake_feature_csv, model_name="random_forest",
        results_dir=tmp_path / "results", result_file=tmp_path / "results.csv",
    )
    result = run_experiment(config)
    assert 0 <= result.evaluation.accuracy <= 1
```

Das war im ursprünglichen All-in-One-Notebook praktisch nicht möglich.

## Bekannte Grenzen / bewusste Vereinfachungen

- Es gibt keine Validierung der `model_params` gegen das jeweilige
  Modell -- falsche Parameter wirft sklearn selbst mit einer klaren
  Fehlermeldung, eine zusätzliche Validierungsschicht schien angesichts
  der überschaubaren Modellanzahl nicht nötig.
- `experiment_results.csv` wird bei jedem Lauf komplett neu eingelesen
  und geschrieben (nicht angehängt) -- bei sehr vielen tausend Zeilen
  würde sich das bemerkbar machen; für den aktuellen Umfang unkritisch.
- Cross-Validation über Sessions (wie im ursprünglichen Notebook gezeigt)
  ist noch nicht Teil von `ml/pipeline.py` -- ließe sich als
  `run_cv_experiment()`-Variante in `pipeline.py` ergänzen, die
  `evaluate()` pro Fold aufruft und die Werte mittelt.
