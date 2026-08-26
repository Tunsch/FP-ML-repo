/*
  esp32_classifier.ino
  ---------------------
  Der BME688 hängt HIER am ESP32-S3 (nicht am PC). Ablauf pro Heizzyklus:

    1. Heizprofil durchlaufen, rohe Gaswiderstände je Heizstufe sammeln
       (readNextHeaterStep() -- an eure bestehende Sensor-Anbindung anpassen,
       siehe TODO unten).
    2. Vollständigen Rohzyklus als JSON an den PC senden:
         {"type":"raw_cycle","values":[123.4, 567.8, ...]}
       Unvollständige Zyklen (Wrap-Around vor Erreichen aller N_EXPECTED_STEPS)
       werden verworfen und NICHT gesendet -- keine Imputation, siehe
       pc_preprocess_bridge.py.
    3. Auf die vorverarbeitete Antwort vom PC warten:
         {"type":"features","values":[0.12,-0.34, ...]}
    4. Feature-Vektor an run_classifier() übergeben (Edge-Impulse-Modell OHNE
       Processing-Block -- das Preprocessing passiert vollständig auf dem PC).
    5. Ergebnis zurück an den PC senden (nur zur Anzeige/Logging):
         {"type":"result","label":"...","confidence":0.87,"scores":{...}}

  Vorbereitung:
  1. Edge Impulse: Modell OHNE Processing-Block als "Arduino library"
     exportieren.
  2. Exportierte .zip einbinden: Sketch -> Include Library -> Add .ZIP Library...
  3. Include unten (<PROJECT_NAME>_inferencing.h) an euren Bibliotheksnamen anpassen.
  4. Bibliothek "ArduinoJson" (Benoit Blanchon) über den Library Manager installieren.
  5. N_EXPECTED_STEPS unten MUSS mit preprocessing_artifact.json::n_expected_steps
     und mit EI_CLASSIFIER_NN_INPUT_FRAME_SIZE übereinstimmen -- wird beim Start
     geprüft (siehe setup()).
  6. Bibliothek "BME68x Sensor library" (Bosch Sensortec) über den Library
     Manager installieren.
  7. Heizprofil-Werte (heaterTemps[]/heaterDurs[]) sind bereits mit den
     echten "heater_413"-Werten aus eurer .bmerawdata-Beispieldatei belegt.
     Nur relevant, falls ihr je das Heizprofil in Edge Impulse wechselt.
  8. I2C-Pins (21/25, AtomS3 Lite Grove-Port) sind gesetzt. Die I2C-Adresse
     wird beim Start automatisch probiert (0x76 dann 0x77) -- nichts weiter
     zu tun, solange der zweite Sensor am Bus eine andere Adresse hat.
*/

#include <Wire.h>
#include <bme68xLibrary.h>
#include <ArduinoJson.h>
#include <e-nose-2_inferencing.h>   // Name eures tatsächlichen Edge-Impulse-Exports
#include "random_forest_model.h"    // von export_random_forest_to_c.py erzeugt, ins Sketch-Verzeichnis kopieren
using namespace Eloquent::ML::Port;  // micromlgen generiert Klassen in diesem Namespace

// Klassenreihenfolge MUSS zur alphabetischen sklearn-Reihenfolge passen (siehe
// print-Ausgabe von export_random_forest_to_c.py) -- bei euch identisch zur
// Edge-Impulse-Kategorie-Reihenfolge.
const char* RF_CLASS_LABELS[] = { "Deospray", "Rasur", "Raumspray", "Zähneputzen" };
RandomForestGasClassifier rf_classifier;

// Manuelle Vorwärtsdeklaration: die Arduino-IDE-Autoprototypisierung kommt mit
// dem Parametertyp JsonDocument& nicht zuverlässig klar und lässt sendJson()
// sonst unter Umständen ohne Deklaration -- ohne dieses Fix schlägt der Build
// mit "'sendJson' was not declared in this scope" fehl.
void sendJson(JsonDocument &doc);

#define N_EXPECTED_STEPS 10   // Heizprofil hat immer 10 Stufen -- fixer Wert
#define RESPONSE_TIMEOUT_MS 3000

// M5Stack AtomS3 Lite: BME688 hängt am Grove-Port (Pins 21/25), 5V + GND
// versorgt. Ein weiterer (ungenutzter) Sensor hängt am selben Bus -- das ist
// unproblematisch, solange er eine andere I2C-Adresse als der BME688 hat.
#define BME68X_SDA_PIN 21
#define BME68X_SCL_PIN 25
// SDO-Pin-Beschaltung ist unbekannt -- Adresse wird beim Start automatisch
// probiert (0x76 = SDO auf GND, 0x77 = SDO auf VDDIO), siehe setupBme688().

// Werte des Heizprofils "heater_503" (extrahiert aus configBody.heaterProfiles
// in der .bmerawdata-Beispieldatei: temperatureTimeVectors [Temp°C, Einheiten],
// Dauer = Einheiten * timeBase(140ms); genutzt von Sensor-Index 2/3 laut
// configBody.sensorConfigurations). BME68X_I2C_ADDR und die I2C-Pins in
// Wire.begin() (siehe setupBme688()) müsst ihr noch an eure ESP32-S3-
// Verdrahtung anpassen -- das steht nicht in der Rohdatendatei.
// WICHTIG: Bei Wechsel des Heizprofils muss auch config.heater_profile
// (config.py) und das Edge-Impulse-Training auf "heater_503" umgestellt
// bzw. neu trainiert werden -- die aktuelle e-nose-2-Bibliothek ist noch
// auf heater_413 trainiert!
static uint16_t heaterTemps[N_EXPECTED_STEPS] = { 210, 280, 280, 350, 350, 280, 210, 140, 70, 140 };
static uint16_t heaterDurs[N_EXPECTED_STEPS]  = { 3360, 280, 3080, 280, 3080, 3360, 3360, 3360, 3360, 3360 };

Bme68x bme;
static bme68xData bmeData[N_EXPECTED_STEPS];
static uint8_t bmeNFetched = 0;
static uint8_t bmeReadIdx = 0;

static float rawCycle[N_EXPECTED_STEPS];
static bool stepFilled[N_EXPECTED_STEPS];
static int lastStepIndex = -1;

static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];       // log-only, fuer run_classifier() (EI-Modell)
static float scaledFeatures[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];  // log10 + StandardScaler, fuer euer eigenes NN

// Callback, über den Edge Impulse die Feature-Daten aus dem statischen Array liest
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void resetCycle() {
    for (int i = 0; i < N_EXPECTED_STEPS; i++) stepFilled[i] = false;
}

// ---------------------------------------------------------------------
// BME68x-Initialisierung: sequenzieller Modus mit festem 10-Stufen-
// Heizprofil (siehe heaterTemps/heaterDurs oben -- TODO: echte Werte
// eintragen, bevor ihr live testet).
// ---------------------------------------------------------------------
void setupBme688() {
    Wire.begin(BME68X_SDA_PIN, BME68X_SCL_PIN);

    // Adresse unbekannt (hängt von SDO-Pin-Beschaltung ab) -- beide probieren.
    bme.begin(0x76, Wire);
    if (bme.checkStatus() == BME68X_ERROR) {
        bme.begin(0x77, Wire);
    }
    if (bme.checkStatus() == BME68X_ERROR) {
        StaticJsonDocument<64> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "bme688_init_failed";
        sendJson(errDoc);
        return;
    }

    bme.setTPH();   // Standard-Oversampling für Temperatur/Druck/Feuchte
    bme.setHeaterProf(heaterTemps, heaterDurs, N_EXPECTED_STEPS);
    bme.setOpMode(BME68X_SEQUENTIAL_MODE);
}

// Liefert die nächste GÜLTIGE Heizstufen-Messung (gas_index, gas_resistance).
// Nicht-blockierend: gibt false zurück, wenn gerade keine neuen Daten bereit
// sind oder die aktuelle Stufe noch nicht stabil war (heat_stab) -- loop()
// ruft die Funktion dann einfach beim nächsten Durchlauf erneut auf.
//
// bme.getData() liefert IMMER nur EINE Messung pro Aufruf (nicht mehrere auf
// einmal) -- nach fetchData() wird getData() daher wiederholt aufgerufen, bis
// keine weiteren Messungen mehr bereitstehen (bis zu 3 im sequenziellen Modus,
// analog zu Field 0/1/2 im Datenblatt).
bool readNextHeaterStep(int &stepIndex, float &resistance) {
    if (bmeReadIdx >= bmeNFetched) {
        bmeReadIdx = 0;
        bmeNFetched = 0;
        if (!bme.fetchData()) return false;   // noch keine neue Messung bereit

        while (bmeNFetched < 3) {
            bme68xData d;
            uint8_t got = bme.getData(d);
            if (got == 0) break;
            bmeData[bmeNFetched] = d;
            bmeNFetched++;
        }
        if (bmeNFetched == 0) return false;
    }

    bme68xData &d = bmeData[bmeReadIdx];
    bmeReadIdx++;

    if (!(d.status & BME68X_GASM_VALID_MSK) || !(d.status & BME68X_HEAT_STAB_MSK)) {
        return false;   // Heizstufe noch nicht stabil/gültig -- überspringen
    }

    stepIndex = d.gas_index;
    resistance = d.gas_resistance;
    return true;
}

void sendJson(JsonDocument &doc) {
    serializeJson(doc, Serial);
    Serial.println();
}

// Wartet blockierend (mit Timeout) auf eine {"type":"features",...}-Antwort
// vom PC. Liefert BEIDE Varianten zurueck:
//   outLogFeatures    -> fuer run_classifier() (Edge-Impulse-Modell, das
//                        seine Skalierung selbst mitbringt)
//   outScaledFeatures -> fuer euer eigenes NN (log10 + StandardScaler,
//                        noch ungenutzt bis dessen Inferenzcode dazukommt)
bool waitForFeatures(float *outLogFeatures, float *outScaledFeatures, size_t n, unsigned long timeoutMs) {
    unsigned long start = millis();
    while (millis() - start < timeoutMs) {
        if (Serial.available()) {
            String line = Serial.readStringUntil('\n');
            line.trim();
            if (line.length() == 0) continue;

            StaticJsonDocument<2048> doc;
            if (deserializeJson(doc, line)) continue;
            if (String((const char *)doc["type"]) != "features") continue;

            JsonArray logArr = doc["log"];
            JsonArray scaledArr = doc["scaled"];
            if (logArr.isNull() || logArr.size() != n) return false;
            if (scaledArr.isNull() || scaledArr.size() != n) return false;
            for (size_t i = 0; i < n; i++) {
                outLogFeatures[i] = logArr[i].as<float>();
                outScaledFeatures[i] = scaledArr[i].as<float>();
            }
            return true;
        }
    }
    return false; // Timeout
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }

    if (N_EXPECTED_STEPS != EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) {
        StaticJsonDocument<128> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "config_mismatch";
        errDoc["n_expected_steps"] = N_EXPECTED_STEPS;
        errDoc["ei_frame_size"] = EI_CLASSIFIER_NN_INPUT_FRAME_SIZE;
        sendJson(errDoc);
        // Läuft trotzdem weiter, damit die Fehlermeldung sichtbar bleibt --
        // aber Klassifikation wird mit diesem Mismatch nicht funktionieren.
    }

    setupBme688();

    resetCycle();
    StaticJsonDocument<64> readyDoc;
    readyDoc["type"] = "status";
    readyDoc["status"] = "ready";
    sendJson(readyDoc);
}

void loop() {
    int stepIndex;
    float resistance;
    if (!readNextHeaterStep(stepIndex, resistance)) return;

    bool wrapped = (lastStepIndex != -1 && stepIndex <= lastStepIndex);
    if (wrapped) {
        bool complete = true;
        for (int i = 0; i < N_EXPECTED_STEPS; i++) {
            if (!stepFilled[i]) { complete = false; break; }
        }

        if (complete) {
            // 1. Rohzyklus an PC senden
            StaticJsonDocument<1024> rawDoc;
            rawDoc["type"] = "raw_cycle";
            JsonArray valuesArr = rawDoc.createNestedArray("values");
            for (int i = 0; i < N_EXPECTED_STEPS; i++) valuesArr.add(rawCycle[i]);
            sendJson(rawDoc);

            // 2. Auf vorverarbeiteten Feature-Vektor warten (zwei Varianten:
            //    features = log-only fuer Edge Impulse, scaledFeatures = fuer
            //    das eigene NN, sobald dessen Inferenzcode dazukommt)
            if (waitForFeatures(features, scaledFeatures, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE, RESPONSE_TIMEOUT_MS)) {
                // 3. Edge-Impulse-Klassifikation
                signal_t signal;
                signal.total_length = EI_CLASSIFIER_NN_INPUT_FRAME_SIZE;
                signal.get_data = &raw_feature_get_data;

                ei_impulse_result_t result = { 0 };
                EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false /* debug */);

                // Random Forest laeuft parallel zum Edge-Impulse-Modell, auf
                // denselben scaledFeatures (log10 + StandardScaler) -- siehe
                // export_random_forest_to_c.py.
                int rf_class_idx = rf_classifier.predict(scaledFeatures);
                const char *rf_label = (rf_class_idx >= 0 &&
                    rf_class_idx < (int)(sizeof(RF_CLASS_LABELS) / sizeof(RF_CLASS_LABELS[0])))
                    ? RF_CLASS_LABELS[rf_class_idx] : "unknown";

                StaticJsonDocument<512> outDoc;
                if (res != EI_IMPULSE_OK) {
                    outDoc["type"] = "status";
                    outDoc["status"] = "classifier_failed";
                    outDoc["code"] = (int)res;
                } else {
                    outDoc["type"] = "result";
                    JsonObject scores = outDoc.createNestedObject("scores");
                    float best_score = -1.0f;
                    const char *best_label = "";
                    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
                        float v = result.classification[i].value;
                        scores[result.classification[i].label] = v;
                        if (v > best_score) {
                            best_score = v;
                            best_label = result.classification[i].label;
                        }
                    }
                    outDoc["label"] = best_label;
                    outDoc["confidence"] = best_score;
                    outDoc["rf_label"] = rf_label;
                }
                sendJson(outDoc);
            } else {
                StaticJsonDocument<128> errDoc;
                errDoc["type"] = "status";
                errDoc["status"] = "features_timeout";
                sendJson(errDoc);
            }
        }
        // Unvollständiger Zyklus: einfach verwerfen, keine Meldung nötig
        resetCycle();
    }

    rawCycle[stepIndex] = resistance;
    stepFilled[stepIndex] = true;
    lastStepIndex = stepIndex;
}
