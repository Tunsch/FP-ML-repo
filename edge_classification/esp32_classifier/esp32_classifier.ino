/*
  esp32_classifier.ino
  ---------------------
  Der BME688 hängt HIER am ESP32-S3 (nicht am PC). Ablauf pro Heizzyklus:

    1. Heizprofil BLOCKIEREND durchlaufen: alle 10 Stufen im PARALLEL
       Mode (measureFullCycle() -- nutzt das Hardware-Heizprofil des BME688,
       verhindert das Abkühlen der Heizplatte zwischen den Stufen).
    2. Vollständigen Rohzyklus als JSON an den PC senden.
    3. Auf die vorverarbeitete Antwort vom PC warten.
    4. Feature-Vektor an run_classifier() übergeben.
    5. Ergebnis zurück an den PC senden (Anzeige + CSV-Log).
*/

#include <Wire.h>
#include <bme68xLibrary.h>
#include <ArduinoJson.h>
#include <e-nose-2_inferencing.h>   // Name eures tatsächlichen Edge-Impulse-Exports
#include "random_forest_model.h"    // von export_random_forest_to_c.py erzeugt

using namespace Eloquent::ML::Port;

#define LED_PIN 35

void setStatusLed(uint8_t r, uint8_t g, uint8_t b) {
    rgbLedWrite(LED_PIN, r, g, b);
}

const char* RF_CLASS_LABELS[] = { "Deospray", "Rasur", "Raumspray", "Zähneputzen" };
RandomForestGasClassifier rf_classifier;

void sendJson(JsonDocument &doc);

#define N_EXPECTED_STEPS 10   // Heizprofil hat immer 10 Stufen -- fixer Wert
#define RESPONSE_TIMEOUT_MS 3000

static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];       
static float scaledFeatures[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];  

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void scanI2C() {
    Serial.println("I2C-Scan startet...");
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        uint8_t error = Wire.endTransmission();
        if (error == 0) {
            Serial.printf("  I2C-Geraet gefunden bei Adresse 0x%02X\n", addr);
            found++;
        }
    }
    if (found == 0) {
        Serial.println("I2C-Scan: KEINE Geraete gefunden! Pruefen: SDA/SCL-Pins vertauscht?");
    } else {
        Serial.printf("I2C-Scan abgeschlossen: %d Geraet(e) gefunden.\n", found);
    }
}

// Angepasste Pins für den AtomS3 Lite (aus deiner Pin-Analyse)
#define BME68X_SDA_PIN 38
#define BME68X_SCL_PIN 39

// ---------------------------------------------------------------------
// Werte des Heizprofils "heater_503"
// ---------------------------------------------------------------------
static uint16_t heaterTemps[N_EXPECTED_STEPS] = { 210, 280, 280, 350, 350, 280, 210, 140, 70, 140 };

// Im Parallelmodus MÜSSEN wir Multiplikatoren verwenden (Millisekunden / 140ms)
// 3360/140=24, 280/140=2, 3080/140=22 ...
static uint16_t heaterMultipliers[N_EXPECTED_STEPS] = { 24, 2, 22, 2, 22, 24, 24, 24, 24, 24 };

Bme68x bme;

// ---------------------------------------------------------------------
// BME68x-Initialisierung
// ---------------------------------------------------------------------
void setupBme688() {
    Wire.begin(BME68X_SDA_PIN, BME68X_SCL_PIN);
    scanI2C();

    bme.begin(0x76, Wire);
    if (bme.checkStatus() == BME68X_ERROR) {
        bme.begin(0x77, Wire);
    }
    if (bme.checkStatus() == BME68X_ERROR) {
        setStatusLed(40, 0, 0);   // Rot
        Serial.print("DEBUG bme688_init_failed, statusString=");
        Serial.println(bme.statusString());
        StaticJsonDocument<64> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "bme688_init_failed";
        sendJson(errDoc);
        return;
    }
    Serial.print("DEBUG BME688 init OK, statusString=");
    Serial.println(bme.statusString());

    bme.setTPH();   // Standard-Oversampling für Temperatur/Druck/Feuchte

    // PARALLEL-MODUS KONFIGURATION:
    // Die Basiszeit ist hier 140ms. Davon müssen wir die Zeit abziehen, die der
    // Sensor intern für Temperatur/Druck/Feuchte (TPH) braucht.
    uint16_t tphDurMs = bme.getMeasDur(BME68X_PARALLEL_MODE) / 1000;
    uint16_t sharedHeatrDur = 140 - tphDurMs; 
    
    // Die Überladung von setHeaterProf mit 4 Argumenten schaltet den Sensor
    // korrekt für den Parallel-Modus scharf!
    bme.setHeaterProf(heaterTemps, heaterMultipliers, sharedHeatrDur, N_EXPECTED_STEPS);
}

// ---------------------------------------------------------------------
// Messzyklus im Parallel-Modus abfahren
// ---------------------------------------------------------------------
bool measureFullCycle(float *outResistances) {
    bme68xData data;
    uint8_t nFieldsLeft = 0;
    
    bool step_measured[N_EXPECTED_STEPS] = {false};
    int steps_collected = 0;

    // Startet das hardwareseitige Abfahren des Profils ohne Abkühlung
    bme.setOpMode(BME68X_PARALLEL_MODE);
    Serial.println("DEBUG Start Parallel Mode Cycle...");

    unsigned long startTime = millis();
    unsigned long maxDuration = 35000; // Maximale Dauer für das Profil liegt bei ~28s

    while (steps_collected < N_EXPECTED_STEPS) {
        if (millis() - startTime > maxDuration) {
            Serial.println("DEBUG Timeout beim Warten auf Parallel Mode Daten.");
            bme.setOpMode(BME68X_SLEEP_MODE); 
            return false;
        }

        delay(50); 

        if (bme.fetchData()) {
            do {
                nFieldsLeft = bme.getData(data); 
                uint8_t step_idx = data.gas_index;

                if (step_idx < N_EXPECTED_STEPS && !step_measured[step_idx]) {
                    
                    if ((data.status & BME68X_GASM_VALID_MSK) && 
                        (data.status & BME68X_HEAT_STAB_MSK)) {
                        
                        outResistances[step_idx] = data.gas_resistance;
                        step_measured[step_idx] = true;
                        steps_collected++;
                        
                        Serial.printf("DEBUG step=%d status=0x%02X gas_res=%.1f "
                                      "temp=%.1fC hum=%.1f%%\n",
                                      step_idx, data.status, data.gas_resistance, 
                                      data.temperature, data.humidity);
                    }
                }
            } while (nFieldsLeft > 0);
        }
    }
    
    // Zyklus fertig: Sensor zurück in den Schlafmodus
    bme.setOpMode(BME68X_SLEEP_MODE);

    Serial.println("DEBUG Parallel Zyklus vollstaendig, alle 10 Stufen gueltig.");
    return true;
}

void sendJson(JsonDocument &doc) {
    serializeJson(doc, Serial);
    Serial.println();
}

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

    Serial.println("DEBUG BOOT esp32_classifier gestartet");

    setStatusLed(0, 0, 40);

    if (N_EXPECTED_STEPS != EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) {
        StaticJsonDocument<128> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "config_mismatch";
        errDoc["n_expected_steps"] = N_EXPECTED_STEPS;
        errDoc["ei_frame_size"] = EI_CLASSIFIER_NN_INPUT_FRAME_SIZE;
        sendJson(errDoc);
    }

    setupBme688();

    StaticJsonDocument<64> readyDoc;
    readyDoc["type"] = "status";
    readyDoc["status"] = "ready";
    sendJson(readyDoc);
    setStatusLed(0, 40, 0);
}

void loop() {
    static float rawCycle[N_EXPECTED_STEPS];

    if (!measureFullCycle(rawCycle)) {
        Serial.println("DEBUG Zyklus verworfen, naechster Versuch in 500ms.");
        delay(500);
        return;
    }

    setStatusLed(30, 30, 30);
    StaticJsonDocument<1024> rawDoc;
    rawDoc["type"] = "raw_cycle";
    JsonArray valuesArr = rawDoc.createNestedArray("values");
    for (int i = 0; i < N_EXPECTED_STEPS; i++) valuesArr.add(rawCycle[i]);
    sendJson(rawDoc);

    if (waitForFeatures(features, scaledFeatures, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE, RESPONSE_TIMEOUT_MS)) {
        signal_t signal;
        signal.total_length = EI_CLASSIFIER_NN_INPUT_FRAME_SIZE;
        signal.get_data = &raw_feature_get_data;

        ei_impulse_result_t result = { 0 };
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

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
        setStatusLed(40, 40, 0);
        delay(150);
        setStatusLed(0, 40, 0);
    } else {
        setStatusLed(40, 0, 0);
        StaticJsonDocument<128> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "features_timeout";
        sendJson(errDoc);
        delay(300);
        setStatusLed(0, 40, 0);
    }
}