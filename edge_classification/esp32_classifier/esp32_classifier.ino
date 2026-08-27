/*
  esp32_classifier.ino
  ---------------------
  Der BME688 hängt HIER am ESP32-S3 (nicht am PC). Ablauf pro Heizzyklus:

    1. Heizprofil BLOCKIEREND durchlaufen: alle 10 Stufen einzeln im Forced
       Mode (measureFullCycle() -- robuster als Sequential-/Parallel-Modus,
       die undokumentiertes Verhalten zeigten, siehe Chat). Dauert insgesamt
       ca. die Summe von heaterDurs[] (bei heater_503: ~28 s).
    2. Vollständigen Rohzyklus als JSON an den PC senden:
         {"type":"raw_cycle","values":[123.4, 567.8, ...]}
       Ungültige/instabile Zyklen werden komplett verworfen und NICHT
       gesendet -- keine Imputation, siehe pc_preprocess_bridge.py.
    3. Auf die vorverarbeitete Antwort vom PC warten:
         {"type":"features","log":[...],"scaled":[...]}
    4. Feature-Vektor an run_classifier() übergeben (Edge-Impulse-Modell OHNE
       Processing-Block -- das Preprocessing passiert vollständig auf dem PC)
       UND an den Random Forest (auf denselben skalierten Werten).
    5. Ergebnis zurück an den PC senden (Anzeige + CSV-Log):
         {"type":"result","label":"...","confidence":0.87,"scores":{...},"rf_label":"..."}

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
  6b. Für die Status-LED an GPIO35 ist KEINE zusätzliche Bibliothek nötig
      (rgbLedWrite() ist Teil des ESP32-Arduino-Cores selbst).
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

// Onboard-RGB-LED des AtomS3 Lite (WS2812B-2020, ein einzelnes LED an GPIO35).
// Gibt visuelles Feedback, auch wenn der serielle Port gerade von
// pc_preprocess_bridge.py belegt ist und ihr den Serial Monitor nicht nutzen
// koennt:
//   Blau (kurz)   -- Start, Sensor wird initialisiert
//   Rot            -- BME688-Initialisierung fehlgeschlagen (I2C-Problem)
//   Gruen          -- bereit, wartet auf Heizzyklen (Normalzustand)
//   Weiss (Blitz)  -- Rohzyklus wurde ans PC gesendet
//   Gelb (Blitz)   -- Klassifikationsergebnis erhalten, dann zurueck zu Gruen
// Hinweis: FastLED UND Adafruit_NeoPixel kollidieren beide mit dem aktuellen
// ESP32-Arduino-Core (3.x, Pin-Remapping via digitalPinToGPIONumber). Deshalb
// hier stattdessen rgbLedWrite() -- Teil des Cores selbst (core-rgb-led.c),
// keine externe Bibliothek noetig, kein Konflikt moeglich.
#define LED_PIN 35

void setStatusLed(uint8_t r, uint8_t g, uint8_t b) {
    rgbLedWrite(LED_PIN, r, g, b);
}

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

static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];       // log-only, fuer run_classifier() (EI-Modell)
static float scaledFeatures[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];  // log10 + StandardScaler, fuer Random Forest / euer eigenes NN

// Callback, über den Edge Impulse die Feature-Daten aus dem statischen Array liest
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

// Scannt den I2C-Bus und listet alle erreichbaren Adressen auf (Klartext,
// nicht JSON -- zum direkten Lesen im Serial Monitor gedacht). Zeigt objektiv,
// ob der BME688 (0x76/0x77) ueberhaupt am Bus sichtbar ist, unabhaengig davon,
// ob bme.begin() das richtig interpretiert -- hilft, Verkabelungs-/
// Stromversorgungsprobleme von Adress-/Logikproblemen zu unterscheiden.
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
        Serial.println("I2C-Scan: KEINE Geraete gefunden! Pruefen: SDA/SCL-Pins "
                        "vertauscht? Sensor an 5V/GND angeschlossen? Kabel/Loetstelle "
                        "locker?");
    } else {
        Serial.printf("I2C-Scan abgeschlossen: %d Geraet(e) gefunden.\n", found);
    }
}

// Laut Aufdruck auf dem Board sind an den genutzten Löchern tatsächlich
// GPIO38 (SDA) und GPIO39 (SCL) herausgeführt -- bestätigt durch
// i2c_pin_finder.ino: BME688 antwortet bei SDA=38, SCL=39 auf Adresse 0x77.
// (Die ursprüngliche Annahme 21/25 war falsch -- GPIO25 existiert auf
// diesem Chip nicht einmal.)
#define BME68X_SDA_PIN 38
#define BME68X_SCL_PIN 39

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

// ---------------------------------------------------------------------
// BME68x-Initialisierung: KEIN Sequential-/Parallel-Modus mehr -- beide
// zeigten undokumentiertes Verhalten (siehe Chat-Diskussion, Bosch-Forum-
// Berichte zu Aussetzern). Stattdessen wird jede Heizstufe einzeln im
// gut dokumentierten Forced Mode angesteuert (siehe measureFullCycle()).
// ---------------------------------------------------------------------
void setupBme688() {
    Wire.begin(BME68X_SDA_PIN, BME68X_SCL_PIN);
    scanI2C();

    // Adresse unbekannt (hängt von SDO-Pin-Beschaltung ab) -- beide probieren.
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
}

// Misst einen vollständigen Heizzyklus BLOCKIEREND, Stufe für Stufe im
// Forced Mode (offiziell dokumentiertes, robustes Bosch-API-Muster --
// vermeidet die undokumentierten Aussetzer von Sequential-/Parallel-Modus,
// siehe forced_mode.ino-Beispiel der Bosch-BME68x-Library). Dauert insgesamt
// ca. die Summe von heaterDurs[] (bei heater_503: ~28 s) -- das ist bewusst
// so, der ganze Zyklus wird ohnehin als atomare Einheit behandelt (danach
// wartet loop() sowieso blockierend auf die PC-Antwort).
// Gibt false zurück, sobald eine einzelne Stufe ungültig/instabil war --
// der gesamte Zyklus wird dann verworfen (keine Imputation).
bool measureFullCycle(float *outResistances) {
    for (int i = 0; i < N_EXPECTED_STEPS; i++) {
        bme.setHeaterProf(heaterTemps[i], heaterDurs[i]);
        bme.setOpMode(BME68X_FORCED_MODE);
        uint32_t measDurUs = bme.getMeasDur(BME68X_FORCED_MODE);
        Serial.printf("DEBUG step=%d temp=%uC dur=%ums measDurUs=%lu\n",
                      i, heaterTemps[i], heaterDurs[i], (unsigned long)measDurUs);
        delayMicroseconds(measDurUs);

        bool gotData = bme.fetchData();
        if (!gotData) {
            Serial.printf("DEBUG step=%d fetchData()=false -- Zyklus abgebrochen. "
                          "Moegliche Ursache: measDurUs zu kurz, oder Sensor antwortet "
                          "nicht mehr (I2C-Verbindung waehrend der Messung verloren?).\n", i);
            return false;
        }

        bme68xData d;
        bme.getData(d);
        Serial.printf("DEBUG step=%d status=0x%02X gas_index=%d gas_res=%.1f "
                      "temp=%.1fC hum=%.1f%% (GASM_VALID=%d HEAT_STAB=%d)\n",
                      i, d.status, d.gas_index, d.gas_resistance, d.temperature, d.humidity,
                      (d.status & BME68X_GASM_VALID_MSK) ? 1 : 0,
                      (d.status & BME68X_HEAT_STAB_MSK) ? 1 : 0);

        if (!(d.status & BME68X_GASM_VALID_MSK) || !(d.status & BME68X_HEAT_STAB_MSK)) {
            Serial.printf("DEBUG step=%d ungueltig/instabil -- Zyklus abgebrochen.\n", i);
            return false;   // Stufe ungültig/instabil -- ganzen Zyklus verwerfen
        }
        outResistances[i] = d.gas_resistance;
    }
    Serial.println("DEBUG Zyklus vollstaendig, alle 10 Stufen gueltig.");
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

    // Allererste Zeile, VOR jeder Sensor-/Bibliothekslogik -- erscheint diese
    // nicht in der Konsole, ist es definitiv kein Sensorproblem, sondern die
    // neue Firmware laeuft nicht bzw. der Port/die Verbindung stimmt nicht.
    Serial.println("DEBUG BOOT esp32_classifier gestartet");

    setStatusLed(0, 0, 40);   // Blau (gedimmt): Start / Initialisierung läuft

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

    setupBme688();   // setzt bei Fehler selbst die LED auf Rot (siehe oben)

    StaticJsonDocument<64> readyDoc;
    readyDoc["type"] = "status";
    readyDoc["status"] = "ready";
    sendJson(readyDoc);
    setStatusLed(0, 40, 0);   // Grün: bereit, wartet auf Heizzyklen
}

void loop() {
    static float rawCycle[N_EXPECTED_STEPS];

    if (!measureFullCycle(rawCycle)) {
        // Zyklus verworfen (ungültige/instabile Stufe) -- keine Imputation,
        // einfach neu versuchen. Kurze Pause, damit nicht sofort wieder
        // dieselbe (evtl. dauerhaft instabile) Messung gestartet wird.
        Serial.println("DEBUG Zyklus verworfen, naechster Versuch in 500ms.");
        delay(500);
        return;
    }

    // 1. Rohzyklus an PC senden
    setStatusLed(30, 30, 30);   // Weiß
    StaticJsonDocument<1024> rawDoc;
    rawDoc["type"] = "raw_cycle";
    JsonArray valuesArr = rawDoc.createNestedArray("values");
    for (int i = 0; i < N_EXPECTED_STEPS; i++) valuesArr.add(rawCycle[i]);
    sendJson(rawDoc);

    // 2. Auf vorverarbeiteten Feature-Vektor warten (zwei Varianten:
    //    features = log-only fuer Edge Impulse, scaledFeatures = fuer
    //    Random Forest / euer eigenes NN)
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
        setStatusLed(40, 40, 0);   // Gelb
        delay(150);
        setStatusLed(0, 40, 0);     // zurück zu Grün
    } else {
        setStatusLed(40, 0, 0);     // Rot
        StaticJsonDocument<128> errDoc;
        errDoc["type"] = "status";
        errDoc["status"] = "features_timeout";
        sendJson(errDoc);
        delay(300);
        setStatusLed(0, 40, 0);     // zurück zu Grün
    }
}
