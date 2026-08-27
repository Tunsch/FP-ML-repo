#include <Wire.h>
#include <bsec2.h>
#include <ArduinoJson.h>
#include "bsec_serialized_configurations_selectivity.h"

// Pin-Belegung für ESP32-S3 (AtomS3 Lite)
#define BME68X_SDA_PIN 38
#define BME68X_SCL_PIN 39
#define LED_PIN        35

Bsec2 envSensor;

// Liste der BSEC-Outputs zur Subskription
bsec_virtual_sensor_t sensorList[] = {
    BSEC_OUTPUT_RAW_TEMPERATURE,
    BSEC_OUTPUT_RAW_PRESSURE,
    BSEC_OUTPUT_RAW_HUMIDITY,
    BSEC_OUTPUT_RAW_GAS,
    BSEC_OUTPUT_RAW_GAS_INDEX,
    BSEC_OUTPUT_GAS_ESTIMATE_1,
    BSEC_OUTPUT_GAS_ESTIMATE_2,
    BSEC_OUTPUT_GAS_ESTIMATE_3,
    BSEC_OUTPUT_GAS_ESTIMATE_4
};

void setStatusLed(uint8_t r, uint8_t g, uint8_t b) {
    rgbLedWrite(LED_PIN, r, g, b);
}

void checkBsecStatus(Bsec2 &bsec, const char* context) {
    if (bsec.status < BSEC_OK || bsec.sensor.status < BME68X_OK) {
        setStatusLed(40, 0, 0); // Rot
        while (true) {
            Serial.printf("{\"type\":\"status\",\"status\":\"error\",\"context\":\"%s\",\"bsec_code\":%d,\"sensor_code\":%d}\n",
                          context, bsec.status, bsec.sensor.status);
            delay(1000); // Fehler dauerhaft senden, damit das Python-Skript ihn fangen kann
        }
    } else if (bsec.status > BSEC_OK) {
        Serial.printf("{\"type\":\"status\",\"status\":\"warning\",\"bsec_code\":%d}\n", bsec.status);
    }
}

// Callback, sobald BSEC neue Daten berechnet hat
void newDataCallback(const bme68xData data, const bsecOutputs outputs, Bsec2 bsec) {
    if (!outputs.nOutputs) return;

    StaticJsonDocument<1024> doc;
    doc["type"] = "bsec_result";
    doc["timestamp_ms"] = millis();

    JsonObject raw = doc.createNestedObject("raw");
    JsonObject gas_estimates = doc.createNestedObject("estimates");

    float max_prob = -1.0f;
    int best_class_idx = -1;

    for (uint8_t i = 0; i < outputs.nOutputs; i++) {
        const bsecData output = outputs.output[i];

        switch (output.sensor_id) {
            case BSEC_OUTPUT_RAW_TEMPERATURE:
                raw["temp"] = output.signal;
                break;
            case BSEC_OUTPUT_RAW_HUMIDITY:
                raw["humidity"] = output.signal;
                break;
            case BSEC_OUTPUT_RAW_GAS:
                raw["gas_resistance"] = output.signal;
                break;
            case BSEC_OUTPUT_RAW_GAS_INDEX:
                raw["gas_index"] = (int)output.signal;
                break;
            case BSEC_OUTPUT_GAS_ESTIMATE_1:
            case BSEC_OUTPUT_GAS_ESTIMATE_2:
            case BSEC_OUTPUT_GAS_ESTIMATE_3:
            case BSEC_OUTPUT_GAS_ESTIMATE_4: {
                int class_id = output.sensor_id - BSEC_OUTPUT_GAS_ESTIMATE_1 + 1;
                char key[16];
                snprintf(key, sizeof(key), "class_%d", class_id);
                gas_estimates[key] = output.signal;

                if (output.signal > max_prob) {
                    max_prob = output.signal;
                    best_class_idx = class_id;
                }
                break;
            }
            default:
                break;
        }
    }

    // Bestimmte Zielklasse und Konfidenz
    if (best_class_idx > 0) {
        doc["predicted_class"] = best_class_idx;
        doc["confidence"] = max_prob;
        setStatusLed(0, 40, 0); // Grün bei erfolgreicher Schätzung
    }

    serializeJson(doc, Serial);
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    setStatusLed(0, 0, 40); // Blau beim Boot

    Wire.begin(BME68X_SDA_PIN, BME68X_SCL_PIN);

    // Initialisierung des Sensors über I2C
    if (!envSensor.begin(BME68X_I2C_ADDR_LOW, Wire)) {
        if (!envSensor.begin(BME68X_I2C_ADDR_HIGH, Wire)) {
            checkBsecStatus(envSensor, "I2C_Init");
        }
    }

    // BSEC Version auslesen und ausgeben
    Serial.printf("{\"type\":\"status\",\"status\":\"info\",\"bsec_version\":\"%d.%d.%d.%d\"}\n",
                  envSensor.version.major, 
                  envSensor.version.minor, 
                  envSensor.version.major_bugfix, 
                  envSensor.version.minor_bugfix);

    // BSEC AI-Studio Konfiguration laden
    if (!envSensor.setConfig(bsec_config_selectivity)) {
        checkBsecStatus(envSensor, "SetConfig");
    }

    // Output-Subscriptions aktivieren
    if (!envSensor.updateSubscription(sensorList, sizeof(sensorList) / sizeof(sensorList[0]), BSEC_SAMPLE_RATE_SCAN)) {
        checkBsecStatus(envSensor, "Subscription");
    }

    envSensor.attachCallback(newDataCallback);

    StaticJsonDocument<64> readyDoc;
    readyDoc["type"] = "status";
    readyDoc["status"] = "ready";
    serializeJson(readyDoc, Serial);
    Serial.println();

    setStatusLed(0, 40, 0);
}

void loop() {
    // BSEC-Scheduler regelt die Heiz- und Messzyklen nach hinterlegtem Profil autonom
    if (!envSensor.run()) {
        checkBsecStatus(envSensor, "BSEC_Run");
    }
}