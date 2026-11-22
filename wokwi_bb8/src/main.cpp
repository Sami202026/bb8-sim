#include <Arduino.h>
#include <Wire.h>
#include <ESP32Servo.h>
#include <math.h>
#include "bb8_policy.h"

// ============================================================================
// PINOUT ESP32-S3 (identique au schéma de câblage réel)
// ============================================================================
constexpr int PIN_VBATT_ADC   = 1;
constexpr int PIN_LED_BMS     = 2;
constexpr int PIN_ENC_A       = 4;
constexpr int PIN_ENC_B       = 5;
constexpr int PIN_I2C_SDA     = 8;
constexpr int PIN_I2C_SCL     = 9;
constexpr int PIN_MOTOR_PWM   = 10;
constexpr int PIN_MOTOR_DIR   = 11;
constexpr int PIN_SERVO_STEER = 13;
constexpr int PIN_SERVO_HEAD  = 14;

// Constantes physiques & sécurité (alignées sur bb8_env.py)
constexpr uint8_t MPU_ADDR       = 0x68;
constexpr float   DT_CTRL        = 0.02f;   // 50 Hz
constexpr float   SPHERE_RADIUS  = 0.25f;   // Rayon sphère BB-8 (m)
constexpr float   VBATT_CUTOFF   = 12.6f;   // Coupure BMS LiPo 4S
constexpr float   DIVIDER_RATIO  = 5.7f;    // (47k + 10k) / 10k
constexpr float   TICKS_PER_REV  = 20.0f;   // KY-040 Wokwi (remplacer par CPR*réducteur en réel)
constexpr float   MAX_STEER_RAD  = 0.50f;   // ±28.6°

Servo servoSteer;
Servo servoHead;

// Encodeur en quadrature (interruption matérielle)
volatile long encTicks = 0;
long prevEncTicks = 0;

void IRAM_ATTR onEncoderA() {
  if (digitalRead(PIN_ENC_B) == digitalRead(PIN_ENC_A)) {
    encTicks = encTicks + 1;
  } else {
    encTicks = encTicks - 1;
  }
}

// État capteurs
float g_est[3]     = {0.0f, 0.0f, -1.0f};
float accel_ms2[3] = {0.0f, 0.0f, 9.81f};
float gyro_rads[3] = {0.0f, 0.0f, 0.0f};
float omega_enc    = 0.0f;
float v_batt       = 16.0f;
float steer_pos    = 0.0f;  // Position pendule (rad) — remplacer par AS5600 ou STS3215 en réel

// État interne PPO v3 (identique à bb8_env.py)
bool  ppo_mode       = true;
float cmd_v          = 0.0f;
float cmd_yaw        = 0.0f;
float action_filt[2] = {0.0f, 0.0f};
float v_err_int      = 0.0f;
float yaw_err_int    = 0.0f;

// Consignes mode manuel
float manual_pwm   = 0.0f;
float manual_steer = 0.0f;

bool initMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  return (Wire.endTransmission() == 0);
}

bool readMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) return false;

  auto read16 = []() -> int16_t {
    return (int16_t)((Wire.read() << 8) | Wire.read());
  };

  int16_t ax = read16(), ay = read16(), az = read16();
  (void)read16();
  int16_t gx = read16(), gy = read16(), gz = read16();

  accel_ms2[0] = (ax / 16384.0f) * 9.81f;
  accel_ms2[1] = (ay / 16384.0f) * 9.81f;
  accel_ms2[2] = (az / 16384.0f) * 9.81f;

  constexpr float DEG2RAD = 3.14159265f / 180.0f;
  gyro_rads[0] = (gx / 131.0f) * DEG2RAD;
  gyro_rads[1] = (gy / 131.0f) * DEG2RAD;
  gyro_rads[2] = (gz / 131.0f) * DEG2RAD;
  return true;
}

void updateComplementaryFilter(float dt) {
  float cross[3] = {
    gyro_rads[1] * g_est[2] - gyro_rads[2] * g_est[1],
    gyro_rads[2] * g_est[0] - gyro_rads[0] * g_est[2],
    gyro_rads[0] * g_est[1] - gyro_rads[1] * g_est[0]
  };
  float g_pred[3] = {
    g_est[0] - cross[0] * dt,
    g_est[1] - cross[1] * dt,
    g_est[2] - cross[2] * dt
  };
  float n_pred = sqrtf(g_pred[0]*g_pred[0] + g_pred[1]*g_pred[1] + g_pred[2]*g_pred[2]);
  if (n_pred > 1e-6f) {
    g_pred[0] /= n_pred; g_pred[1] /= n_pred; g_pred[2] /= n_pred;
  }

  float a_norm = sqrtf(accel_ms2[0]*accel_ms2[0] + accel_ms2[1]*accel_ms2[1] + accel_ms2[2]*accel_ms2[2]);
  if (a_norm > 6.5f && a_norm < 13.0f) {
    constexpr float alpha = 0.04f;
    for (int i = 0; i < 3; i++) {
      float g_acc = -accel_ms2[i] / a_norm;
      g_est[i] = (1.0f - alpha) * g_pred[i] + alpha * g_acc;
    }
    float n = sqrtf(g_est[0]*g_est[0] + g_est[1]*g_est[1] + g_est[2]*g_est[2]);
    if (n > 1e-6f) {
      g_est[0] /= n; g_est[1] /= n; g_est[2] /= n;
    }
  } else {
    for (int i = 0; i < 3; i++) g_est[i] = g_pred[i];
  }
}

void runPolicySelfTest() {
  float act_out[BB8_ACT_DIM];
  uint32_t t0 = micros();
  bb8_policy_step(BB8_TEST_OBS, act_out);
  uint32_t dt_us = micros() - t0;

  float max_err = 0.0f;
  for (int i = 0; i < BB8_ACT_DIM; ++i) {
    float e = fabsf(act_out[i] - BB8_TEST_ACT_REF[i]);
    if (e > max_err) max_err = e;
  }

  Serial.println("\n=== AUTO-TEST PPO v3 SUR ESP32-S3 ===");
  Serial.printf("  Sortie ESP32 : [%+.5f, %+.5f]\n", act_out[0], act_out[1]);
  Serial.printf("  Ref PyTorch  : [%+.5f, %+.5f]\n", BB8_TEST_ACT_REF[0], BB8_TEST_ACT_REF[1]);
  Serial.printf("  Ecart max    : %.2e  -> %s\n", max_err, max_err < 1e-4f ? "OK" : "ERREUR");
  Serial.printf("  Temps infer. : %lu us\n", (unsigned long)dt_us);
}

void handleSerialCommands() {
  while (Serial.available()) {
    char c = (char)tolower(Serial.read());
    if (c == 'm') {
      ppo_mode = !ppo_mode;
      v_err_int = 0.0f;
      yaw_err_int = 0.0f;
      action_filt[0] = 0.0f;
      action_filt[1] = 0.0f;
      Serial.printf(">>> Mode bascule : %s\n", ppo_mode ? "PPO v3" : "MANUEL");
    } else if (ppo_mode) {
      if (c == 'z') cmd_v = constrain(cmd_v + 0.25f, -0.80f, 1.48f);
      else if (c == 's') cmd_v = constrain(cmd_v - 0.25f, -0.80f, 1.48f);
      else if (c == 'q') cmd_yaw = constrain(cmd_yaw + 0.20f, -0.78f, 0.78f);
      else if (c == 'd') cmd_yaw = constrain(cmd_yaw - 0.20f, -0.78f, 0.78f);
      else if (c == 'x') { cmd_v = 0.0f; cmd_yaw = 0.0f; v_err_int = 0.0f; yaw_err_int = 0.0f; }
    } else {
      if (c == 'z') manual_pwm = constrain(manual_pwm + 0.25f, -1.0f, 1.0f);
      else if (c == 's') manual_pwm = constrain(manual_pwm - 0.25f, -1.0f, 1.0f);
      else if (c == 'q') manual_steer = constrain(manual_steer + 0.20f, -MAX_STEER_RAD, MAX_STEER_RAD);
      else if (c == 'd') manual_steer = constrain(manual_steer - 0.20f, -MAX_STEER_RAD, MAX_STEER_RAD);
      else if (c == 'x') { manual_pwm = 0.0f; manual_steer = 0.0f; }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_LED_BMS, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);
  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_ENC_A, INPUT_PULLUP);
  pinMode(PIN_ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), onEncoderA, CHANGE);

  analogReadResolution(12);
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
  bool imu_ok = initMPU6050();

  servoSteer.setPeriodHertz(50);
  servoHead.setPeriodHertz(50);
  servoSteer.attach(PIN_SERVO_STEER, 500, 2400);
  servoHead.attach(PIN_SERVO_HEAD, 500, 2400);
  servoSteer.write(90);
  servoHead.write(90);

  Serial.println("\n=== BB-8 FIRMWARE PPO v3 SIM-TO-REAL (ESP32-S3 @ 50 Hz) ===");
  Serial.printf("MPU-6050 I2C : %s\n", imu_ok ? "OK" : "ERREUR");
  runPolicySelfTest();
  Serial.println("Commandes : z/s (vitesse) | q/d (lacet) | x (stop) | m (PPO/Manuel)\n");
}

void loop() {
  static uint32_t lastTickUs = micros();
  static uint32_t printCounter = 0;

  uint32_t nowUs = micros();
  if (nowUs - lastTickUs < 20000) return; // 50 Hz strict
  lastTickUs += 20000;
  uint32_t tStartUs = micros();

  handleSerialCommands();

  // 1. Batterie LiPo 4S
  int rawAdc = analogRead(PIN_VBATT_ADC);
  v_batt = (rawAdc / 4095.0f) * 3.3f * DIVIDER_RATIO;
  bool lowBatt = (v_batt <= VBATT_CUTOFF);

  // 2. IMU + filtre complémentaire g_est
  readMPU6050();
  updateComplementaryFilter(DT_CTRL);

  // 3. Encodeur propulsion + vitesse absolue coque
  long ticks = encTicks;
  long dTicks = ticks - prevEncTicks;
  prevEncTicks = ticks;
  omega_enc = ((float)dTicks / TICKS_PER_REV) * (2.0f * 3.14159265f) / DT_CTRL;
  float omega_shell_est = omega_enc + gyro_rads[1];

  // 4. Intégrateurs d'erreur vitesse & lacet (identiques à bb8_env.py lignes 315-318)
  float v_shell_est = omega_shell_est * SPHERE_RADIUS;
  float yaw_est = -(gyro_rads[0] * g_est[0] + gyro_rads[1] * g_est[1] + gyro_rads[2] * g_est[2]);
  if (lowBatt) {
    v_err_int = 0.0f;
    yaw_err_int = 0.0f;
  } else {
    v_err_int   = bb8_clamp(0.96f * v_err_int   + (cmd_v   - v_shell_est) * DT_CTRL, -1.5f, 1.5f);
    yaw_err_int = bb8_clamp(0.96f * yaw_err_int + (cmd_yaw - yaw_est)     * DT_CTRL, -1.5f, 1.5f);
  }

  // 5. Inférence PPO v3 sur les 16 observations EXACTES de bb8_env.py
  float pwm_cmd = manual_pwm;
  float steer_cmd_rad = manual_steer;
  uint32_t inferUs = 0;

  if (ppo_mode) {
    float v_batt_norm = (v_batt - 14.8f) / 2.0f;

    float obs[BB8_OBS_DIM] = {
      g_est[0], g_est[1], g_est[2],
      gyro_rads[0], gyro_rads[1], gyro_rads[2],
      omega_enc,
      omega_shell_est,
      steer_pos,
      v_batt_norm,
      action_filt[0],
      action_filt[1],
      cmd_v,
      cmd_yaw,
      v_err_int,
      yaw_err_int
    };

    float raw_act[BB8_ACT_DIM];
    uint32_t tInf0 = micros();
    bb8_policy_step(obs, raw_act);
    inferUs = micros() - tInf0;

    // Filtrage passe-bas + limiteur de pente (identique à bb8_env.py lignes 244-254)
    float target_pwm   = action_filt[0] + 0.45f * (raw_act[0] - action_filt[0]);
    float target_steer = action_filt[1] + 0.35f * (raw_act[1] - action_filt[1]);

    float pwm_f   = action_filt[0] + bb8_clamp(target_pwm   - action_filt[0], -0.22f, 0.22f);
    float steer_f = action_filt[1] + bb8_clamp(target_steer - action_filt[1], -0.09f, 0.09f);

    action_filt[0] = bb8_clamp(pwm_f,   -1.0f, 1.0f);
    action_filt[1] = bb8_clamp(steer_f, -1.0f, 1.0f);

    pwm_cmd = action_filt[0];
    steer_cmd_rad = action_filt[1] * MAX_STEER_RAD;
    steer_pos += 0.25f * (steer_cmd_rad - steer_pos);
  }

  // 6. Sécurité BMS + sorties actionneurs
  float pwm_out = lowBatt ? 0.0f : pwm_cmd;
  digitalWrite(PIN_MOTOR_DIR, pwm_out < 0.0f ? HIGH : LOW);
  analogWrite(PIN_MOTOR_PWM, (int)(fabsf(pwm_out) * 255.0f));

  if (lowBatt) {
    digitalWrite(PIN_LED_BMS, (printCounter % 5 == 0) ? !digitalRead(PIN_LED_BMS) : digitalRead(PIN_LED_BMS));
  } else {
    digitalWrite(PIN_LED_BMS, LOW);
  }

  int degSteer = constrain(90 + (int)(steer_cmd_rad * 57.2958f), 55, 125);
  int degHead  = constrain(90 + (int)(steer_cmd_rad * 1.5f * 57.2958f), 35, 145);
  servoSteer.write(degSteer);
  servoHead.write(degHead);

  uint32_t execUs = micros() - tStartUs;

  // Télémétrie 5 Hz (compatible analyze_wokwi.py)
  if (++printCounter % 10 == 0) {
    Serial.printf(
      "V_batt=%5.2fV %s | g_est=(%+.2f,%+.2f,%+.2f) | gyroY=%+.2f | w_enc=%+5.1f | PWM=%+.2f | Steer=%3d deg | loop=%lu us (PPO=%lu us)\n",
      v_batt, lowBatt ? "[BMS CUTOFF!]" : "[OK]         ",
      g_est[0], g_est[1], g_est[2],
      gyro_rads[1], omega_enc, pwm_out, degSteer,
      (unsigned long)execUs, (unsigned long)inferUs
    );
  }
}