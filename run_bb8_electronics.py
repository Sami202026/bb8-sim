import math
import time
import matplotlib
matplotlib.use("Agg")  # Rendu hors-écran sûr sous mjpython sur macOS
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from bb8_electronics import (
    CommandLatencyBuffer,
    DualDCGearMotor,
    LiPo4SBattery,
    NoisyIMU,
    QuadratureEncoder,
)

SPHERE_RADIUS = 0.25
CONTROL_HZ = 100.0        # Fréquence du microcontrôleur embarqué (10 ms)
LATENCY_MS = 12.0         # Latence totale capteurs -> PWM (12 ms = 6 pas physiques)
MAX_ACCEL = 0.65          # Rampe d'accélération / freinage max (m/s²)
MAX_STEER_RATE = 1.2      # Rampe de braquage du pendule (rad/s)
MAX_HEAD_RATE = 3.0       # Rampe de rotation de la tête (rad/s)


def get_demo_command(t: float) -> tuple[float, float, float]:
    """Parcours de test cyclique : (vitesse_cible m/s, braquage rad, tête rad)."""
    cycle = t % 18.0
    head_yaw = 0.6 * math.sin(1.2 * t)

    if cycle < 4.0:
        return 1.4, 0.0, head_yaw          # Ligne droite soutenue
    elif cycle < 9.0:
        return 1.2, 0.42, 0.5              # Virage à gauche
    elif cycle < 14.0:
        return 1.2, -0.42, -0.5            # Virage à droite
    else:
        return 0.0, 0.0, 0.0               # Freinage et stabilisation


def save_telemetry_plot(log: dict, path: str = "telemetry_electronics.png"):
    if len(log["t"]) < 10:
        return

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = log["t"]

    # 1. Batterie LiPo 4S
    ax1 = axes[0]
    ax1.plot(t, log["v_ocv"], "--", label="OCV (à vide)", color="#888888")
    ax1.plot(t, log["v_batt"], label="V_batt (en charge)", color="#1f77b4", lw=1.8)
    ax1.set_ylabel("Tension (V)")
    ax1.set_title("Télémétrie électronique BB-8 (LiPo 4S, Moteurs DC, IMU, Latence 12 ms)")
    ax1.grid(True, alpha=0.3)
    ax1_r = ax1.twinx()
    ax1_r.plot(t, log["i_batt"], label="I_batt (A)", color="#d62728", alpha=0.65)
    ax1_r.set_ylabel("Courant (A)", color="#d62728")
    ax1.legend(loc="upper left")
    ax1_r.legend(loc="upper right")

    # 2. Moteurs DC (PWM + Couple net)
    ax2 = axes[1]
    ax2.plot(t, log["tau_net"], label="Couple net appliqué (N·m)", color="#ff7f0e", lw=1.5)
    ax2.set_ylabel("Couple (N·m)")
    ax2.grid(True, alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(t, log["pwm"], label="PWM (-1..+1)", color="#2ca02c", alpha=0.55)
    ax2_r.set_ylabel("PWM", color="#2ca02c")
    ax2_r.set_ylim(-1.15, 1.15)
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")

    # 3. IMU Gyroscope Y (vrai vs bruité + dérive)
    ax3 = axes[2]
    ax3.plot(t, log["gyro_y_meas"], label="Gyro Y mesuré (bruit + dérive)", color="#9467bd", alpha=0.55)
    ax3.plot(t, log["gyro_y_true"], label="Gyro Y réel (MuJoCo)", color="#111111", lw=1.3)
    ax3.set_ylabel("Gyro Y (rad/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # 4. Vitesse linéaire
    ax4 = axes[3]
    ax4.plot(t, log["v_target"], "--", label="Vitesse cible (rampe)", color="#444444")
    ax4.plot(t, log["v_enc"], label="Vitesse estimée (enc 12800 CPR + gyro)", color="#17becf", alpha=0.65)
    ax4.plot(t, log["v_true"], label="Vitesse réelle coque", color="#1f77b4", lw=1.5)
    ax4.set_ylabel("Vitesse (m/s)")
    ax4.set_xlabel("Temps (s)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nGraphique de télémétrie sauvegardé : {path}")


def main():
    model = mujoco.MjModel.from_xml_path("bb8_cad.xml")
    data = mujoco.MjData(model)

    dt_sim = model.opt.timestep
    dt_ctrl = 1.0 / CONTROL_HZ
    ctrl_decimation = max(1, round(dt_ctrl / dt_sim))
    delay_steps = round((LATENCY_MS * 1e-3) / dt_sim)

    shell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shell")
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_servo")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_servo")

    # Initialisation de la chaîne électronique
    # Astuce : passez initial_soc=0.15 ou soc_speedup=80.0 pour observer l'effet batterie faible
    battery = LiPo4SBattery(capacity_ah=5.0, r_internal=0.08, initial_soc=1.0, soc_speedup=1.0)
    motors = DualDCGearMotor()
    imu = NoisyIMU(seed=42)
    # Encodeur monté côté moteur (512 CPR * réducteur 25 = 12 800 CPR en sortie)
    encoder = QuadratureEncoder(counts_per_rev=512 * 25)
    latency_buf = CommandLatencyBuffer(delay_steps=delay_steps)

    # État interne du firmware MCU (PID + filtres + rampes)
    integral_err = 0.0
    gyro_y_filt = 0.0
    omega_shell_filt = 0.0
    v_cmd_ramped = 0.0
    steer_cmd_ramped = 0.0
    head_cmd_ramped = 0.0
    mcu_cmd = (0.0, 0.0, 0.0)
    last_v_enc = 0.0
    last_gyro_y_meas = 0.0

    log = {k: [] for k in (
        "t", "v_batt", "v_ocv", "i_batt", "pwm", "tau_net",
        "gyro_y_true", "gyro_y_meas", "v_target", "v_true", "v_enc",
    )}

    step_idx = 0
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        while viewer.is_running():
            step_start = time.perf_counter()

            # --- 1. Capteurs réels -> Capteurs bruités (cadence physique) ---
            true_accel = data.sensor("imu_accel").data.copy()
            true_gyro = data.sensor("imu_gyro").data.copy()
            _, gyro_meas = imu.sample(true_accel, true_gyro, dt_sim)
            last_gyro_y_meas = float(gyro_meas[1])

            # --- 2. Boucle de contrôle embarquée MCU à 100 Hz ---
            if step_idx % ctrl_decimation == 0:
                q_drive = float(data.joint("drive_joint").qpos[0])
                omega_enc = encoder.measure_velocity(q_drive, dt_ctrl)

                # Filtrage passe-bas plus réactif du gyro (réduit le retard de phase)
                gyro_y_filt = 0.65 * last_gyro_y_meas + 0.35 * gyro_y_filt

                # Fusion encodeur + gyro -> vitesse absolue de la sphère (découple le pendule)
                omega_shell_est = omega_enc + gyro_y_filt
                omega_shell_filt = 0.45 * omega_shell_est + 0.55 * omega_shell_filt
                last_v_enc = omega_shell_filt * SPHERE_RADIUS

                # Rampe d'accélération / freinage sur la consigne de vitesse
                raw_target_speed, target_steer, target_head = get_demo_command(data.time)
                max_dv = MAX_ACCEL * dt_ctrl
                v_cmd_ramped += float(np.clip(raw_target_speed - v_cmd_ramped, -max_dv, max_dv))
                target_omega = v_cmd_ramped / SPHERE_RADIUS

                # Rampes de direction et de tête (supprime les pics en aiguilles aux transitions)
                max_ds = MAX_STEER_RATE * dt_ctrl
                steer_cmd_ramped += float(np.clip(target_steer - steer_cmd_ramped, -max_ds, max_ds))
                max_dh = MAX_HEAD_RATE * dt_ctrl
                head_cmd_ramped += float(np.clip(target_head - head_cmd_ramped, -max_dh, max_dh))

                # Compensation FCEM (feedforward) + régulateur PI vitesse absolue + amortissement pendule
                err = target_omega - omega_shell_filt
                integral_err = float(np.clip(integral_err + err * dt_ctrl, -2.5, 2.5))
                pwm_ff = (motors.ke * motors.gear_ratio * target_omega) / max(battery.v_terminal, 11.0)

                pwm_cmd = pwm_ff + 0.18 * err + 0.03 * integral_err + 0.34 * gyro_y_filt
                pwm_cmd = float(np.clip(pwm_cmd, -1.0, 1.0))

                if battery.low_voltage_cutoff:
                    pwm_cmd = 0.0  # Sécurité BMS coupure basse tension

                mcu_cmd = (pwm_cmd, steer_cmd_ramped, head_cmd_ramped)

            # --- 3. Latence de commande (buffer FIFO) ---
            pwm_delayed, steer_delayed, head_delayed = latency_buf.push_and_pop(mcu_cmd)

            # --- 4. Modèle électromécanique moteur DC + Batterie LiPo 4S ---
            omega_true = float(data.sensor("drive_encoder_vel").data[0])
            tau_net, i_motor, i_drive_batt = motors.compute(
                pwm=pwm_delayed,
                omega_out=omega_true,
                v_batt=battery.v_terminal,
            )

            # Courant estimé des servos de direction et tête (proportionnel au couple)
            tau_steer = abs(float(data.actuator_force[steer_id]))
            tau_head = abs(float(data.actuator_force[head_id]))
            i_servos = 0.12 * tau_steer + 0.08 * tau_head

            v_batt = battery.step(i_drive_batt + i_servos, dt_sim)

            # --- 5. Application dans MuJoCo ---
            data.ctrl[drive_id] = float(np.clip(tau_net, -5.5, 5.5))
            data.ctrl[steer_id] = steer_delayed
            data.ctrl[head_id] = head_delayed

            mujoco.mj_step(model, data)

            viewer.cam.lookat[:] = data.xpos[shell_id]
            viewer.sync()

            # Vitesse linéaire réelle de la coque au sol (découplée du tangage interne)
            v_shell_true = (omega_true + float(true_gyro[1])) * SPHERE_RADIUS

            # Enregistrement télémétrie à 50 Hz (1 pas sur 10)
            if step_idx % 10 == 0:
                log["t"].append(data.time)
                log["v_batt"].append(v_batt)
                log["v_ocv"].append(battery.open_circuit_voltage())
                log["i_batt"].append(battery.last_current_a)
                log["pwm"].append(pwm_delayed)
                log["tau_net"].append(tau_net)
                log["gyro_y_true"].append(float(true_gyro[1]))
                log["gyro_y_meas"].append(last_gyro_y_meas)
                log["v_target"].append(v_cmd_ramped)
                log["v_true"].append(v_shell_true)
                log["v_enc"].append(last_v_enc)

            # Affichage terminal toutes les 0.5 s
            if data.time - last_print >= 0.5:
                print(
                    f"t={data.time:5.1f}s | V={v_batt:5.2f}V (SoC {battery.soc*100:5.1f}%) | "
                    f"I={battery.last_current_a:4.1f}A | PWM={pwm_delayed:+5.2f} | "
                    f"tau={tau_net:+5.2f} N·m | v={v_shell_true:4.2f} m/s"
                )
                last_print = data.time

            step_idx += 1
            remaining = dt_sim - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    save_telemetry_plot(log)


if __name__ == "__main__":
    main()