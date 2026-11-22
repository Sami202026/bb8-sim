from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")  # Rendu hors-écran sûr sous mjpython sur macOS
import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bb8_env import BB8Env


def get_eval_command(t: float) -> tuple[float, float]:
    """Parcours d'évaluation cyclique de 24 s : (vitesse_cible m/s, taux_virage rad/s)."""
    cycle = t % 24.0
    if cycle < 4.5:
        return 1.2, 0.0       # Ligne droite avant
    elif cycle < 9.5:
        return 1.1, 0.55      # Virage à gauche
    elif cycle < 14.5:
        return 1.1, -0.55     # Virage à droite
    elif cycle < 18.0:
        return 0.0, 0.0       # Freinage et arrêt
    elif cycle < 21.5:
        return -0.6, 0.0      # Marche arrière
    else:
        return 0.0, 0.0       # Retour à l'arrêt


def save_ppo_telemetry(log: dict, path: str = "telemetry_ppo.png"):
    if len(log["t"]) < 10:
        return

    t = np.array(log["t"])
    v_rmse = float(np.sqrt(np.mean((np.array(log["v_true"]) - np.array(log["v_cmd"])) ** 2)))
    yaw_rmse = float(np.sqrt(np.mean((np.array(log["yaw_true"]) - np.array(log["yaw_cmd"])) ** 2)))
    gyro_rms = float(np.sqrt(np.mean(np.array(log["gyro_y_true"]) ** 2)))

    print("\n--- Bilan chiffré PPO ---")
    print(f"  RMSE Vitesse avant : {v_rmse:.3f} m/s")
    print(f"  RMSE Taux virage   : {yaw_rmse:.3f} rad/s")
    print(f"  RMS Tangage Gyro Y : {gyro_rms:.3f} rad/s")
    print(f"  Courant moyen/pic  : {np.mean(log['i_batt']):.2f} A / {np.max(log['i_batt']):.2f} A")

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    # 1. Batterie LiPo 4S
    ax1 = axes[0]
    ax1.plot(t, log["v_ocv"], "--", label="OCV (à vide)", color="#888888")
    ax1.plot(t, log["v_batt"], label="V_batt (en charge)", color="#1f77b4", lw=1.6)
    ax1.set_ylabel("Tension (V)")
    ax1.set_title("Télémétrie BB-8 piloté par PPO (Sim-to-Real : LiPo 4S, Moteurs DC, IMU, Latence)")
    ax1.grid(True, alpha=0.3)
    ax1_r = ax1.twinx()
    ax1_r.plot(t, log["i_batt"], label="I_batt (A)", color="#d62728", alpha=0.65)
    ax1_r.set_ylabel("Courant (A)", color="#d62728")
    ax1.legend(loc="upper left")
    ax1_r.legend(loc="upper right")

    # 2. Moteurs DC (Couple + PWM)
    ax2 = axes[1]
    ax2.plot(t, log["tau_net"], label="Couple net appliqué (N·m)", color="#ff7f0e", lw=1.4)
    ax2.set_ylabel("Couple (N·m)")
    ax2.grid(True, alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(t, log["pwm"], label="PWM (-1..+1)", color="#2ca02c", alpha=0.55)
    ax2_r.set_ylabel("PWM", color="#2ca02c")
    ax2_r.set_ylim(-1.15, 1.15)
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")

    # 3. IMU Gyroscope Y
    ax3 = axes[2]
    ax3.plot(t, log["gyro_y_meas"], label="Gyro Y mesuré (bruit + dérive)", color="#9467bd", alpha=0.55)
    ax3.plot(t, log["gyro_y_true"], label="Gyro Y réel", color="#111111", lw=1.3)
    ax3.set_ylabel("Gyro Y (rad/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # 4. Suivi vitesse linéaire
    ax4 = axes[3]
    ax4.plot(t, log["v_cmd"], "--", label="Vitesse cible (m/s)", color="#444444")
    ax4.plot(t, log["v_true"], label="Vitesse réelle (m/s)", color="#1f77b4", lw=1.6)
    ax4.set_ylabel("Vitesse (m/s)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    # 5. Suivi taux de virage (lacet)
    ax5 = axes[4]
    ax5.plot(t, log["yaw_cmd"], "--", label="Virage cible (rad/s)", color="#444444")
    ax5.plot(t, log["yaw_true"], label="Virage réel (rad/s)", color="#17becf", lw=1.6)
    ax5.set_ylabel("Lacet (rad/s)")
    ax5.set_xlabel("Temps (s)")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Graphique PPO sauvegardé : {path}")


def load_model_and_env():
    raw_env = BB8Env(xml_path="bb8_cad.xml", episode_seconds=120.0, domain_rand=False)
    venv = DummyVecEnv([lambda: raw_env])

    best_model_path = Path("models/best_model/best_model.zip")
    best_norm_path = Path("models/best_model/vec_normalize.pkl")

    if best_model_path.exists() and best_norm_path.exists():
        print(f"Chargement du meilleur modèle v2 : {best_model_path}")
        venv = VecNormalize.load(str(best_norm_path), venv)
        model = PPO.load(str(best_model_path), device="cpu")
    else:
        print("Chargement du modèle v1 : models/bb8_ppo.zip")
        venv = VecNormalize.load("models/vec_normalize.pkl", venv)
        model = PPO.load("models/bb8_ppo", device="cpu")

    venv.training = False
    venv.norm_reward = False
    return raw_env, venv, model


def main():
    raw_env, venv, model = load_model_and_env()
    obs = venv.reset()

    log = {k: [] for k in (
        "t", "v_batt", "v_ocv", "i_batt", "pwm", "tau_net",
        "gyro_y_true", "gyro_y_meas", "v_cmd", "v_true", "yaw_cmd", "yaw_true",
    )}
    last_print = 0.0

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        while viewer.is_running():
            t_start = time.perf_counter()

            v_cmd, yaw_cmd = get_eval_command(raw_env.data.time)
            raw_env.set_command(v_cmd, yaw_cmd)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            info = infos[0]

            viewer.cam.lookat[:] = raw_env.data.xpos[raw_env.shell_id]
            viewer.sync()

            log["t"].append(raw_env.data.time)
            log["v_batt"].append(info["v_batt"])
            log["v_ocv"].append(info["v_ocv"])
            log["i_batt"].append(info["i_batt"])
            log["pwm"].append(info["pwm"])
            log["tau_net"].append(info["tau_net"])
            log["gyro_y_true"].append(info["gyro_y_true"])
            log["gyro_y_meas"].append(info["gyro_y_meas"])
            log["v_cmd"].append(info["v_cmd"])
            log["v_true"].append(info["v_forward"])
            log["yaw_cmd"].append(info["yaw_cmd"])
            log["yaw_true"].append(info["yaw_rate"])

            if raw_env.data.time - last_print >= 0.5:
                print(
                    f"t={raw_env.data.time:5.1f}s | "
                    f"v={info['v_forward']:+4.2f}/{info['v_cmd']:+4.2f} m/s | "
                    f"yaw={info['yaw_rate']:+5.2f}/{info['yaw_cmd']:+5.2f} rad/s | "
                    f"V={info['v_batt']:5.2f}V | I={info['i_batt']:4.1f}A"
                )
                last_print = raw_env.data.time

            if dones[0]:
                break

            remaining = raw_env.dt_ctrl - (time.perf_counter() - t_start)
            if remaining > 0:
                time.sleep(remaining)

    save_ppo_telemetry(log)


if __name__ == "__main__":
    main()