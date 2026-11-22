import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bb8_env import BB8Env


def load_policy_and_env():
    raw_env = BB8Env(xml_path="bb8_cad.xml", episode_seconds=15.0, domain_rand=False)
    venv = DummyVecEnv([lambda: raw_env])

    best_model = Path("models/best_model/best_model.zip")
    best_norm = Path("models/best_model/vec_normalize.pkl")

    if best_model.exists() and best_norm.exists():
        venv = VecNormalize.load(str(best_norm), venv)
        model = PPO.load(str(best_model), device="cpu")
    else:
        venv = VecNormalize.load("models/vec_normalize_last.pkl", venv)
        model = PPO.load("models/bb8_ppo_last", device="cpu")

    venv.training = False
    venv.norm_reward = False
    return raw_env, venv, model


def run_straight_test(
    label: str,
    v_target: float = 1.2,
    duration_s: float = 12.0,
    warmup_s: float = 2.0,
    zero_imu_bias: bool = False,
    seed: int = 42,
):
    raw_env, venv, model = load_policy_and_env()
    raw_env.set_command(v_target, 0.0)
    raw_env.reset(seed=seed)
    obs = venv.reset()

    if zero_imu_bias:
        raw_env.imu.gyro_bias[:] = 0.0
        raw_env.imu.accel_bias[:] = 0.0
        raw_env.imu.gyro_bias_drift = 0.0
        raw_env.imu.gyro_noise_std = 0.0

    gyro_z_bias_init = float(raw_env.imu.gyro_bias[2])
    n_steps = int(duration_s / raw_env.dt_ctrl)
    warmup_steps = int(warmup_s / raw_env.dt_ctrl)

    log = {k: [] for k in (
        "t", "x", "y", "heading_deg", "v", "yaw_rate",
        "steer_cmd_deg", "steer_pos_deg", "roll_deg", "yaw_err_int",
    )}

    ref_pos = None
    ref_heading = None

    for step in range(n_steps):
        raw_env.set_command(v_target, 0.0)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]

        pos = raw_env.data.xpos[raw_env.shell_id].copy()
        R_frame = raw_env.data.xmat[raw_env.frame_id].reshape(3, 3)
        heading = math.atan2(R_frame[1, 0], R_frame[0, 0])
        g_body_true, _, _ = raw_env._get_frame_kinematics()

        if step == warmup_steps:
            ref_pos = pos[:2].copy()
            ref_heading = heading

        log["t"].append(raw_env.data.time)
        log["x"].append(float(pos[0]))
        log["y"].append(float(pos[1]))
        log["heading_deg"].append(math.degrees(heading))
        log["v"].append(float(info["v_forward"]))
        log["yaw_rate"].append(float(info["yaw_rate"]))
        log["steer_cmd_deg"].append(math.degrees(float(raw_env.action_filt[1]) * 0.50))
        log["steer_pos_deg"].append(math.degrees(float(raw_env.data.sensor("steer_encoder_pos").data[0])))
        log["roll_deg"].append(math.degrees(math.asin(np.clip(float(g_body_true[1]), -1.0, 1.0))))
        log["yaw_err_int"].append(float(raw_env.yaw_err_int))

        if dones[0]:
            break

    # Analyse en régime établi (après warmup_s)
    sl = slice(warmup_steps, len(log["t"]))
    v_mean = float(np.mean(log["v"][sl]))
    yaw_arr = np.array(log["yaw_rate"][sl])
    yaw_mean = float(np.mean(yaw_arr))
    yaw_std = float(np.std(yaw_arr))

    # Projection de la position finale dans le repère du cap au début du régime établi
    dx_world = log["x"][-1] - float(ref_pos[0])
    dy_world = log["y"][-1] - float(ref_pos[1])
    c, s = math.cos(ref_heading), math.sin(ref_heading)
    dist_fwd = c * dx_world + s * dy_world
    lat_drift = -s * dx_world + c * dy_world  # > 0 : dérive à gauche, < 0 : à droite

    headings_unwrapped = np.unwrap(np.radians(log["heading_deg"][sl]))
    delta_heading_deg = float(np.degrees(headings_unwrapped[-1] - headings_unwrapped[0]))
    radius_m = abs(v_mean / yaw_mean) if abs(yaw_mean) > 1e-4 else float("inf")
    drift_cm_per_m = (lat_drift / max(dist_fwd, 1e-6)) * 100.0
    direction = "GAUCHE" if yaw_mean > 0 else "DROITE"

    print(f"\n=== {label} (v_cmd={v_target:.2f} m/s, régime établi {duration_s - warmup_s:.0f} s) ===")
    print(f"  Vitesse moyenne         : {v_mean:.3f} m/s (distance : {dist_fwd:.2f} m)")
    print(f"  Taux de lacet moyen     : {yaw_mean:+.4f} rad/s  ({math.degrees(yaw_mean):+.2f} °/s)  ± {yaw_std:.3f}  [{direction}]")
    print(f"  Dérive de cap cumulée   : {delta_heading_deg:+.2f} °  ({delta_heading_deg / max(dist_fwd, 1e-6):+.2f} °/m)")
    print(f"  Écart latéral total     : {lat_drift:+.3f} m  ({drift_cm_per_m:+.2f} cm par mètre parcouru)")
    print(f"  Rayon de courbure éq.   : {radius_m:.1f} m  (infini = ligne parfaitement droite)")
    print(f"  Pendule commandé (PPO)  : {np.mean(log['steer_cmd_deg'][sl]):+.2f} °  (mesuré : {np.mean(log['steer_pos_deg'][sl]):+.2f} °)")
    print(f"  Roulis moyen châssis    : {np.mean(log['roll_deg'][sl]):+.2f} °")
    print(f"  Intégrateur yaw_err_int : {np.mean(log['yaw_err_int'][sl]):+.4f}")
    print(f"  Biais initial Gyro Z    : {gyro_z_bias_init:+.4f} rad/s ({math.degrees(gyro_z_bias_init):+.2f} °/s)")

    return log


def save_drift_plot(log_real: dict, log_ideal: dict, path: str = "drift_report.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 1. Trajectoire vue de dessus (X, Y)
    ax1 = axes[0]
    ax1.plot(log_real["x"], log_real["y"], label="IMU réelle (bruit + biais)", lw=2.0, color="#1f77b4")
    ax1.plot(log_ideal["x"], log_ideal["y"], "--", label="IMU parfaite (biais = 0)", lw=1.8, color="#ff7f0e")
    ax1.axhline(0.0, color="#666666", ls=":", lw=1.0, label="Ligne droite idéale (y = 0)")
    ax1.set_title("Diagnostic de drift en ligne droite (v_cmd = 1.2 m/s, yaw_cmd = 0)")
    ax1.set_xlabel("X avant (m)")
    ax1.set_ylabel("Y latéral (m)  [+ = gauche]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Taux de lacet et braquage PPO
    ax2 = axes[1]
    ax2.plot(log_real["t"], log_real["yaw_rate"], label="Lacet réel IMU normale (rad/s)", color="#1f77b4")
    ax2.plot(log_ideal["t"], log_ideal["yaw_rate"], "--", label="Lacet réel IMU parfaite (rad/s)", color="#ff7f0e")
    ax2.axhline(0.0, color="#444444", ls=":", lw=1.0)
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Lacet (rad/s)")
    ax2.grid(True, alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(log_ideal["t"], log_ideal["steer_cmd_deg"], alpha=0.55, color="#2ca02c", label="Pendule PPO (°)")
    ax2_r.set_ylabel("Pendule PPO (°)", color="#2ca02c")
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nGraphique sauvegardé : {path}")


def main():
    log_real = run_straight_test("TEST 1 : IMU réelle", v_target=1.2, zero_imu_bias=False, seed=42)
    log_ideal = run_straight_test("TEST 2 : IMU parfaite (biais=0)", v_target=1.2, zero_imu_bias=True, seed=42)
    run_straight_test("TEST 3 : Vitesse modérée (IMU parfaite)", v_target=0.8, zero_imu_bias=True, seed=42)
    save_drift_plot(log_real, log_ideal)


if __name__ == "__main__":
    main()