import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bb8_env import BB8Env
from teleop_bb8 import TeleopController


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


def run_heading_hold_test(
    label: str,
    use_heading_hold: bool = True,
    boost: bool = False,
    duration_s: float = 12.0,
    warmup_s: float = 2.0,
    seed: int = 42,
):
    raw_env, venv, model = load_policy_and_env()
    driver = TeleopController()

    raw_env.set_command(0.0, 0.0)
    raw_env.reset(seed=seed)
    obs = venv.reset()

    n_steps = int(duration_s / raw_env.dt_ctrl)
    warmup_steps = int(warmup_s / raw_env.dt_ctrl)

    v_actual = 0.0
    yaw_actual = 0.0
    ref_pos = None
    ref_heading = None

    log = {k: [] for k in (
        "t", "x", "y", "heading_deg", "v", "yaw_rate", "cmd_yaw", "yaw_trim",
    )}

    for step in range(n_steps):
        vz = float(raw_env.data.qvel[2])
        gyro_y = float(raw_env.last_gyro_meas[1])
        pitch_sin = float(raw_env.g_est[0])
        R_frame = raw_env.data.xmat[raw_env.frame_id].reshape(3, 3)
        heading = math.atan2(R_frame[1, 0], R_frame[0, 0])

        cmd_v, cmd_yaw = driver.update(
            throttle=True,
            brake=False,
            left=False,
            right=False,
            boost=boost,
            handbrake=False,
            v_actual=v_actual,
            vz=vz,
            gyro_y=gyro_y,
            pitch_sin=pitch_sin,
            heading=heading,
            yaw_actual=yaw_actual,
            dt=raw_env.dt_ctrl,
        )

        if not use_heading_hold:
            cmd_yaw = 0.0

        raw_env.set_command(cmd_v, cmd_yaw)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]

        v_actual = float(info["v_forward"])
        yaw_actual = float(info["yaw_rate"])

        pos = raw_env.data.xpos[raw_env.shell_id].copy()
        R_frame = raw_env.data.xmat[raw_env.frame_id].reshape(3, 3)
        heading_post = math.atan2(R_frame[1, 0], R_frame[0, 0])

        if step == warmup_steps:
            ref_pos = pos[:2].copy()
            ref_heading = heading_post

        log["t"].append(raw_env.data.time)
        log["x"].append(float(pos[0]))
        log["y"].append(float(pos[1]))
        log["heading_deg"].append(math.degrees(heading_post))
        log["v"].append(v_actual)
        log["yaw_rate"].append(yaw_actual)
        log["cmd_yaw"].append(cmd_yaw)
        log["yaw_trim"].append(float(driver.yaw_trim))

        if dones[0]:
            break

    # Analyse sur le régime établi (après warmup_s)
    sl = slice(warmup_steps, len(log["t"]))
    v_mean = float(np.mean(log["v"][sl]))
    yaw_arr = np.array(log["yaw_rate"][sl])
    yaw_mean = float(np.mean(yaw_arr))
    yaw_std = float(np.std(yaw_arr))

    dx_w = log["x"][-1] - float(ref_pos[0])
    dy_w = log["y"][-1] - float(ref_pos[1])
    c, s = math.cos(ref_heading), math.sin(ref_heading)
    dist_fwd = c * dx_w + s * dy_w
    lat_drift = -s * dx_w + c * dy_w  # > 0 : gauche, < 0 : droite

    # Écart latéral maximal par rapport à l'axe de référence sur toute la ligne droite
    xs = np.array(log["x"][sl]) - float(ref_pos[0])
    ys = np.array(log["y"][sl]) - float(ref_pos[1])
    lat_traj = -s * xs + c * ys
    max_abs_lat = float(np.max(np.abs(lat_traj)))

    headings_unwrapped = np.unwrap(np.radians(log["heading_deg"][sl]))
    delta_heading_deg = float(np.degrees(headings_unwrapped[-1] - headings_unwrapped[0]))
    radius_m = abs(v_mean / yaw_mean) if abs(yaw_mean) > 1e-4 else float("inf")
    drift_cm_per_m = (lat_drift / max(dist_fwd, 1e-6)) * 100.0
    direction = "GAUCHE" if yaw_mean > 0 else "DROITE"

    print(f"\n=== {label} (régime établi {duration_s - warmup_s:.0f} s) ===")
    print(f"  Vitesse moyenne         : {v_mean:.3f} m/s (distance : {dist_fwd:.2f} m)")
    print(f"  Taux de lacet moyen     : {yaw_mean:+.4f} rad/s  ({math.degrees(yaw_mean):+.2f} °/s)  ± {yaw_std:.3f}  [{direction}]")
    print(f"  Dérive de cap cumulée   : {delta_heading_deg:+.2f} °  ({delta_heading_deg / max(dist_fwd, 1e-6):+.2f} °/m)")
    print(f"  Écart latéral final     : {lat_drift:+.3f} m  ({drift_cm_per_m:+.2f} cm/m)  | max : {max_abs_lat:.3f} m")
    print(f"  Rayon de courbure éq.   : {radius_m:.1f} m  (infini = parfaitement droit)")
    if use_heading_hold:
        print(f"  Correction maintien de cap active : cmd_yaw moy = {np.mean(log['cmd_yaw'][sl]):+.4f} rad/s | yaw_trim final = {log['yaw_trim'][-1]:+.4f}")

    return log


def save_comparison_plot(log_raw: dict, log_cruise: dict, log_boost: dict, path: str = "heading_hold_report.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1. Trajectoire vue de dessus (X, Y)
    ax1 = axes[0]
    ax1.plot(log_raw["x"], log_raw["y"], "--", label="Sans maintien de cap (PPO pur)", lw=1.8, color="#d62728")
    ax1.plot(log_cruise["x"], log_cruise["y"], label="Croisière (1.25 m/s) + Maintien de cap", lw=2.2, color="#1f77b4")
    ax1.plot(log_boost["x"], log_boost["y"], label="Haute vitesse (1.48 m/s) + Maintien de cap", lw=2.0, color="#2ca02c")
    ax1.axhline(0.0, color="#555555", ls=":", lw=1.0, label="Ligne droite idéale (y = 0)")
    ax1.set_title("Validation du maintien de cap avec TeleopController")
    ax1.set_xlabel("X avant (m)")
    ax1.set_ylabel("Y latéral (m)  [+ = gauche]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Cap (heading) dans le temps
    ax2 = axes[1]
    ax2.plot(log_raw["t"], log_raw["heading_deg"], "--", label="Cap sans maintien (°)", color="#d62728")
    ax2.plot(log_cruise["t"], log_cruise["heading_deg"], label="Cap Croisière (°)", color="#1f77b4", lw=1.8)
    ax2.plot(log_boost["t"], log_boost["heading_deg"], label="Cap Haute vitesse (°)", color="#2ca02c", lw=1.6)
    ax2.axhline(0.0, color="#555555", ls=":", lw=1.0)
    ax2.set_ylabel("Cap (°)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Action du maintien de cap (cmd_yaw et yaw_trim)
    ax3 = axes[2]
    ax3.plot(log_cruise["t"], log_cruise["cmd_yaw"], label="cmd_yaw Croisière (rad/s)", color="#1f77b4")
    ax3.plot(log_cruise["t"], log_cruise["yaw_trim"], "--", label="yaw_trim Croisière", color="#17becf")
    ax3.plot(log_boost["t"], log_boost["cmd_yaw"], label="cmd_yaw Boost (rad/s)", color="#2ca02c", alpha=0.75)
    ax3.axhline(0.0, color="#555555", ls=":", lw=1.0)
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Correction lacet (rad/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nGraphique comparatif sauvegardé : {path}")


def main():
    log_raw = run_heading_hold_test(
        "TEST 1 : Référence SANS maintien de cap (Croisière)", use_heading_hold=False, boost=False
    )
    log_cruise = run_heading_hold_test(
        "TEST 2 : Croisière AVEC maintien de cap", use_heading_hold=True, boost=False
    )
    log_boost = run_heading_hold_test(
        "TEST 3 : Haute vitesse AVEC maintien de cap", use_heading_hold=True, boost=True
    )
    save_comparison_plot(log_raw, log_cruise, log_boost)


if __name__ == "__main__":
    main()