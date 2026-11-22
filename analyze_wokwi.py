import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("wokwi_log.txt")
OUT_PLOT = Path("wokwi_report.png")

LINE_RE = re.compile(
    r"V_batt=\s*([\d.]+)V\s*\[([^\]]+)\]\s*\|\s*"
    r"g_est=\(([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)\)\s*\|\s*"
    r"gyroY=([+-]?[\d.]+)\s*\|\s*"
    r"w_enc=\s*([+-]?[\d.]+)\s*\|\s*"
    r"PWM=([+-]?[\d.]+)\s*\|\s*"
    r"Steer=\s*([+-]?\d+)\s*deg\s*\|\s*"
    r"loop=(\d+)\s*us"
)


def parse_log(path: Path) -> dict:
    keys = ("t", "v_batt", "low_batt", "gx", "gy", "gz", "gyro_y", "w_enc", "pwm", "steer_deg", "loop_us")
    data = {k: [] for k in keys}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    t_idx = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Format texte par défaut
        m = LINE_RE.search(line)
        if m:
            data["t"].append(t_idx * 0.2)  # télémétrie à 5 Hz
            data["v_batt"].append(float(m.group(1)))
            data["low_batt"].append(1 if "CUTOFF" in m.group(2) else 0)
            data["gx"].append(float(m.group(3)))
            data["gy"].append(float(m.group(4)))
            data["gz"].append(float(m.group(5)))
            data["gyro_y"].append(float(m.group(6)))
            data["w_enc"].append(float(m.group(7)))
            data["pwm"].append(float(m.group(8)))
            data["steer_deg"].append(float(m.group(9)))
            data["loop_us"].append(float(m.group(10)))
            t_idx += 1
            continue

        # 2. Format CSV (si activé)
        parts = line.split(",")
        if len(parts) == 11 and parts[0].isdigit():
            data["t"].append(float(parts[0]) / 1000.0)
            data["v_batt"].append(float(parts[1]))
            data["low_batt"].append(int(parts[2]))
            data["gx"].append(float(parts[3]))
            data["gy"].append(float(parts[4]))
            data["gz"].append(float(parts[5]))
            data["gyro_y"].append(float(parts[6]))
            data["w_enc"].append(float(parts[7]))
            data["pwm"].append(float(parts[8]))
            data["steer_deg"].append(float(parts[9]))
            data["loop_us"].append(float(parts[10]))

    if not data["t"]:
        raise ValueError(f"Aucune ligne de télémétrie valide trouvée dans {path}.")

    return {k: np.array(v, dtype=float) for k, v in data.items()}


def main():
    d = parse_log(LOG_PATH)
    t = d["t"] - d["t"][0]

    # Vérifications clés
    loop_mean = float(np.mean(d["loop_us"]))
    loop_max = float(np.max(d["loop_us"]))
    cpu_load_pct = (loop_max / 20000.0) * 100.0

    g_norm = np.sqrt(d["gx"] ** 2 + d["gy"] ** 2 + d["gz"] ** 2)
    bms_mask = d["low_batt"] == 1
    bms_triggered = bool(np.any(bms_mask))
    bms_pwm_leak = float(np.max(np.abs(d["pwm"][bms_mask]))) if bms_triggered else 0.0

    print("\n=== BILAN SIMULATION ÉLECTRONIQUE WOKWI (ESP32-S3 @ 50 Hz) ===")
    print(f"  Échantillons analysés : {len(t)} ({t[-1]:.1f} s)")
    print(f"  Temps de boucle       : moy = {loop_mean:.0f} us | max = {loop_max:.0f} us ({cpu_load_pct:.1f} % du budget 20 ms)")
    print(f"  Tension LiPo 4S       : min = {np.min(d['v_batt']):.2f} V | max = {np.max(d['v_batt']):.2f} V")
    if bms_triggered:
        status = "OK (PWM forcé à 0.00)" if bms_pwm_leak < 1e-3 else f"ERREUR (fuite PWM = {bms_pwm_leak:.2f})"
        print(f"  Sécurité BMS <=12.6V  : déclenchée -> {status}")
    else:
        print("  Sécurité BMS <=12.6V  : non testée (descendez le potentiomètre sous 12.6 V)")
    print(f"  Norme gravité |g_est| : min = {np.min(g_norm):.3f} | max = {np.max(g_norm):.3f} (attendu ~1.000)")
    print(f"  Encodeur w_enc        : min = {np.min(d['w_enc']):+.1f} | max = {np.max(d['w_enc']):+.1f} rad/s")
    print(f"  Sorties actionneurs   : PWM [{np.min(d['pwm']):+.2f}, {np.max(d['pwm']):+.2f}] | Steer [{np.min(d['steer_deg']):.0f}°, {np.max(d['steer_deg']):.0f}°]")

    # Graphique 4 panneaux
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. Batterie & BMS
    ax = axes[0]
    ax.plot(t, d["v_batt"], color="#1f77b4", lw=2.0, label="V_batt (V)")
    ax.axhline(12.6, color="#d62728", ls="--", lw=1.2, label="Seuil coupure BMS (12.6 V)")
    ax.set_ylabel("Tension (V)")
    ax.set_title("Validation banc électronique BB-8 (Wokwi ESP32-S3)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 2. IMU (g_est & gyroY)
    ax = axes[1]
    ax.plot(t, d["gx"], label="g_est.x (tangage)", lw=1.6)
    ax.plot(t, d["gy"], label="g_est.y (roulis)", lw=1.6)
    ax.plot(t, d["gz"], label="g_est.z (verticale)", lw=1.4, ls="--")
    ax.set_ylabel("Gravité norm.")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 3. Actionneurs & Encodeur
    ax = axes[2]
    ax.plot(t, d["pwm"], color="#2ca02c", lw=2.0, label="PWM moteur")
    ax.set_ylabel("PWM (-1..1)", color="#2ca02c")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    ax_r = ax.twinx()
    ax_r.plot(t, d["steer_deg"], color="#ff7f0e", lw=1.6, ls="--", label="Servo pendule (°)")
    ax_r.set_ylabel("Pendule (°)", color="#ff7f0e")
    ax.legend(loc="upper left")
    ax_r.legend(loc="upper right")

    # 4. Budget temps réel 50 Hz
    ax = axes[3]
    ax.plot(t, d["loop_us"], color="#9467bd", lw=1.8, label="Durée boucle (µs)")
    ax.axhline(20000, color="#d62728", ls="--", lw=1.2, label="Limite 50 Hz (20 000 µs)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Temps (µs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=140)
    plt.close(fig)
    print(f"\nGraphique sauvegardé : {OUT_PLOT}")


if __name__ == "__main__":
    main()