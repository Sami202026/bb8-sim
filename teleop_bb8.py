import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import glfw
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bb8_env import BB8Env

# ==============================================================================
# PARAMÈTRES TÉLÉOPÉRATION + ASSISTANCES HDC / ESP
# ==============================================================================
V_CRUISE = 1.25           # Vitesse max normale (m/s)
V_BOOST = 1.48            # Vitesse max avec Boost Shift (m/s)
V_REVERSE = -0.80         # Vitesse max marche arrière (m/s)
V_DOWNHILL_SAFE = 1.00    # Plafond de vitesse sécurisé en descente (HDC)

ACCEL_RATE = 2.15         # Accélération vive (m/s²)
BOOST_ACCEL = 2.70        # Accélération Boost (m/s²)
COAST_DECEL = 0.85        # Frein moteur roue libre (m/s²)
BRAKE_DECEL = 3.10        # Freinage appuyé S / ↓ (m/s²)
HANDBRAKE_DECEL = 3.80    # Frein à main Espace (m/s²)
REV_ACCEL = 1.35          # Accélération marche arrière (m/s²)
REV_HOLD_DELAY = 0.16     # Transition courte vers marche arrière (s)

STEER_IN_RATE = 4.8       # Braquage rapide au maintien (unités/s)
STEER_RETURN_RATE = 6.8   # Retour au centre franc au relâchement (unités/s)
YAW_RAMP = 4.2            # Réactivité de virage (rad/s²)

CAM_PRESETS = [
    ("SUIVI PROCHE", 2.7, -17.0),
    ("SUIVI LOIN", 4.4, -24.0),
    ("LIBRE (SOURIS)", 5.0, -32.0),
]


def build_arena_xml(base_xml: str = "bb8_cad.xml", out_xml: str = "bb8_arena.xml") -> str:
    """Génère le parcours d'obstacles depuis bb8_cad.xml sans modifier la CAO."""
    tree = ET.parse(base_xml)
    worldbody = tree.getroot().find("worldbody")

    obstacles = [
        # Slalom central (+X)
        ("slalom_1", "cylinder", "3.2 0.8 0.45", "0 0 0", "0.16 0.45", "0.95 0.45 0.10 1"),
        ("slalom_2", "cylinder", "6.2 -0.8 0.45", "0 0 0", "0.16 0.45", "0.95 0.45 0.10 1"),
        ("slalom_3", "cylinder", "9.2 0.8 0.45", "0 0 0", "0.16 0.45", "0.95 0.45 0.10 1"),
        ("slalom_4", "cylinder", "11.0 -0.8 0.45", "0 0 0", "0.16 0.45", "0.95 0.45 0.10 1"),
        # Pont à pente douce (6°) à gauche (y = +4.5 m)
        ("ramp_up", "box", "3.5 4.5 0.1071", "0 -6 0", "1.5 1.2 0.05", "0.32 0.52 0.72 1"),
        ("ramp_top", "box", "6.2 4.5 0.2636", "0 0 0", "1.25 1.2 0.05", "0.26 0.44 0.62 1"),
        ("ramp_down", "box", "8.9 4.5 0.1071", "0 6 0", "1.5 1.2 0.05", "0.32 0.52 0.72 1"),
        # Dos d'âne doux à droite (y = -4.0 m)
        ("bump_1", "cylinder", "4.0 -4.0 -0.085", "90 0 0", "0.12 1.2", "0.85 0.70 0.15 1"),
        ("bump_2", "cylinder", "6.5 -4.0 -0.085", "90 0 0", "0.12 1.2", "0.85 0.70 0.15 1"),
        # Murets d'enceinte (24 m x 24 m)
        ("wall_n", "box", "12.0 0 0.22", "0 0 0", "0.15 12.15 0.22", "0.45 0.47 0.50 1"),
        ("wall_s", "box", "-12.0 0 0.22", "0 0 0", "0.15 12.15 0.22", "0.45 0.47 0.50 1"),
        ("wall_e", "box", "0 12.0 0.22", "0 0 0", "12.15 0.15 0.22", "0.45 0.47 0.50 1"),
        ("wall_w", "box", "0 -12.0 0.22", "0 0 0", "12.15 0.15 0.22", "0.45 0.47 0.50 1"),
    ]

    for name, gtype, pos, euler, size, rgba in obstacles:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": gtype,
                "pos": pos,
                "euler": euler,
                "size": size,
                "rgba": rgba,
                "friction": "1.0 0.02 0.01",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    tree.write(out_xml, encoding="utf-8")
    return out_xml


def load_policy_and_arena(arena_xml: str):
    raw_env = BB8Env(xml_path=arena_xml, episode_seconds=3600.0, domain_rand=False)
    venv = DummyVecEnv([lambda: raw_env])

    best_model = Path("models/best_model/best_model.zip")
    best_norm = Path("models/best_model/vec_normalize.pkl")

    if best_model.exists() and best_norm.exists():
        venv = VecNormalize.load(str(best_norm), venv)
        model = PPO.load(str(best_model), device="cpu")
        print(f"Modèle chargé : {best_model}")
    else:
        venv = VecNormalize.load("models/vec_normalize_last.pkl", venv)
        model = PPO.load("models/bb8_ppo_last", device="cpu")
        print("Modèle chargé : models/bb8_ppo_last.zip")

    venv.training = False
    venv.norm_reward = False
    return raw_env, venv, model


class TeleopController:
    """Contrôleur de téléopération avec maintien de cap, contrôle de descente (HDC) et stabilisation anti-tangage (ESP)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cmd_v = 0.0
        self.cmd_yaw = 0.0
        self.steer_axis = 0.0
        self.brake_hold_s = 0.0
        self.hdc_active = False
        self.esp_active = False
        self.hold_heading = None
        self.yaw_trim = 0.0

    def update(
        self,
        throttle: bool,
        brake: bool,
        left: bool,
        right: bool,
        boost: bool,
        handbrake: bool,
        v_actual: float,
        vz: float,
        gyro_y: float,
        pitch_sin: float,
        heading: float,
        yaw_actual: float,
        dt: float,
    ) -> tuple[float, float]:
        # --- 1. Gestion longitudinale vive ---
        if handbrake:
            self.brake_hold_s = 0.0
            step = HANDBRAKE_DECEL * dt
            self.cmd_v = 0.0 if abs(self.cmd_v) <= step else self.cmd_v - math.copysign(step, self.cmd_v)
        elif throttle and not brake:
            self.brake_hold_s = 0.0
            v_max = V_BOOST if boost else V_CRUISE
            rate = BRAKE_DECEL if self.cmd_v < 0.0 else (BOOST_ACCEL if boost else ACCEL_RATE)
            self.cmd_v = min(v_max, self.cmd_v + rate * dt)
        elif brake and not throttle:
            if self.cmd_v > 0.02 or v_actual > 0.14:
                self.cmd_v = max(0.0, self.cmd_v - BRAKE_DECEL * dt)
                self.brake_hold_s = 0.0
            else:
                self.brake_hold_s += dt
                if self.brake_hold_s >= REV_HOLD_DELAY:
                    self.cmd_v = max(V_REVERSE, self.cmd_v - REV_ACCEL * dt)
                else:
                    self.cmd_v = 0.0
        else:
            self.brake_hold_s = 0.0
            step = COAST_DECEL * dt
            self.cmd_v = 0.0 if abs(self.cmd_v) <= step else self.cmd_v - math.copysign(step, self.cmd_v)

        if not boost and self.cmd_v > V_CRUISE:
            self.cmd_v = max(V_CRUISE, self.cmd_v - COAST_DECEL * dt)

        # --- 2. Contrôle de descente (HDC) : empêche l'emballement en pente descendante ---
        descending = vz < -0.035
        cmd_v_out = self.cmd_v
        self.hdc_active = False

        if descending and abs(v_actual) > 0.45:
            # Maintient la vitesse interne sous le plafond sûr en descente
            if abs(self.cmd_v) > V_DOWNHILL_SAFE:
                self.cmd_v = math.copysign(V_DOWNHILL_SAFE, self.cmd_v)
                cmd_v_out = self.cmd_v
                self.hdc_active = True

            overspeed = abs(v_actual) - V_DOWNHILL_SAFE
            if overspeed > 0.0:
                self.hdc_active = True
                # Frein moteur préventif progressif proportionnel à la survitesse
                brake_bias = min(0.65, 1.8 * overspeed)
                if v_actual > 0.0:
                    cmd_v_out = min(cmd_v_out, max(0.15, V_DOWNHILL_SAFE - brake_bias))
                else:
                    cmd_v_out = max(cmd_v_out, min(-0.15, -V_DOWNHILL_SAFE + brake_bias))

        # --- 3. ESP anti-tangage (protège l'équilibre sur les cassures de pente / dos d'âne) ---
        instability = max(
            0.0,
            (abs(gyro_y) - 0.65) / 1.10,
            (abs(pitch_sin) - 0.28) / 0.32,
        )
        esp_factor = float(np.clip(1.0 - 0.80 * instability, 0.20, 1.0))
        self.esp_active = esp_factor < 0.92

        if instability > 0.35 and abs(cmd_v_out) > 0.85:
            # Libère du couple moteur pour laisser PPO redresser le pendule
            cmd_v_out = math.copysign(max(0.85, abs(cmd_v_out) * esp_factor), cmd_v_out)

        # --- 4. Direction directe + retour au centre rapide + sensibilité vitesse ---
        steer_dir = (1.0 if left else 0.0) - (1.0 if right else 0.0)
        if handbrake:
            steer_dir = 0.0

        if steer_dir != 0.0:
            rate = STEER_RETURN_RATE if steer_dir * self.steer_axis < 0.0 else STEER_IN_RATE
            self.steer_axis = float(np.clip(self.steer_axis + steer_dir * rate * dt, -1.0, 1.0))
        else:
            step = STEER_RETURN_RATE * dt
            self.steer_axis = (
                0.0 if abs(self.steer_axis) <= step else self.steer_axis - math.copysign(step, self.steer_axis)
            )

        # Courbe Sport : attaque directe sur petits appuis tout en restant dosable
        s = self.steer_axis
        steer_shaped = math.copysign(0.68 * abs(s) + 0.32 * (s * s), s)

        if cmd_v_out < -0.05:
            steer_shaped = -steer_shaped

        # Enveloppe dynamique : très vif à moyenne vitesse, stabilisé à haute vitesse / descente
        v_eff = max(abs(cmd_v_out), 0.70 * abs(v_actual))
        if v_eff > 0.12:
            high_speed_tame = 1.0 - 0.25 * float(np.clip((abs(v_actual) - 1.15) / 0.45, 0.0, 1.0))
            max_yaw = min(0.78, 0.72 * v_eff) * high_speed_tame * esp_factor
            if descending:
                max_yaw *= 0.75
        else:
            max_yaw = 0.0

        # Maintien de cap automatique en ligne droite (compense le biais PPO)
        if abs(self.steer_axis) <= 0.02 and abs(v_actual) >= 0.20 and not handbrake:
            if self.hold_heading is None:
                if abs(yaw_actual) < 0.25:
                    self.hold_heading = heading
                target_yaw = 0.0
            else:
                h_err = (self.hold_heading - heading + math.pi) % (2.0 * math.pi) - math.pi
                # Si choc contre un obstacle/muret, on ré-accroche le nouveau cap
                if abs(h_err) > 0.45:
                    self.hold_heading = heading
                    h_err = 0.0
                self.yaw_trim = float(np.clip(self.yaw_trim + 0.85 * h_err * dt, -0.24, 0.24))
                target_yaw = float(np.clip(1.35 * h_err + self.yaw_trim - 0.18 * yaw_actual, -0.32, 0.32))
        else:
            self.hold_heading = None
            target_yaw = steer_shaped * max_yaw

        dy = YAW_RAMP * dt
        self.cmd_yaw += float(np.clip(target_yaw - self.cmd_yaw, -dy, dy))
        return cmd_v_out, self.cmd_yaw


def steer_gauge(axis: float) -> str:
    pos = max(0, min(8, int(round((1.0 - axis) * 4.0))))
    chars = ["-" for _ in range(9)]
    chars[4] = "|"
    chars[pos] = "O"
    return "[" + "".join(chars) + "]"


def main():
    arena_xml = build_arena_xml()
    raw_env, venv, model = load_policy_and_arena(arena_xml)
    driver = TeleopController()

    if not glfw.init():
        raise RuntimeError("Impossible d'initialiser GLFW.")

    window = glfw.create_window(1280, 780, "BB-8 — Téléopération + HDC/ESP (PPO v3)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Impossible de créer la fenêtre GLFW.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.35]
    cam.distance = CAM_PRESETS[0][1]
    cam.elevation = CAM_PRESETS[0][2]
    cam.azimuth = 0.0

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()
    scn = mujoco.MjvScene(raw_env.model, maxgeom=2000)
    ctx = mujoco.MjrContext(raw_env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    ui = {
        "cam_idx": 0,
        "base_dist": CAM_PRESETS[0][1],
        "orbiting": False,
        "last_cursor": (0.0, 0.0),
        "reset_req": False,
    }

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_C:
            ui["cam_idx"] = (ui["cam_idx"] + 1) % len(CAM_PRESETS)
            _, dist, elev = CAM_PRESETS[ui["cam_idx"]]
            ui["base_dist"] = dist
            cam.elevation = elev
        elif key == glfw.KEY_R:
            ui["reset_req"] = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    def on_mouse_button(win, button, action, mods):
        if button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
            ui["orbiting"] = action == glfw.PRESS
            ui["last_cursor"] = glfw.get_cursor_pos(win)

    def on_cursor_pos(win, xpos, ypos):
        lx, ly = ui["last_cursor"]
        dx, dy = xpos - lx, ypos - ly
        ui["last_cursor"] = (xpos, ypos)
        if ui["orbiting"]:
            cam.azimuth -= 0.30 * dx
            cam.elevation = float(np.clip(cam.elevation - 0.25 * dy, -80.0, -8.0))

    def on_scroll(win, xoffset, yoffset):
        ui["base_dist"] = float(np.clip(ui["base_dist"] * (0.92 ** yoffset), 1.8, 12.0))

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)

    obs = venv.reset()
    v_actual = 0.0
    yaw_actual = 0.0

    print("\n=== TÉLÉOPÉRATION BB-8 — PPO v3 + HDC / ESP ===")
    print("  Z / W / ↑        : Accelerer (vif)")
    print("  S / ↓            : Freiner -> Marche arriere")
    print("  Q / A / ← / D / →: Direction directe + retour au centre")
    print("  Shift / Espace   : Boost / Frein a main")
    print("  C / R            : Camera / Reset\n")

    try:
        while not glfw.window_should_close(window):
            t_start = time.perf_counter()
            glfw.poll_events()

            if ui["reset_req"]:
                obs = venv.reset()
                driver.reset()
                v_actual = 0.0
                yaw_actual = 0.0
                cam.azimuth = 0.0
                ui["reset_req"] = False

            def held(*keys) -> bool:
                return any(glfw.get_key(window, k) == glfw.PRESS for k in keys)

            throttle = held(glfw.KEY_Z, glfw.KEY_W, glfw.KEY_UP)
            brake = held(glfw.KEY_S, glfw.KEY_DOWN)
            left = held(glfw.KEY_Q, glfw.KEY_A, glfw.KEY_LEFT)
            right = held(glfw.KEY_D, glfw.KEY_RIGHT)
            boost = held(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
            handbrake = held(glfw.KEY_SPACE)

            # Capteurs temps réel pour HDC (vitesse verticale), ESP (IMU) et maintien de cap
            vz = float(raw_env.data.qvel[2])
            gyro_y = float(raw_env.last_gyro_meas[1])
            pitch_sin = float(raw_env.g_est[0])
            R_frame = raw_env.data.xmat[raw_env.frame_id].reshape(3, 3)
            heading = math.atan2(R_frame[1, 0], R_frame[0, 0])

            cmd_v, cmd_yaw = driver.update(
                throttle, brake, left, right, boost, handbrake,
                v_actual, vz, gyro_y, pitch_sin, heading, yaw_actual, raw_env.dt_ctrl,
            )
            raw_env.set_command(cmd_v, cmd_yaw)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            info = infos[0]
            v_actual = float(info["v_forward"])
            yaw_actual = float(info["yaw_rate"])

            if dones[0]:
                obs = venv.reset()
                driver.reset()
                v_actual = 0.0
                yaw_actual = 0.0

            # Caméra poursuite amortie
            pos_3d = raw_env.data.xpos[raw_env.shell_id]
            target_lookat = pos_3d + np.array([0.0, 0.0, 0.14])
            cam.lookat[:] += 0.22 * (target_lookat - cam.lookat[:])

            cam_name = CAM_PRESETS[ui["cam_idx"]][0]
            if ui["cam_idx"] < 2 and not ui["orbiting"]:
                R_frame = raw_env.data.xmat[raw_env.frame_id].reshape(3, 3)
                heading_deg = math.degrees(math.atan2(R_frame[1, 0], R_frame[0, 0]))
                diff = (heading_deg - cam.azimuth + 180.0) % 360.0 - 180.0
                if v_actual > 0.15:
                    cam.azimuth += 0.065 * diff
                elif v_actual < -0.15:
                    cam.azimuth += 0.015 * diff

            target_dist = ui["base_dist"] + (0.35 * abs(v_actual) if ui["cam_idx"] < 2 else 0.0)
            cam.distance += 0.08 * (target_dist - cam.distance)

            # Rendu OpenGL + HUD avec voyants HDC / ESP
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
            mujoco.mjv_updateScene(
                raw_env.model, raw_env.data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn
            )
            mujoco.mjr_render(viewport, scn, ctx)

            speed_kmh = abs(v_actual) * 3.6
            gear = "R" if cmd_v < -0.05 else ("N" if abs(cmd_v) < 0.05 else ("B" if boost else "D"))
            assists = " ".join(
                tag for tag, on in (("[HDC DESCENTE]", driver.hdc_active), ("[ESP]", driver.esp_active)) if on
            ) or "OK"

            hud_left = (
                "Vitesse\n"
                "Direction\n"
                "Assistances\n"
                "Batterie\n"
                "Camera"
            )
            hud_right = (
                f"{speed_kmh:4.1f} km/h  ({v_actual:+.2f} m/s)   [{gear}]\n"
                f"{steer_gauge(driver.steer_axis)}   lacet {info['yaw_rate']:+.2f} rad/s\n"
                f"{assists}\n"
                f"{info['v_batt']:.2f} V  |  {info['i_batt']:.1f} A  ({raw_env.battery.soc*100:.0f}%)\n"
                f"{cam_name} (C)  |  ZQSD/Fleches  |  Shift: Boost  |  Espace: Frein  |  R: Reset"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                hud_left,
                hud_right,
                ctx,
            )

            glfw.swap_buffers(window)

            remaining = raw_env.dt_ctrl - (time.perf_counter() - t_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()