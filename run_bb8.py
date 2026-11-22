import math
import time
import mujoco
import mujoco.viewer

SPHERE_RADIUS = 0.25


def get_demo_command(t: float) -> tuple[float, float, float]:
    """Scénario de test cyclique de 18 secondes : (vitesse m/s, inclinaison rad, tête rad)."""
    cycle = t % 18.0
    head_yaw = 0.6 * math.sin(1.2 * t)  # La tête balaye doucement

    if cycle < 4.0:
        return 1.2, 0.0, head_yaw          # Ligne droite
    elif cycle < 9.0:
        return 1.2, 0.45, 0.5              # Virage à gauche (pendule incliné)
    elif cycle < 14.0:
        return 1.2, -0.45, -0.5            # Virage à droite
    else:
        return 0.0, 0.0, 0.0               # Freinage et stabilisation


def main():
    model = mujoco.MjModel.from_xml_path("bb8_cad.xml")
    data = mujoco.MjData(model)

    shell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shell")
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_servo")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_servo")

    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        while viewer.is_running():
            step_start = time.perf_counter()

            # 1. Lecture des capteurs virtuels
            omega_meas = data.sensor("drive_encoder_vel").data[0]
            gyro_y = data.sensor("imu_gyro").data[1]

            # 2. Consignes de démonstration
            target_speed, target_steer, target_head = get_demo_command(data.time)
            target_omega = target_speed / SPHERE_RADIUS

            # 3. Contrôleur de vitesse + amortissement actif du balancement du pendule
            kp_vel = 0.8
            kd_pitch = 0.8
            drive_torque = kp_vel * (target_omega - omega_meas) + kd_pitch * gyro_y

            # 4. Envoi des commandes aux actionneurs
            data.ctrl[drive_id] = max(-6.0, min(6.0, drive_torque))
            data.ctrl[steer_id] = target_steer
            data.ctrl[head_id] = target_head

            # 5. Avancement d'un pas physique (2 ms)
            mujoco.mj_step(model, data)

            # Suivi automatique de la caméra sur la sphère
            viewer.cam.lookat[:] = data.xpos[shell_id]
            viewer.sync()

            # Télémétrie dans le terminal toutes les 0.5 s
            if data.time - last_print >= 0.5:
                pos = data.xpos[shell_id]
                v_lin = omega_meas * SPHERE_RADIUS
                print(
                    f"t={data.time:5.1f}s | pos=({pos[0]:5.2f}, {pos[1]:5.2f}) m | "
                    f"v={v_lin:4.2f} m/s | couple={data.ctrl[drive_id]:+5.2f} N·m | "
                    f"gyro_y={gyro_y:+5.2f} rad/s"
                )
                last_print = data.time

            # Cadencement temps réel
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()