import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from bb8_electronics import (
    CommandLatencyBuffer,
    DualDCGearMotor,
    LiPo4SBattery,
    NoisyIMU,
    QuadratureEncoder,
)


class BB8Env(gym.Env):
    """Environnement Gymnasium v3 pour robot sphérique pendulaire : commande lissée et 100 % Sim-to-Real."""

    metadata = {"render_modes": ["human"]}

    SPHERE_RADIUS = 0.25

    def __init__(
        self,
        xml_path: str = "bb8_cad.xml",
        control_hz: float = 50.0,
        episode_seconds: float = 12.0,
        domain_rand: bool = True,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt_sim = float(self.model.opt.timestep)
        self.dt_ctrl = 1.0 / control_hz
        self.n_substeps = max(1, round(self.dt_ctrl / self.dt_sim))
        self.max_steps = int(episode_seconds * control_hz)
        self.domain_rand = domain_rand

        # Identifiants MuJoCo
        self.shell_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shell")
        self.frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "inner_frame")
        self.pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.shell_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "shell_collision")
        self.drive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
        self.steer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_servo")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_servo")

        # Paramètres nominaux pour la Domain Randomization
        self._base_floor_fric = self.model.geom_friction[self.floor_id].copy()
        self._base_shell_fric = self.model.geom_friction[self.shell_geom_id].copy()
        self._base_pend_mass = float(self.model.body_mass[self.pend_id])
        self._base_pend_inertia = self.model.body_inertia[self.pend_id].copy()

        # Chaîne électronique embarquée
        self.battery = LiPo4SBattery(capacity_ah=5.0, r_internal=0.08, initial_soc=1.0)
        self.motors = DualDCGearMotor()
        self.imu = NoisyIMU(seed=0)
        self.encoder = QuadratureEncoder(counts_per_rev=512 * 25)
        self.latency_buf = CommandLatencyBuffer(delay_steps=6)

        # Actions : [pwm_cible (-1..1), steer_norm (-1..1 -> ±0.50 rad)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations v3 (16 valeurs, 100 % issues des capteurs embarqués) :
        # [g_est(3), gyro_meas(3), omega_enc(1), omega_shell_est(1),
        #  steer_pos(1), v_batt_norm(1), action_filt(2),
        #  v_cmd(1), yaw_cmd(1), v_err_int(1), yaw_err_int(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev2_action = np.zeros(2, dtype=np.float32)
        self.action_filt = np.zeros(2, dtype=np.float32)
        self.head_cmd_ramped = 0.0

        self.v_cmd = 0.0
        self.yaw_cmd = 0.0
        self.external_command = False
        self.last_gyro_meas = np.zeros(3, dtype=np.float64)
        self.last_omega_enc = 0.0
        self.g_est = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.v_err_int = 0.0
        self.yaw_err_int = 0.0
        self.next_push_step = -1

    def set_command(self, v_cmd: float, yaw_cmd: float):
        """Force une consigne manuelle (utile en évaluation / pilotage interactif)."""
        self.external_command = True
        self.v_cmd = float(v_cmd)
        self.yaw_cmd = float(yaw_cmd)

    def _sample_command(self):
        """Répartition rééquilibrée : 55 % avant, 25 % arrière, 20 % arrêt."""
        r = self.np_random.random()
        if r < 0.20:
            self.v_cmd = 0.0
            self.yaw_cmd = 0.0
        elif r < 0.45:
            # Marche arrière renforcée (25 % du temps)
            self.v_cmd = float(self.np_random.uniform(-0.85, -0.30))
            max_yaw = min(0.55, 0.65 * abs(self.v_cmd))
            self.yaw_cmd = float(self.np_random.uniform(-max_yaw, max_yaw))
        else:
            # Marche avant agile
            self.v_cmd = float(self.np_random.uniform(0.30, 1.50))
            max_yaw = min(0.80, 0.72 * self.v_cmd)
            self.yaw_cmd = float(self.np_random.uniform(-max_yaw, max_yaw))

    def _apply_domain_randomization(self):
        if not self.domain_rand:
            self.model.geom_friction[self.floor_id] = self._base_floor_fric
            self.model.geom_friction[self.shell_geom_id] = self._base_shell_fric
            self.model.body_mass[self.pend_id] = self._base_pend_mass
            self.model.body_inertia[self.pend_id] = self._base_pend_inertia
            self.motors.r_armature = 1.8
            self.motors.coulomb_friction = 0.15
            self.battery.reset(soc=1.0)
            self.latency_buf = CommandLatencyBuffer(delay_steps=6)
            return

        # 1. Adhérence du sol
        fric_scale = float(self.np_random.uniform(0.70, 1.30))
        self.model.geom_friction[self.floor_id, 0] = self._base_floor_fric[0] * fric_scale
        self.model.geom_friction[self.shell_geom_id, 0] = self._base_shell_fric[0] * fric_scale

        # 2. Masse et inertie du pendule (±12 %)
        mass_scale = float(self.np_random.uniform(0.88, 1.12))
        self.model.body_mass[self.pend_id] = self._base_pend_mass * mass_scale
        self.model.body_inertia[self.pend_id] = self._base_pend_inertia * mass_scale

        # 3. Dispersion moteurs DC & niveau batterie
        self.motors.r_armature = float(self.np_random.uniform(1.55, 2.05))
        self.motors.coulomb_friction = float(self.np_random.uniform(0.08, 0.24))
        self.battery.reset(soc=float(self.np_random.uniform(0.25, 1.0)))

        # 4. Latence aléatoire entre 8 ms et 18 ms (4 à 9 pas de 2 ms)
        delay_steps = int(self.np_random.integers(4, 10))
        self.latency_buf = CommandLatencyBuffer(delay_steps=delay_steps)

    def _update_imu_attitude(self, accel_meas: np.ndarray, gyro_meas: np.ndarray):
        """Filtre complémentaire embarqué estimant la gravité locale (100 % Sim-to-Real)."""
        g_pred = self.g_est - np.cross(gyro_meas, self.g_est) * self.dt_sim
        n_pred = float(np.linalg.norm(g_pred))
        if n_pred > 1e-6:
            g_pred /= n_pred

        a_norm = float(np.linalg.norm(accel_meas))
        if 6.5 < a_norm < 13.0:
            g_acc = -accel_meas / a_norm
            alpha = 0.01
            self.g_est = (1.0 - alpha) * g_pred + alpha * g_acc
            self.g_est /= max(float(np.linalg.norm(self.g_est)), 1e-6)
        else:
            self.g_est = g_pred

    def _get_frame_kinematics(self) -> tuple[np.ndarray, float, float]:
        """Retourne (g_body_vrai, vitesse_avant_réelle, taux_lacet_réel)."""
        R_frame = self.data.xmat[self.frame_id].reshape(3, 3)
        g_body = R_frame.T @ np.array([0.0, 0.0, -1.0])

        fwd_xy = R_frame[:2, 0]
        norm = float(np.linalg.norm(fwd_xy))
        fwd_dir = fwd_xy / norm if norm > 1e-6 else np.array([1.0, 0.0])

        v_forward = float(np.dot(self.data.qvel[:2], fwd_dir))

        true_gyro = self.data.sensor("imu_gyro").data
        omega_world = R_frame @ true_gyro
        yaw_rate = float(omega_world[2])

        return g_body, v_forward, yaw_rate

    def _get_obs(self) -> np.ndarray:
        omega_shell_est = self.last_omega_enc + float(self.last_gyro_meas[1])
        steer_pos = float(self.data.sensor("steer_encoder_pos").data[0])
        v_batt_norm = (self.battery.v_terminal - 14.8) / 2.0

        return np.array(
            [
                self.g_est[0],
                self.g_est[1],
                self.g_est[2],
                self.last_gyro_meas[0],
                self.last_gyro_meas[1],
                self.last_gyro_meas[2],
                self.last_omega_enc,
                omega_shell_est,
                steer_pos,
                v_batt_norm,
                self.action_filt[0],
                self.action_filt[1],
                self.v_cmd,
                self.yaw_cmd,
                self.v_err_int,
                self.yaw_err_int,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._apply_domain_randomization()
        self.imu.rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))
        self.imu.reset()
        self.encoder.reset()

        if self.domain_rand:
            self.data.qpos[7] = float(self.np_random.uniform(-0.18, 0.18))
            self.data.qpos[8] = float(self.np_random.uniform(-0.12, 0.12))

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev2_action = np.zeros(2, dtype=np.float32)
        self.action_filt = np.zeros(2, dtype=np.float32)
        self.head_cmd_ramped = 0.0
        self.last_gyro_meas = np.zeros(3, dtype=np.float64)
        self.last_omega_enc = 0.0
        self.v_err_int = 0.0
        self.yaw_err_int = 0.0
        self.encoder.measure_velocity(float(self.data.joint("drive_joint").qpos[0]), self.dt_ctrl)

        g_body_true, _, _ = self._get_frame_kinematics()
        self.g_est = g_body_true + self.np_random.normal(0.0, 0.01, size=3)
        self.g_est /= max(float(np.linalg.norm(self.g_est)), 1e-6)

        self.next_push_step = int(self.np_random.integers(120, 220)) if self.domain_rand else -1

        if not self.external_command:
            self._sample_command()

        return self._get_obs(), {}

    def step(self, action):
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # --- Filtrage passe-bas + limitation de pente (supprime le chatter 3 Hz et les pics 9 A
        #     tout en préservant la réactivité et l'inclinaison dynamique du pendule) ---
        target_pwm = self.action_filt[0] + 0.45 * (raw_action[0] - self.action_filt[0])
        target_steer = self.action_filt[1] + 0.35 * (raw_action[1] - self.action_filt[1])

        pwm_f = float(self.action_filt[0] + np.clip(target_pwm - self.action_filt[0], -0.22, 0.22))
        steer_f = float(self.action_filt[1] + np.clip(target_steer - self.action_filt[1], -0.09, 0.09))
        self.action_filt = np.array(
            [np.clip(pwm_f, -1.0, 1.0), np.clip(steer_f, -1.0, 1.0)], dtype=np.float32
        )

        pwm_cmd = float(self.action_filt[0])
        steer_cmd = float(self.action_filt[1]) * 0.50

        # Orientation dynamique de la tête : anticipation proportionnelle au virage sans pic servo
        head_target = float(np.clip(1.0 * self.yaw_cmd, -0.95, 0.95))
        max_dh = 3.0 * self.dt_ctrl
        self.head_cmd_ramped += float(np.clip(head_target - self.head_cmd_ramped, -max_dh, max_dh))

        if self.battery.low_voltage_cutoff:
            pwm_cmd = 0.0

        # Poussée externe aléatoire pendant l'entraînement
        if self.domain_rand and self.step_count == self.next_push_step:
            angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            dv = float(self.np_random.uniform(0.15, 0.45))
            self.data.qvel[0] += dv * math.cos(angle)
            self.data.qvel[1] += dv * math.sin(angle)
            self.next_push_step += int(self.np_random.integers(150, 250))

        mcu_cmd = (pwm_cmd, steer_cmd, self.head_cmd_ramped)
        mean_i_batt = 0.0
        peak_i_batt = 0.0
        mean_tau_net = 0.0
        last_pwm_delayed = 0.0

        # Intégration physique 500 Hz + électronique complète
        for _ in range(self.n_substeps):
            true_accel = self.data.sensor("imu_accel").data.copy()
            true_gyro = self.data.sensor("imu_gyro").data.copy()
            accel_meas, gyro_meas = self.imu.sample(true_accel, true_gyro, self.dt_sim)
            self.last_gyro_meas = gyro_meas
            self._update_imu_attitude(accel_meas, gyro_meas)

            pwm_d, steer_d, head_d = self.latency_buf.push_and_pop(mcu_cmd)
            last_pwm_delayed = pwm_d

            omega_true = float(self.data.sensor("drive_encoder_vel").data[0])
            tau_net, _, i_drive = self.motors.compute(
                pwm=pwm_d, omega_out=omega_true, v_batt=self.battery.v_terminal
            )

            tau_steer = abs(float(self.data.actuator_force[self.steer_id]))
            tau_head = abs(float(self.data.actuator_force[self.head_id]))
            i_servos = 0.12 * tau_steer + 0.08 * tau_head

            self.battery.step(i_drive + i_servos, self.dt_sim)
            mean_i_batt += self.battery.last_current_a
            peak_i_batt = max(peak_i_batt, self.battery.last_current_a)
            mean_tau_net += tau_net

            self.data.ctrl[self.drive_id] = float(np.clip(tau_net, -5.5, 5.5))
            self.data.ctrl[self.steer_id] = steer_d
            self.data.ctrl[self.head_id] = head_d

            mujoco.mj_step(self.model, self.data)

        mean_i_batt /= self.n_substeps
        mean_tau_net /= self.n_substeps
        q_drive = float(self.data.joint("drive_joint").qpos[0])
        self.last_omega_enc = self.encoder.measure_velocity(q_drive, self.dt_ctrl)

        # Mise à jour des intégrateurs d'erreur (100 % capteurs embarqués -> supprime les biais)
        v_shell_est = (self.last_omega_enc + float(self.last_gyro_meas[1])) * self.SPHERE_RADIUS
        yaw_est = -float(np.dot(self.last_gyro_meas, self.g_est))
        self.v_err_int = float(np.clip(0.96 * self.v_err_int + (self.v_cmd - v_shell_est) * self.dt_ctrl, -1.5, 1.5))
        self.yaw_err_int = float(np.clip(0.96 * self.yaw_err_int + (self.yaw_cmd - yaw_est) * self.dt_ctrl, -1.5, 1.5))

        # Cinématique réelle pour la fonction de récompense
        g_body_true, v_forward, yaw_rate = self._get_frame_kinematics()
        true_gyro_y = float(self.data.sensor("imu_gyro").data[1])

        # --- Récompense v3 : précision avant/arrière, anti-chatter et protection pics courant ---
        r_vel = 2.4 * math.exp(-4.5 * (v_forward - self.v_cmd) ** 2)
        r_yaw = 1.8 * math.exp(-4.5 * (yaw_rate - self.yaw_cmd) ** 2)
        r_upright = 0.4 * max(0.0, float(-g_body_true[2]))

        # Amortit le tangage haute fréquence sans empêcher le léger penchement naturel au démarrage
        p_pitch = -0.18 * (true_gyro_y ** 2)
        # Double pénalité (vitesse + jerk des actions) pour tuer le cycle limite à 3 Hz
        p_smooth = -0.55 * float(np.sum((raw_action - self.prev_action) ** 2))
        p_jerk = -0.35 * float(np.sum((raw_action - 2.0 * self.prev_action + self.prev2_action) ** 2))
        # Barrière douce au-dessus de 4.0 A pour éliminer les pics batterie sans brider la marche arrière
        p_energy = -0.012 * (mean_i_batt ** 2) - 0.10 * (max(0.0, peak_i_batt - 4.0) ** 2)
        # Évite les à-coups de pendule latéral à l'arrêt
        p_stop_steer = -0.45 * (float(self.action_filt[1]) ** 2) if abs(self.v_cmd) < 0.08 else 0.0

        reward = r_vel + r_yaw + r_upright + p_pitch + p_smooth + p_jerk + p_energy + p_stop_steer

        terminated = bool(-g_body_true[2] < 0.34)
        if terminated:
            reward -= 20.0

        self.prev2_action = self.prev_action.copy()
        self.prev_action = raw_action.copy()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        if not self.external_command and (self.step_count % 200 == 0) and not truncated:
            self._sample_command()

        info = {
            "v_forward": v_forward,
            "v_cmd": self.v_cmd,
            "yaw_rate": yaw_rate,
            "yaw_cmd": self.yaw_cmd,
            "v_batt": self.battery.v_terminal,
            "v_ocv": self.battery.open_circuit_voltage(),
            "i_batt": mean_i_batt,
            "tau_net": mean_tau_net,
            "pwm": last_pwm_delayed,
            "gyro_y_true": true_gyro_y,
            "gyro_y_meas": float(self.last_gyro_meas[1]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info