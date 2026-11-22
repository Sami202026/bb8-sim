from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class LiPo4SBattery:
    """Batterie LiPo 4S avec courbe OCV réaliste et chute de tension dynamique."""

    capacity_ah: float = 5.0
    r_internal: float = 0.08          # Pack + câblage + connecteur XT60 (Ohms)
    initial_soc: float = 1.0          # État de charge initial (0.0 à 1.0)
    soc_speedup: float = 1.0          # >1.0 pour accélérer la décharge en test
    sag_tau: float = 0.02             # Constante de temps polarisation (s)
    quiescent_current_a: float = 0.40 # ESP32 + IMU + radio + LEDs (A)

    soc: float = field(init=False)
    v_sag: float = field(default=0.0, init=False)
    v_terminal: float = field(init=False)
    last_current_a: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.reset(self.initial_soc)

    def reset(self, soc: float | None = None):
        self.soc = float(np.clip(self.initial_soc if soc is None else soc, 0.0, 1.0))
        self.v_sag = 0.0
        self.last_current_a = self.quiescent_current_a
        self.v_terminal = self.open_circuit_voltage()

    def open_circuit_voltage(self) -> float:
        """Courbe caractéristique de tension à vide d'une LiPo 4S (4 cellules)."""
        soc_pts = np.array([0.00, 0.05, 0.15, 0.30, 0.50, 0.75, 0.90, 1.00])
        v_cell = np.array([3.20, 3.55, 3.68, 3.75, 3.84, 3.98, 4.10, 4.20])
        return float(4.0 * np.interp(self.soc, soc_pts, v_cell))

    def step(self, load_current_a: float, dt: float) -> float:
        total_current = max(0.0, load_current_a) + self.quiescent_current_a
        self.last_current_a = total_current

        # Consommation en Ampères-heures
        consumed_ah = (total_current * dt / 3600.0) * self.soc_speedup
        self.soc = max(0.0, self.soc - consumed_ah / self.capacity_ah)

        # Chute de tension interne filtrée (évite les boucles algébriques instables)
        target_sag = total_current * self.r_internal
        alpha = min(1.0, dt / max(self.sag_tau, dt))
        self.v_sag += alpha * (target_sag - self.v_sag)

        self.v_terminal = max(11.0, self.open_circuit_voltage() - self.v_sag)
        return self.v_terminal

    @property
    def low_voltage_cutoff(self) -> bool:
        return self.v_terminal <= 12.6


@dataclass
class DualDCGearMotor:
    """Double motoréducteur DC en parallèle piloté par pont en H (PWM -1..+1)."""

    n_motors: int = 2
    gear_ratio: float = 25.0
    gearbox_eff: float = 0.82
    kt: float = 0.036                 # Constante de couple (N·m/A)
    ke: float = 0.036                 # Constante de FCEM (V·s/rad)
    r_armature: float = 1.8           # Résistance bobinage + MOSFETs (Ohms)
    i_max_per_motor: float = 3.7      # Limite courant du driver (A/moteur)
    coulomb_friction: float = 0.15    # Frottement sec des réducteurs (N·m)
    driver_eff: float = 0.92          # Rendement pont en H

    def compute(self, pwm: float, omega_out: float, v_batt: float) -> tuple[float, float, float]:
        """
        Retourne (couple_net_Nm, courant_par_moteur_A, courant_tiré_batterie_A).
        """
        u = float(np.clip(pwm, -1.0, 1.0))
        v_applied = u * v_batt

        # Équation électrique : V = R * I + Ke * (N * omega)
        omega_motor = self.gear_ratio * omega_out
        back_emf = self.ke * omega_motor
        i_motor = float(np.clip((v_applied - back_emf) / self.r_armature,
                                -self.i_max_per_motor, self.i_max_per_motor))

        # Couple électromagnétique ramené à l'axe de sortie
        tau_em = self.n_motors * self.gearbox_eff * self.gear_ratio * self.kt * i_motor

        # Zone morte / frottement sec (Coulomb)
        if abs(omega_out) > 0.02:
            tau_net = tau_em - math.copysign(self.coulomb_friction, omega_out)
        else:
            if abs(tau_em) <= self.coulomb_friction:
                tau_net = 0.0
            else:
                tau_net = tau_em - math.copysign(self.coulomb_friction, tau_em)

        # Courant moyen tiré sur la batterie (pont en H non régénératif)
        i_batt = self.n_motors * max(0.0, u * i_motor) / self.driver_eff
        return tau_net, i_motor, i_batt


class NoisyIMU:
    """IMU 6 axes avec bruit blanc gaussien, biais à l'allumage et dérive lente."""

    def __init__(
        self,
        accel_noise_std: float = 0.04,        # m/s²
        gyro_noise_std: float = 0.012,        # rad/s
        accel_bias_init_std: float = 0.05,    # m/s²
        gyro_bias_init_std: float = 0.015,    # rad/s
        accel_bias_drift: float = 0.002,      # m/s² / sqrt(s)
        gyro_bias_drift: float = 0.0015,      # rad/s / sqrt(s)
        seed: int = 42,
    ):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_init_std = accel_bias_init_std
        self.gyro_bias_init_std = gyro_bias_init_std
        self.accel_bias_drift = accel_bias_drift
        self.gyro_bias_drift = gyro_bias_drift
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.accel_bias = self.rng.normal(0.0, self.accel_bias_init_std, size=3)
        self.gyro_bias = self.rng.normal(0.0, self.gyro_bias_init_std, size=3)

    def sample(self, true_accel: np.ndarray, true_gyro: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        sqrt_dt = math.sqrt(dt)
        self.accel_bias += self.rng.normal(0.0, self.accel_bias_drift * sqrt_dt, size=3)
        self.gyro_bias += self.rng.normal(0.0, self.gyro_bias_drift * sqrt_dt, size=3)

        accel_meas = true_accel + self.accel_bias + self.rng.normal(0.0, self.accel_noise_std, size=3)
        gyro_meas = true_gyro + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise_std, size=3)
        return accel_meas, gyro_meas


class QuadratureEncoder:
    """Encodeur incrémental quantifié mesurant la vitesse sur la période MCU."""

    def __init__(self, counts_per_rev: int = 1024):
        self.rad_per_tick = (2.0 * math.pi) / counts_per_rev
        self.prev_ticks: int | None = None

    def reset(self):
        self.prev_ticks = None

    def measure_velocity(self, joint_angle_rad: float, dt_ctrl: float) -> float:
        ticks = round(joint_angle_rad / self.rad_per_tick)
        if self.prev_ticks is None:
            self.prev_ticks = ticks
            return 0.0
        delta_ticks = ticks - self.prev_ticks
        self.prev_ticks = ticks
        return (delta_ticks * self.rad_per_tick) / dt_ctrl


class CommandLatencyBuffer:
    """Retard pur (buffer FIFO) simulant latence bus + calcul MCU + driver."""

    def __init__(self, delay_steps: int, default_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.delay_steps = max(0, int(delay_steps))
        self.default_cmd = default_cmd
        self.buffer: deque[tuple[float, float, float]] = deque()
        self.reset()

    def reset(self):
        self.buffer = deque([self.default_cmd] * self.delay_steps, maxlen=max(1, self.delay_steps))

    def push_and_pop(self, cmd: tuple[float, float, float]) -> tuple[float, float, float]:
        if self.delay_steps == 0:
            return cmd
        delayed_cmd = self.buffer[0]
        self.buffer.append(cmd)
        return delayed_cmd