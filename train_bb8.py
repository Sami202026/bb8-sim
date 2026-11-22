import os
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from bb8_env import BB8Env

N_ENVS = 6
TOTAL_TIMESTEPS = 1_500_000   # ~3,5 min sur MacBook Air M3
MODELS_DIR = Path("models")
BEST_DIR = MODELS_DIR / "best_model"
LOGS_DIR = Path("tb_logs")


def linear_schedule(initial_lr: float, final_lr: float):
    """Décroissance linéaire du taux d'apprentissage pour affiner la convergence."""
    def _schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return _schedule


class SaveBestWithNormCallback(EvalCallback):
    """Sauvegarde vec_normalize.pkl au moment exact où un nouveau best_model est trouvé."""

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > prev_best and self.best_model_save_path is not None:
            self.training_env.save(str(Path(self.best_model_save_path) / "vec_normalize.pkl"))
        return result


def make_env(rank: int, seed: int = 42, domain_rand: bool = True):
    def _init():
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        env = BB8Env(xml_path="bb8_cad.xml", domain_rand=domain_rand)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    BEST_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Environnements d'entraînement parallèles
    train_env = SubprocVecEnv([make_env(i, seed=42, domain_rand=True) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Environnement d'évaluation déterministe (synchronisé automatiquement avec train_env)
    eval_env = DummyVecEnv([make_env(0, seed=1000, domain_rand=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(3e-4, 3e-5),
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=str(LOGS_DIR),
        device="cpu",
        verbose=1,
    )

    eval_cb = SaveBestWithNormCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOGS_DIR / "eval"),
        eval_freq=max(60_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(250_000 // N_ENVS, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="bb8_ppo_v2",
        save_vecnormalize=True,
    )

    print(f"Démarrage entraînement PPO v2 ({N_ENVS} envs, {TOTAL_TIMESTEPS:,} pas)...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([eval_cb, checkpoint_cb]),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEntraînement interrompu manuellement : sauvegarde en cours...")
    finally:
        model.save(str(MODELS_DIR / "bb8_ppo_last"))
        train_env.save(str(MODELS_DIR / "vec_normalize_last.pkl"))
        train_env.close()
        eval_env.close()
        print("Sauvegarde terminée : meilleur modèle dans models/best_model/")


if __name__ == "__main__":
    main()