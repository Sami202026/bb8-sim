import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

MODEL_PATH = Path("models/best_model/best_model.zip")
NORM_PATH = Path("models/best_model/vec_normalize.pkl")
OUT_HEADER = Path("wokwi_bb8/include/bb8_policy.h")


def format_c_array(name: str, arr: np.ndarray, per_line: int = 8) -> str:
    flat = arr.astype(np.float32).ravel()
    lines = [f"static const float {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(f"{v:.8e}f" for v in flat[i : i + per_line])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def detect_activation(policy_net: nn.Sequential) -> tuple[str, str]:
    for module in policy_net:
        if isinstance(module, nn.Tanh):
            return "tanh", "tanhf(x)"
        if isinstance(module, nn.ReLU):
            return "relu", "(x > 0.0f ? x : 0.0f)"
        if isinstance(module, nn.ELU):
            return "elu", "(x > 0.0f ? x : expf(x) - 1.0f)"
        if isinstance(module, nn.SiLU):
            return "silu", "(x / (1.0f + expf(-x)))"
    raise ValueError("Fonction d'activation non reconnue dans mlp_extractor.policy_net.")


def simulate_cpp_forward(
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_inv_std: np.ndarray,
    clip_obs: float,
    layers: list[tuple[np.ndarray, np.ndarray]],
    act_name: str,
    act_low: np.ndarray,
    act_high: np.ndarray,
) -> np.ndarray:
    x = np.clip((obs.astype(np.float32) - obs_mean) * obs_inv_std, -clip_obs, clip_obs)
    for idx, (w, b) in enumerate(layers):
        x = w @ x + b
        if idx < len(layers) - 1:
            if act_name == "tanh":
                x = np.tanh(x)
            elif act_name == "relu":
                x = np.maximum(x, 0.0)
            elif act_name == "elu":
                x = np.where(x > 0.0, x, np.exp(x) - 1.0)
            elif act_name == "silu":
                x = x / (1.0 + np.exp(-x))
    return np.clip(x, act_low, act_high)


def main():
    if not MODEL_PATH.exists() or not NORM_PATH.exists():
        raise FileNotFoundError(
            f"Fichiers introuvables : vérifiez {MODEL_PATH} et {NORM_PATH}."
        )

    # 1. Chargement du modèle PPO et des stats VecNormalize
    model = PPO.load(str(MODEL_PATH), device="cpu")
    with open(NORM_PATH, "rb") as f:
        vec_norm = pickle.load(f)

    obs_mean = np.asarray(vec_norm.obs_rms.mean, dtype=np.float32)
    obs_var = np.asarray(vec_norm.obs_rms.var, dtype=np.float32)
    eps = float(getattr(vec_norm, "epsilon", 1e-8))
    clip_obs = float(getattr(vec_norm, "clip_obs", 10.0))
    obs_inv_std = (1.0 / np.sqrt(obs_var + eps)).astype(np.float32)

    obs_dim = int(obs_mean.shape[0])
    act_low = np.asarray(model.action_space.low, dtype=np.float32).ravel()
    act_high = np.asarray(model.action_space.high, dtype=np.float32).ravel()
    act_dim = int(act_low.shape[0])

    # 2. Extraction des couches de l'acteur (policy_net + action_net)
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net
    act_name, act_c_expr = detect_activation(policy_net)

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for m in policy_net:
        if isinstance(m, nn.Linear):
            layers.append((
                m.weight.detach().cpu().numpy().astype(np.float32),
                m.bias.detach().cpu().numpy().astype(np.float32),
            ))
    layers.append((
        action_net.weight.detach().cpu().numpy().astype(np.float32),
        action_net.bias.detach().cpu().numpy().astype(np.float32),
    ))

    # 3. Vérification numérique stricte contre Stable-Baselines3 (100 vecteurs)
    rng = np.random.default_rng(1234)
    max_err = 0.0
    test_obs = None
    test_act_ref = None

    for i in range(100):
        raw_obs = rng.normal(loc=obs_mean, scale=np.sqrt(obs_var + 1e-4)).astype(np.float32)
        norm_obs = np.clip((raw_obs - obs_mean) / np.sqrt(obs_var + eps), -clip_obs, clip_obs)
        sb3_act, _ = model.predict(norm_obs, deterministic=True)
        sb3_act = np.asarray(sb3_act, dtype=np.float32).ravel()

        cpp_act = simulate_cpp_forward(
            raw_obs, obs_mean, obs_inv_std, clip_obs, layers, act_name, act_low, act_high
        )
        err = float(np.max(np.abs(sb3_act - cpp_act)))
        max_err = max(max_err, err)
        if i == 0:
            test_obs = raw_obs
            test_act_ref = sb3_act

    if max_err > 1e-4:
        raise RuntimeError(f"Écart trop élevé entre SB3 et l'export C++ : {max_err:.2e}")

    # 4. Génération du header C++ autonome (poids stockés en Flash via const)
    max_hidden = max(w.shape[0] for w, _ in layers[:-1])
    total_params = sum(w.size + b.size for w, b in layers) + obs_mean.size + obs_inv_std.size

    code = [
        "#pragma once",
        "#include <math.h>",
        "",
        "// Généré automatiquement depuis models/best_model/best_model.zip + vec_normalize.pkl",
        f"// Architecture : {obs_dim} -> "
        + " -> ".join(str(w.shape[0]) for w, _ in layers)
        + f" (activation: {act_name})",
        f"// Paramètres float32 : {total_params} (~{total_params * 4 / 1024.0:.1f} Ko en Flash)",
        "",
        f"constexpr int BB8_OBS_DIM = {obs_dim};",
        f"constexpr int BB8_ACT_DIM = {act_dim};",
        f"constexpr float BB8_CLIP_OBS = {clip_obs:.6f}f;",
        "",
        format_c_array("BB8_OBS_MEAN", obs_mean),
        "",
        format_c_array("BB8_OBS_INV_STD", obs_inv_std),
        "",
        format_c_array("BB8_ACT_LOW", act_low),
        "",
        format_c_array("BB8_ACT_HIGH", act_high),
        "",
    ]

    for idx, (w, b) in enumerate(layers):
        code.append(f"// Couche {idx} : {w.shape[1]} -> {w.shape[0]}")
        code.append(format_c_array(f"BB8_W{idx}", w))
        code.append(format_c_array(f"BB8_B{idx}", b))
        code.append("")

    # Vecteur d'auto-test embarqué pour vérifier le calcul sur ESP32-S3
    code.append("// Vecteur de référence PyTorch/SB3 pour auto-test au démarrage")
    code.append(format_c_array("BB8_TEST_OBS", test_obs))
    code.append(format_c_array("BB8_TEST_ACT_REF", test_act_ref))
    code.append("")

    code.extend([
        "static inline float bb8_clamp(float x, float lo, float hi) {",
        "  return x < lo ? lo : (x > hi ? hi : x);",
        "}",
        "",
        "static inline float bb8_act(float x) {",
        f"  return {act_c_expr};",
        "}",
        "",
        "static inline void bb8_dense(",
        "    const float* in, int in_dim,",
        "    const float* w, const float* b,",
        "    float* out, int out_dim, bool apply_act) {",
        "  for (int i = 0; i < out_dim; ++i) {",
        "    const float* row = w + i * in_dim;",
        "    float sum = b[i];",
        "    for (int j = 0; j < in_dim; ++j) {",
        "      sum += row[j] * in[j];",
        "    }",
        "    out[i] = apply_act ? bb8_act(sum) : sum;",
        "  }",
        "}",
        "",
        "// Entrée : obs[BB8_OBS_DIM] brut (non normalisé)",
        "// Sortie : action_out[BB8_ACT_DIM] bornée dans [BB8_ACT_LOW, BB8_ACT_HIGH]",
        "inline void bb8_policy_step(const float* obs, float* action_out) {",
        "  float x_norm[BB8_OBS_DIM];",
        "  for (int i = 0; i < BB8_OBS_DIM; ++i) {",
        "    x_norm[i] = bb8_clamp((obs[i] - BB8_OBS_MEAN[i]) * BB8_OBS_INV_STD[i], -BB8_CLIP_OBS, BB8_CLIP_OBS);",
        "  }",
        f"  float buf_a[{max_hidden}];",
        f"  float buf_b[{max_hidden}];",
    ])

    in_var = "x_norm"
    for idx, (w, _) in enumerate(layers):
        out_dim, in_dim = w.shape
        is_last = idx == len(layers) - 1
        out_var = "action_out" if is_last else ("buf_a" if idx % 2 == 0 else "buf_b")
        apply_flag = "false" if is_last else "true"
        code.append(
            f"  bb8_dense({in_var}, {in_dim}, BB8_W{idx}, BB8_B{idx}, {out_var}, {out_dim}, {apply_flag});"
        )
        in_var = out_var

    code.extend([
        "  for (int i = 0; i < BB8_ACT_DIM; ++i) {",
        "    action_out[i] = bb8_clamp(action_out[i], BB8_ACT_LOW[i], BB8_ACT_HIGH[i]);",
        "  }",
        "}",
        "",
    ])

    OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
    OUT_HEADER.write_text("\n".join(code), encoding="utf-8")

    print("\n=== EXPORT PPO v3 -> C++ RÉUSSI ===")
    print(f"  Fichier généré    : {OUT_HEADER}")
    print(f"  Architecture      : {obs_dim} -> " + " -> ".join(str(w.shape[0]) for w, _ in layers) + f" ({act_name})")
    print(f"  Empreinte Flash   : {total_params * 4 / 1024.0:.1f} Ko ({total_params} floats)")
    print(f"  Écart max vs SB3  : {max_err:.2e} (sur 100 vecteurs aléatoires)")
    print(f"  Test ref action   : {np.round(test_act_ref, 4)}")


if __name__ == "__main__":
    main()