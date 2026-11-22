from pathlib import Path
import numpy as np
from build123d import (
    Box,
    CenterOf,
    Color,
    Compound,
    Cone,
    Cylinder,
    Pos,
    Rot,
    Sphere,
    export_step,
    export_stl,
)

# ==============================================================================
# 1. PARAMÈTRES GÉOMÉTRIQUES (en mm) ET DENSITÉS MATÉRIAUX (en kg/m³)
# ==============================================================================
R_SHELL = 250.0       # Rayon extérieur de la coque sphérique
T_SHELL = 2.0         # Épaisseur de coque (polycarbonate / ABS)
R_INNER = R_SHELL - T_SHELL

R_DOME = 115.0        # Rayon du dôme (tête)
T_DOME = 2.5          # Épaisseur de paroi du dôme
DOME_Z0 = 240.0       # Centre de courbure du dôme

STEER_PIVOT_Z = -45.0 # Hauteur du pivot de direction du pendule (sous l'axe Y)

DENSITY = {
    "abs": 1050.0,      # Coque, dôme, pièces plastiques
    "alu": 2700.0,      # Axe, châssis interne, mât, bras de pendule
    "steel": 7850.0,    # Lest principal + aimants
    "lipo": 2100.0,     # Pack batterie LiPo 4S
    "motor": 4800.0,    # Bloc moteur-réducteur (cuivre + acier)
    "pcb": 1850.0,      # Carte électronique (FR4 + composants)
}

EXPORT_DIR = Path("cad_export")
STL_DIR = EXPORT_DIR / "stl"
STEP_DIR = EXPORT_DIR / "step"


# ==============================================================================
# 2. UTILITAIRES : PROPRIÉTÉS INERTIELLES EXACTES (HYUGENS-STEINER)
# ==============================================================================
def compute_body_inertia(parts: list[tuple[object, float]]) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calcule la masse totale (kg), le centre de masse (m) et le tenseur d'inertie
    3x3 au centre de masse (kg·m²) à partir d'une liste de (forme_build123d, densité).
    """
    solids_data = []
    total_mass = 0.0
    weighted_com = np.zeros(3)

    for shape, rho in parts:
        for solid in shape.solids():
            vol_m3 = solid.volume * 1e-9
            mass = rho * vol_m3
            if mass <= 0:
                continue
            c = solid.center(CenterOf.MASS)
            com_m = np.array([c.X, c.Y, c.Z], dtype=float) * 1e-3
            inertia_local = np.array(solid.matrix_of_inertia, dtype=float) * rho * 1e-15

            solids_data.append((mass, com_m, inertia_local))
            total_mass += mass
            weighted_com += mass * com_m

    com_total = weighted_com / total_mass
    inertia_total = np.zeros((3, 3))

    for mass, com_m, inertia_local in solids_data:
        d = com_m - com_total
        # Théorème de Huygens-Steiner (signes hors-diagonale compatibles MuJoCo)
        steiner = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        inertia_total += inertia_local + steiner

    inertia_total = 0.5 * (inertia_total + inertia_total.T)
    return total_mass, com_total, inertia_total


def style_part(shape, label: str, rgba: tuple[float, float, float, float]):
    shape.label = label
    shape.color = Color(*rgba)
    return shape


# ==============================================================================
# 3. CONCEPTION DES 4 SOUS-ASSEMBLAGES CAO
# ==============================================================================
def build_shell() -> list[dict]:
    """Coque sphérique creuse avec paliers internes et 6 anneaux orange."""
    base_shell = Sphere(R_SHELL) - Sphere(R_INNER)

    # Paliers internes sur l'axe Y pour recevoir l'axe principal
    boss_pos = Pos(0, 235, 0) * Rot(90, 0, 0) * Cylinder(26, 28)
    boss_neg = Pos(0, -235, 0) * Rot(90, 0, 0) * Cylinder(26, 28)
    bosses = (boss_pos + boss_neg) & Sphere(R_SHELL - 0.5)
    axle_bore = Rot(90, 0, 0) * Cylinder(13, 498)
    shell_white = (base_shell + bosses) - axle_bore

    # 6 panneaux circulaires orange épousant exactement la courbure sphérique
    decal_layer = Sphere(R_SHELL + 0.8) - Sphere(R_SHELL - 0.4)
    ring_template = Pos(0, 0, 235) * (Cylinder(84, 55) - Cylinder(62, 55))
    orientations = [
        Rot(0, 0, 0),
        Rot(180, 0, 0),
        Rot(0, 90, 0),
        Rot(0, -90, 0),
        Rot(90, 0, 0),
        Rot(-90, 0, 0),
    ]
    orange_rings = Compound(
        label="shell_orange_panels",
        children=[(rot * ring_template) & decal_layer for rot in orientations],
    )

    # Anneaux gris techniques au centre des 6 panneaux
    detail_template = Pos(0, 0, 238) * (Cylinder(42, 50) - Cylinder(34, 50))
    gray_details = Compound(
        label="shell_gray_details",
        children=[(rot * detail_template) & decal_layer for rot in orientations],
    )

    return [
        {
            "name": "shell_white",
            "shape": style_part(shell_white, "shell_white", (0.96, 0.96, 0.96, 0.55)),
            "density": DENSITY["abs"],
            "rgba": (0.96, 0.96, 0.96, 0.55),  # Passer 0.55 -> 1.0 pour une coque opaque
        },
        {
            "name": "shell_orange",
            "shape": style_part(orange_rings, "shell_orange", (0.95, 0.45, 0.10, 0.95)),
            "density": DENSITY["abs"],
            "rgba": (0.95, 0.45, 0.10, 0.95),
        },
        {
            "name": "shell_details",
            "shape": style_part(gray_details, "shell_details", (0.55, 0.58, 0.62, 0.95)),
            "density": DENSITY["abs"],
            "rgba": (0.55, 0.58, 0.62, 0.95),
        },
    ]


def build_inner_frame() -> list[dict]:
    """Châssis interne : axe Y, berceau central, moteur, PCB/IMU et mât magnétique."""
    # Axe principal traversant + berceau central + chape de direction
    axle = Rot(90, 0, 0) * (Cylinder(12, 466) - Cylinder(9, 468))
    cradle = Box(96, 110, 64) - Box(76, 86, 44)
    clevis_l = Pos(28, 0, -40) * Box(12, 34, 30)
    clevis_r = Pos(-28, 0, -40) * Box(12, 34, 30)
    mast = Pos(0, 0, 128) * (Cylinder(10, 192) - Cylinder(7.5, 194))
    magnet_plate = Pos(0, 0, 222) * Cylinder(56, 8)
    frame_alu = axle + cradle + clevis_l + clevis_r + mast + magnet_plate

    # 3 aimants néodymes sur le plateau supérieur (sous la paroi interne de la coque)
    magnets = Compound(
        label="head_magnets",
        children=[
            Rot(0, 0, deg) * Pos(38, 0, 229) * Cylinder(12, 8)
            for deg in (0, 120, 240)
        ],
    )

    # Moteur d'entraînement principal (coaxial sur l'axe Y)
    drive_motor = (
    Pos(0, 58, 0) * Rot(90, 0, 0) * Cylinder(22, 40)
    + Pos(0, -58, 0) * Rot(90, 0, 0) * Cylinder(22, 40)
    )

    # Carte électronique (ESP32 + IMU centrale)
    pcb = Pos(0, 0, 36) * Box(68, 52, 4)

    return [
        {
            "name": "frame_alu",
            "shape": style_part(frame_alu, "frame_alu", (0.72, 0.74, 0.78, 1.0)),
            "density": DENSITY["alu"],
            "rgba": (0.72, 0.74, 0.78, 1.0),
        },
        {
            "name": "frame_magnets",
            "shape": style_part(magnets, "frame_magnets", (0.85, 0.87, 0.90, 1.0)),
            "density": DENSITY["steel"],
            "rgba": (0.85, 0.87, 0.90, 1.0),
        },
        {
            "name": "frame_motor",
            "shape": style_part(drive_motor, "frame_motor", (0.18, 0.20, 0.24, 1.0)),
            "density": DENSITY["motor"],
            "rgba": (0.18, 0.20, 0.24, 1.0),
        },
        {
            "name": "frame_pcb",
            "shape": style_part(pcb, "frame_pcb", (0.10, 0.55, 0.25, 1.0)),
            "density": DENSITY["pcb"],
            "rgba": (0.10, 0.55, 0.25, 1.0),
        },
    ]


def build_pendulum() -> list[dict]:
    """Pendule lesté articulé à z=-45 mm : bras alu, pack LiPo 4S et lest acier."""
    pivot_hub = Pos(0, 0, STEER_PIVOT_Z) * Rot(0, 90, 0) * Cylinder(14, 42)
    arm_1 = Pos(26, 0, -90) * Box(10, 26, 92)
    arm_2 = Pos(-26, 0, -90) * Box(10, 26, 92)
    tray_plate = Pos(0, 0, -130) * Box(142, 116, 6)
    pendulum_arm = pivot_hub + arm_1 + arm_2 + tray_plate

    # Pack batterie LiPo 4S posé sur le plateau
    lipo_pack = Pos(0, 0, -148) * Box(125, 95, 32)

    # Bloc de lest principal en acier (centre de gravité très bas)
    steel_ballast = Pos(0, 0, -180) * Box(140, 115, 34)

    return [
        {
            "name": "pendulum_arm",
            "shape": style_part(pendulum_arm, "pendulum_arm", (0.68, 0.70, 0.74, 1.0)),
            "density": DENSITY["alu"],
            "rgba": (0.68, 0.70, 0.74, 1.0),
        },
        {
            "name": "pendulum_lipo",
            "shape": style_part(lipo_pack, "pendulum_lipo", (0.12, 0.35, 0.80, 1.0)),
            "density": DENSITY["lipo"],
            "rgba": (0.12, 0.35, 0.80, 1.0),
        },
        {
            "name": "pendulum_ballast",
            "shape": style_part(steel_ballast, "pendulum_ballast", (0.20, 0.22, 0.26, 1.0)),
            "density": DENSITY["steel"],
            "rgba": (0.20, 0.22, 0.26, 1.0),
        },
    ]


def build_head() -> list[dict]:
    """Dôme (tête) épousant la sphère avec bandes orange, optique caméra et antennes."""
    outer_dome = (Pos(0, 0, DOME_Z0) * Sphere(R_DOME)) + (
        Pos(0, 0, DOME_Z0 - 8) * Cone(R_DOME + 4, R_DOME, 16)
    )
    # Évidement interne + découpe inférieure épousant la coque avec 4 mm de jeu
    dome_shell = outer_dome - Sphere(R_SHELL + 4.0) - (Pos(0, 0, DOME_Z0) * Sphere(R_DOME - T_DOME))

    # Couronne structurelle interne (porte-galets)
    base_ring = (
        Pos(0, 0, 260) * (Cylinder(112, 6) - Cylinder(60, 6))
    ) & (Pos(0, 0, DOME_Z0) * Sphere(R_DOME - 0.5))
    head_white = dome_shell + base_ring

    # Bandes décoratives orange sur le dôme
    dome_decal_layer = Pos(0, 0, DOME_Z0) * (Sphere(R_DOME + 0.7) - Sphere(R_DOME - 0.4))
    band_low = (Pos(0, 0, DOME_Z0 + 8) * Cylinder(125, 11)) & dome_decal_layer
    band_high = (Pos(0, 0, DOME_Z0 + 85) * Cylinder(125, 11)) & dome_decal_layer
    head_orange = band_low + band_high

    # Œil principal (caméra frontale orientée vers +X, inclinée à 24°) + capteur latéral
    eye_bezel = Pos(0, 0, DOME_Z0) * Rot(0, 66, 0) * Pos(0, 0, 111) * Cylinder(24, 14)
    holo_eye = Pos(0, 0, DOME_Z0) * Rot(0, 0, -32) * Rot(0, 74, 0) * Pos(0, 0, 113) * Cylinder(10, 10)
    head_bezel = eye_bezel + holo_eye

    eye_lens = Pos(0, 0, DOME_Z0) * Rot(0, 66, 0) * Pos(0, 0, 115) * Sphere(18)

    # Deux antennes arrière
    ant_long = Pos(-38, 22, 378) * Cylinder(3.0, 85)
    ant_short = Pos(-48, -16, 362) * Cylinder(2.2, 52)
    head_antennas = ant_long + ant_short

    return [
        {
            "name": "head_white",
            "shape": style_part(head_white, "head_white", (0.96, 0.96, 0.96, 1.0)),
            "density": DENSITY["abs"],
            "rgba": (0.96, 0.96, 0.96, 1.0),
        },
        {
            "name": "head_orange",
            "shape": style_part(head_orange, "head_orange", (0.95, 0.45, 0.10, 1.0)),
            "density": DENSITY["abs"],
            "rgba": (0.95, 0.45, 0.10, 1.0),
        },
        {
            "name": "head_bezel",
            "shape": style_part(head_bezel, "head_bezel", (0.35, 0.37, 0.40, 1.0)),
            "density": DENSITY["alu"],
            "rgba": (0.35, 0.37, 0.40, 1.0),
        },
        {
            "name": "head_lens",
            "shape": style_part(eye_lens, "head_lens", (0.05, 0.05, 0.07, 1.0)),
            "density": DENSITY["abs"],
            "rgba": (0.05, 0.05, 0.07, 1.0),
        },
        {
            "name": "head_antennas",
            "shape": style_part(head_antennas, "head_antennas", (0.82, 0.84, 0.88, 1.0)),
            "density": DENSITY["alu"],
            "rgba": (0.82, 0.84, 0.88, 1.0),
        },
    ]


# ==============================================================================
# 4. GÉNÉRATION DU FICHIER MUJOCO (bb8_cad.xml)
# ==============================================================================
def fmt_vec(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def fmt_fullinertia(I: np.ndarray) -> str:
    vals = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    return " ".join(f"{x:.6g}" for x in vals)


def write_mujoco_xml(bodies_meta: dict, output_path: Path):
    mesh_assets = []
    for body_data in bodies_meta.values():
        for part in body_data["parts"]:
            name = part["name"]
            mesh_assets.append(
                f'    <mesh name="{name}_mesh" file="{name}.stl" scale="0.001 0.001 0.001"/>'
            )

    def visual_geoms(body_key: str, indent: str) -> str:
        lines = []
        for part in bodies_meta[body_key]["parts"]:
            name = part["name"]
            rgba = " ".join(f"{c:.2f}" for c in part["rgba"])
            lines.append(
                f'{indent}<geom name="{name}_vis" type="mesh" mesh="{name}_mesh" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>'
            )
        return "\n".join(lines)

    def inertial_tag(body_key: str, indent: str) -> str:
        m = bodies_meta[body_key]["mass"]
        com = bodies_meta[body_key]["com"]
        I = bodies_meta[body_key]["inertia"]
        return (
            f'{indent}<inertial pos="{fmt_vec(com)}" mass="{m:.5f}" '
            f'fullinertia="{fmt_fullinertia(I)}"/>'
        )

    xml = f"""<mujoco model="bb8_cad">
  <compiler angle="degree" coordinate="local" meshdir="cad_export/stl"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.32 0.32 0.32" specular="0.25 0.25 0.25"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.88 0.88 0.88" rgb2="0.76 0.76 0.76" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.1"/>
{chr(10).join(mesh_assets)}
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="25 25 0.1" material="grid_mat"
          friction="1.0 0.02 0.005" contype="1" conaffinity="1"/>

    <!-- Coque sphérique externe (collision par sphère pure + visuel CAO exact) -->
    <body name="shell" pos="0 0 {R_SHELL * 1e-3:.4f}">
      <freejoint name="root"/>
{inertial_tag("shell", "      ")}
      <geom name="shell_collision" type="sphere" size="{R_SHELL * 1e-3:.4f}"
            rgba="1 1 1 0" friction="1.0 0.02 0.005" contype="1" conaffinity="1"/>
{visual_geoms("shell", "      ")}

      <!-- Châssis interne (moteur, PCB/IMU, mât magnétique) -->
      <body name="inner_frame" pos="0 0 0">
        <joint name="drive_joint" type="hinge" pos="0 0 0" axis="0 -1 0" damping="0.05"/>
{inertial_tag("inner_frame", "        ")}
{visual_geoms("inner_frame", "        ")}
        <site name="imu_site" pos="0 0 0.036" size="0.012" rgba="0.1 0.9 0.2 0.9"/>

        <!-- Pendule lesté (articulé sous l'axe Y à z = -45 mm) -->
        <body name="pendulum" pos="0 0 0">
          <joint name="steer_joint" type="hinge" pos="0 0 {STEER_PIVOT_Z * 1e-3:.4f}" axis="1 0 0"
                 limited="true" range="-35 35" damping="0.25"/>
{inertial_tag("pendulum", "          ")}
{visual_geoms("pendulum", "          ")}
        </body>

        <!-- Dôme / Tête (liaison magnétique virtuelle) -->
        <body name="head" pos="0 0 0">
          <joint name="head_yaw" type="hinge" pos="0 0 {R_SHELL * 1e-3:.4f}" axis="0 0 1" damping="0.03"/>
{inertial_tag("head", "          ")}
{visual_geoms("head", "          ")}
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="drive_motor" joint="drive_joint" gear="1" ctrllimited="true" ctrlrange="-5.5 5.5"/>    <position name="steer_servo" joint="steer_joint" kp="45" kv="5" ctrllimited="true" ctrlrange="-0.55 0.55"/>
    <position name="head_servo" joint="head_yaw" kp="2.5" kv="0.25" ctrllimited="true" ctrlrange="-3.14 3.14"/>
  </actuator>

  <sensor>
    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <framequat name="imu_quat" objtype="site" objname="imu_site"/>
    <jointvel name="drive_encoder_vel" joint="drive_joint"/>
    <jointpos name="steer_encoder_pos" joint="steer_joint"/>
  </sensor>
</mujoco>
"""
    output_path.write_text(xml, encoding="utf-8")


# ==============================================================================
# 5. EXÉCUTION PRINCIPALE
# ==============================================================================
def main():
    STL_DIR.mkdir(parents=True, exist_ok=True)
    STEP_DIR.mkdir(parents=True, exist_ok=True)

    print("Construction de la CAO paramétrique BB-8...")
    raw_bodies = {
        "shell": build_shell(),
        "inner_frame": build_inner_frame(),
        "pendulum": build_pendulum(),
        "head": build_head(),
    }

    bodies_meta = {}
    cad_subassemblies = []
    total_robot_mass = 0.0
    total_weighted_com = np.zeros(3)

    print("\n--- Bilan des masses et centres de gravité ---")
    for body_name, parts in raw_bodies.items():
        mass, com, inertia = compute_body_inertia([(p["shape"], p["density"]) for p in parts])
        bodies_meta[body_name] = {
            "parts": parts,
            "mass": mass,
            "com": com,
            "inertia": inertia,
        }
        total_robot_mass += mass
        total_weighted_com += mass * com

        print(
            f"  {body_name:12s} | masse = {mass:5.2f} kg | "
            f"CdG = ({com[0]*1000:+6.1f}, {com[1]*1000:+6.1f}, {com[2]*1000:+6.1f}) mm"
        )

        # Export STL individuel par pièce (pour le rendu multi-couleurs MuJoCo)
        for p in parts:
            export_stl(p["shape"], str(STL_DIR / f"{p['name']}.stl"), tolerance=0.15, angular_tolerance=0.2)

        sub_asm = Compound(label=body_name, children=[p["shape"] for p in parts])
        export_step(sub_asm, str(STEP_DIR / f"{body_name}.step"))
        cad_subassemblies.append(sub_asm)

    global_com = total_weighted_com / total_robot_mass
    print("-" * 62)
    print(
        f"  {'TOTAL BB-8':12s} | masse = {total_robot_mass:5.2f} kg | "
        f"CdG = ({global_com[0]*1000:+6.1f}, {global_com[1]*1000:+6.1f}, {global_com[2]*1000:+6.1f}) mm\n"
    )

    # Export de l'assemblage complet au format STEP industriel
    full_assembly = Compound(label="BB8_Assembly", children=cad_subassemblies)
    export_step(full_assembly, str(STEP_DIR / "bb8_assembly.step"))

    # Génération du fichier MuJoCo prêt à simuler
    write_mujoco_xml(bodies_meta, Path("bb8_cad.xml"))
    print("Fichiers générés :")
    print(f"  - CAO industrielle STEP : {STEP_DIR}/bb8_assembly.step")
    print(f"  - Maillages 3D STL      : {STL_DIR}/*.stl")
    print("  - Modèle MuJoCo         : bb8_cad.xml")

    # Affichage interactif dans l'extension VS Code OCP CAD Viewer
    try:
        from ocp_vscode import show
        show(full_assembly)
        print("\nAssemblage envoyé vers OCP CAD Viewer dans VS Code.")
    except Exception as exc:
        print(f"\nAperçu OCP CAD Viewer ignoré ({exc}). Ouvrez l'onglet OCP CAD Viewer puis relancez.")


if __name__ == "__main__":
    main()