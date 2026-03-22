#!/usr/bin/env python3
"""
Realistic drone keyboard controller — Isaac Sim standalone (no ROS).
  - Warehouse environment
  - Iris quadrotor spawned facing the gate

Controls:
  W / S    pitch fwd/back   → forward / backward
  A / D    roll left/right  → strafe left / right
  Q / E    yaw left / right
  Z        throttle UP   (release = hold altitude)
  X        throttle DOWN
  R        reset to origin
  Esc      quit

Run:
  cd ~/Isaacsimstandalone/isaacsim/_build/linux-x86_64/release
  ./python.sh /mnt/Research/NVIDIA/droneisaac/drone_keyboard_isaac.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math, time
import numpy as np
import carb, carb.input
import omni.appwindow
from pxr import PhysxSchema, Usd

from isaacsim.core.api import World
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, get_stage_units
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
IRIS_USD  = "/mnt/Research/NVIDIA/droneisaac/Collected_d3/iris.usd"
GATE_USD  = "/mnt/Research/NVIDIA/droneisaac/Collected_d3/Gate1.usdz"
BODY_PRIM = "/World/iris/body"
GATE_PRIM = "/World/Gate1"

assets_root = get_assets_root_path()
WAREHOUSE_USD = assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

# ---------------------------------------------------------------------------
# Flight tuning
# ---------------------------------------------------------------------------
MAX_PITCH      = math.radians(28)
MAX_ROLL       = math.radians(28)
PITCH_RATE     = math.radians(90)
ROLL_RATE      = math.radians(90)
YAW_RATE       = math.radians(80)
LEVEL_RATE     = 6.0        # self-leveling (s⁻¹)
MAX_SPEED      = 4.0        # m/s horizontal at full tilt
THROTTLE_ACCEL = 3.5        # m/s² vertical
THROTTLE_DAMP  = 0.85       # altitude-hold damping
DRAG           = 0.92       # horizontal drag

# Spawn: drone at origin, gate 6 m ahead (+X), drone faces gate (yaw=0 → +X)
START_POS  = np.array([0.0, 0.0, 1.5])
GATE_POS   = np.array([6.0, 0.0, 0.0])
GATE_SCALE = 1.0   # metres

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def disable_articulation_physics(prim_path: str):
    stage = get_current_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            attr = PhysxSchema.PhysxArticulationAPI(p).GetArticulationEnabledAttr()
            if attr:
                attr.Set(False)
            return

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)

# Warehouse environment (no physics interaction needed)
add_reference_to_stage(usd_path=WAREHOUSE_USD, prim_path="/World/Warehouse")

# Iris drone
add_reference_to_stage(usd_path=IRIS_USD, prim_path="/World/iris")
disable_articulation_physics("/World/iris")
body_xform = XformPrim(BODY_PRIM, reset_xform_op_properties=True)

# Gate — placed 6 m ahead of drone along +X, centered at drone height
add_reference_to_stage(usd_path=GATE_USD, prim_path=GATE_PRIM)
gate_xform = XformPrim(GATE_PRIM, reset_xform_op_properties=True)
gate_quat  = euler_angles_to_quat(np.array([0.0, 0.0, math.pi / 2]))  # gate opening faces drone
gate_xform.set_world_poses(
    positions    = np.array([GATE_POS]),
    orientations = np.array([gate_quat]),
)
gate_xform.set_local_scales(np.array([[GATE_SCALE, GATE_SCALE, GATE_SCALE]]))

# Camera: behind and above the drone, looking toward the gate
set_camera_view(
    eye    = [-3.0, 0.0, 3.0],
    target = [GATE_POS[0], GATE_POS[1], 1.5],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()

# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
kb          = omni.appwindow.get_default_app_window().get_keyboard()
input_iface = carb.input.acquire_input_interface()
K           = carb.input.KeyboardInput

def key(k) -> bool:
    return input_iface.get_keyboard_value(kb, k) > 0

# ---------------------------------------------------------------------------
# Flight state
# ---------------------------------------------------------------------------
pos   = START_POS.copy().astype(float)
vel   = np.zeros(3)
pitch = 0.0   # rad
roll  = 0.0   # rad
yaw   = 0.0   # rad (0 = facing +X = facing gate)

def reset():
    global pos, vel, pitch, roll, yaw
    pos[:] = START_POS; vel[:] = 0.0
    pitch = roll = yaw = 0.0

def apply_pose():
    su   = get_stage_units()
    quat = euler_angles_to_quat(np.array([roll, pitch, yaw]))
    body_xform.set_world_poses(
        positions    = np.array([pos / su]),
        orientations = np.array([quat]),
    )

apply_pose()

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
print("\n" + "="*52)
print("  Realistic Drone  —  Warehouse + Gate")
print("="*52)
print("  W / S    pitch fwd/back  (forward/backward)")
print("  A / D    roll left/right (strafe)")
print("  Q / E    yaw")
print("  Z / X    throttle up / down  (Z holds altitude)")
print("  R  reset  |  Esc  quit")
print("="*52 + "\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
prev_t = time.time()

while simulation_app.is_running():
    now = time.time()
    dt  = min(now - prev_t, 0.05)
    prev_t = now

    if key(K.ESCAPE):
        break

    if key(K.R):
        reset(); apply_pose()
        world.step(render=True)
        continue

    # --- Yaw ---
    if key(K.Q):   yaw += YAW_RATE * dt
    if key(K.E):   yaw -= YAW_RATE * dt
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    # --- Pitch (W/S) — self-levels on release ---
    if key(K.W):
        pitch = min(MAX_PITCH, pitch + PITCH_RATE * dt)
    elif key(K.S):
        pitch = max(-MAX_PITCH, pitch - PITCH_RATE * dt)
    else:
        pitch *= math.exp(-LEVEL_RATE * dt)

    # --- Roll (A/D) — self-levels on release ---
    if key(K.A):
        roll = max(-MAX_ROLL, roll - ROLL_RATE * dt)
    elif key(K.D):
        roll = min(MAX_ROLL, roll + ROLL_RATE * dt)
    else:
        roll *= math.exp(-LEVEL_RATE * dt)

    # --- Throttle / altitude hold ---
    if key(K.Z):
        vel[2] += THROTTLE_ACCEL * dt
    elif key(K.X):
        vel[2] -= THROTTLE_ACCEL * dt
    else:
        vel[2] *= THROTTLE_DAMP ** dt   # altitude hold: damp to hover

    # --- Horizontal from tilt (body → world) ---
    cy, sy = math.cos(yaw), math.sin(yaw)
    tvx =  math.sin(pitch) * MAX_SPEED
    tvy = -math.sin(roll)  * MAX_SPEED
    tvx_w = cy * tvx - sy * tvy
    tvy_w = sy * tvx + cy * tvy

    alpha  = 1.0 - DRAG ** dt
    vel[0] += (tvx_w - vel[0]) * alpha
    vel[1] += (tvy_w - vel[1]) * alpha

    # --- Integrate ---
    pos    += vel * dt
    pos[2]  = max(0.05, pos[2])

    apply_pose()
    world.step(render=True)

world.stop()
simulation_app.close()
