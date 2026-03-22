#!/usr/bin/env python3
"""
Drone keyboard controller + dual-camera recorder — Isaac Sim standalone.

Records one episode with both:
  - FPV  camera  (nose-mounted, 0.6 m ahead of drone)
  - Follow camera (3rd-person, behind + above drone)

Controls:
  W / S    pitch fwd/back  → forward / backward
  A / D    roll left/right → strafe
  Q / E    yaw left / right
  Z        throttle UP   (release = altitude hold)
  X        throttle DOWN
  F        toggle recording ON / OFF  (compiles MP4 on stop)
  R        reset to origin
  Esc      quit

Output: /mnt/Research/NVIDIA/droneisaac/recordings/<timestamp>/
  fpv_<ts>.mp4        — first-person view
  follow_<ts>.mp4     — third-person follow cam
  frames/fpv/         — raw PNG frames (FPV)
  frames/follow/      — raw PNG frames (follow)
  trajectory.csv

Run:
  cd ~/Isaacsimstandalone/isaacsim/_build/linux-x86_64/release
  ./python.sh /mnt/Research/NVIDIA/droneisaac/drone_record_isaac.py
"""

from isaacsim import SimulationApp
HEADLESS = True   # set False for interactive window
simulation_app = SimulationApp({"headless": HEADLESS, "width": 1280, "height": 720})

import csv, json, math, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import carb, carb.input, carb.settings
import omni.appwindow
import omni.replicator.core as rep

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, get_stage_units
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IRIS_USD  = "/mnt/Research/NVIDIA/droneisaac/Collected_d3/iris.usd"
GATE_USD  = "/mnt/Research/NVIDIA/droneisaac/Collected_d3/CustomGate.usda"
ASSETS    = get_assets_root_path()
WH_USD    = ASSETS + "/Isaac/Environments/Hospital/hospital.usd"

BODY_PATH = "/World/iris"
GATE_PATH = "/World/Gate1"
FPV_CAM   = "/World/FPVCamera"
FOL_CAM   = "/World/FollowCamera"

REC_ROOT  = Path("/mnt/Research/NVIDIA/droneisaac/recordings")
REC_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera / recording config
# ---------------------------------------------------------------------------
CAM_W, CAM_H = 854, 480
SIM_HZ        = 60
RECORD_FPS    = 30
CAP_INTERVAL  = max(1, SIM_HZ // RECORD_FPS)

# ---------------------------------------------------------------------------
# Flight tuning
# ---------------------------------------------------------------------------
MAX_PITCH      = math.radians(35)
MAX_ROLL       = math.radians(35)
PITCH_RATE     = math.radians(110)
ROLL_RATE      = math.radians(110)
YAW_RATE       = math.radians(90)
LEVEL_RATE     = 5.0
MAX_SPEED      = 10.0
THROTTLE_ACCEL = 7.0
THROTTLE_DAMP  = 0.85
DRAG           = 0.94          # aerodynamic drag per second
GRAVITY        = 2.5           # m/s² effective gravity drone must fight
MOTOR_LAG      = 0.12          # seconds — motor response lag (smooths velocity changes)
HOVER_HEIGHT   = 1.5
START_POS      = np.array([ 0.0,  0.0,  1.5])  # drone at origin

# Gate: 6 m ahead, opening faces drone (x=90, y=90, z=0 degrees Euler XYZ)
GATE_POS       = np.array([5.0, 0.0, 0.0])   # gate at x=5
GATE_ROT_DEG   = (0.0, 0.0, 0.0)      # gate pre-aligned, opening faces +X

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euler_to_quat_wxyz(rx_deg, ry_deg, rz_deg):
    rx = math.radians(rx_deg); ry = math.radians(ry_deg); rz = math.radians(rz_deg)
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    return (cx*cy*cz + sx*sy*sz,
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz)   # w, x, y, z


def qrot(q_wxyz, v):
    """Rotate vector v by quaternion (w,x,y,z)."""
    w, x, y, z = [float(a) for a in q_wxyz]
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return R @ np.asarray(v, float)


def look_at(eye, tgt, up=None):
    up = np.array([0., 0., 1.]) if up is None else np.asarray(up, float)
    eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
    if np.linalg.norm(tgt - eye) < 1e-6:
        tgt = eye + np.array([1., 0., 0.])
    mat = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(*up))
    return mat.GetInverse()


def make_kinematic(prim_path):
    stage = get_current_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Set(True)
        if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api  = PhysxSchema.PhysxArticulationAPI(p)
            attr = api.GetArticulationEnabledAttr()
            if attr:
                attr.Set(False)
        # Disable rotor joints so they don't conflict with kinematic body
        if p.GetTypeName() in ("PhysicsJoint", "PhysicsRevoluteJoint", "PhysicsFixedJoint"):
            if p.HasAttribute("physics:jointEnabled"):
                p.GetAttribute("physics:jointEnabled").Set(False)


def move_body(stage, pos, yaw, pitch=0.0, roll=0.0):
    """Move /World/iris (whole drone incl. rotors) using a full rotation matrix."""
    prim = stage.GetPrimAtPath(BODY_PATH)
    if not prim.IsValid():
        return
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch),  math.sin(pitch)
    cr, sr = math.cos(roll),   math.sin(roll)
    # ZYX rotation: yaw * pitch * roll
    m = Gf.Matrix4d(
        cy*cp,          sy*cp,          -sp,     0,
        cy*sp*sr-sy*cr, sy*sp*sr+cy*cr,  cp*sr,  0,
        cy*sp*cr+sy*sr, sy*sp*cr-cy*sr,  cp*cr,  0,
        pos[0],         pos[1],          pos[2], 1,
    )
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


FPV_TILT_DEG = 5.0   # slight upward tilt — keeps gate in frame during approach

def update_fpv_camera(stage, pos, yaw, pitch=0.0, speed=0.0):
    """FPV: nose-mounted, slight upward tilt, subtle vibration at speed."""
    cp  = stage.GetPrimAtPath(FPV_CAM)
    if not cp.IsValid():
        return
    q   = euler_to_quat_wxyz(0, 0, math.degrees(yaw))
    fwd = qrot(q, [1., 0., 0.])
    vib = (np.random.randn(3) * 0.004 * speed) if speed > 0.5 else np.zeros(3)
    eye = pos + fwd * 0.15 + np.array([0., 0., 0.06]) + vib
    tilt_rad = math.radians(FPV_TILT_DEG)
    look = fwd * math.cos(tilt_rad) + np.array([0., 0., 1.]) * math.sin(tilt_rad)
    tgt  = eye + look * 15.0
    xf  = UsdGeom.Xformable(cp)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(look_at(eye, tgt))


def update_follow_camera(stage, pos, yaw):
    """Follow cam: 6 m behind, 1.5 m above, looking at drone."""
    cp  = stage.GetPrimAtPath(FOL_CAM)
    if not cp.IsValid():
        return
    q    = euler_to_quat_wxyz(0, 0, math.degrees(yaw))
    fwd  = qrot(q, [1., 0., 0.])
    eye  = pos - fwd * 6.0 + np.array([0., 0., 1.5])
    tgt  = pos + np.array([0., 0., 0.1])
    xf   = UsdGeom.Xformable(cp)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(look_at(eye, tgt))


# ---------------------------------------------------------------------------
# Camera setup (replicator)
# ---------------------------------------------------------------------------
_fpv_rp  = _fpv_rgb  = None
_fol_rp  = _fol_rgb  = None


def setup_cameras():
    global _fpv_rp, _fpv_rgb, _fol_rp, _fol_rgb
    stage = get_current_stage()

    for path in (FPV_CAM, FOL_CAM):
        prim = stage.DefinePrim(path, "Camera")
        cam  = UsdGeom.Camera(prim)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(11.787)   # 20.955 * (1080/1920) for 16:9
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 2000.0))

    _fpv_rp  = rep.create.render_product(FPV_CAM, (CAM_W, CAM_H))
    _fpv_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    _fpv_rgb.attach([_fpv_rp])

    _fol_rp  = rep.create.render_product(FOL_CAM, (CAM_W, CAM_H))
    _fol_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    _fol_rgb.attach([_fol_rp])


def capture_frame(annotator):
    data = annotator.get_data()
    if data is None or data.size == 0:
        return None
    return data[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
def add_lighting(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
    dome.CreateIntensityAttr(3000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Light/Sun")
    sun.CreateIntensityAttr(4500.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.90))
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 20.0))


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, label=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = label if label else ts
        self.dir  = REC_ROOT / name
        self.fpv_dir = self.dir / "frames" / "fpv"
        self.fol_dir = self.dir / "frames" / "follow"
        self.fpv_dir.mkdir(parents=True, exist_ok=True)
        self.fol_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "trajectory.csv"
        self._csv  = open(self.csv_path, "w", newline="")
        self._wr   = csv.writer(self._csv)
        self._wr.writerow(["frame","t","x","y","z","yaw","pitch","roll",
                           "vx","vy","vz"])
        self.n = 0
        self.ts = ts
        print(f"[REC] session started → {self.dir}")

    def record(self, fpv_frame, fol_frame, pos, yaw, pitch, roll, vel):
        t = self.n / RECORD_FPS
        if fpv_frame is not None:
            self._save_png(fpv_frame, self.fpv_dir / f"frame_{self.n:06d}.png")
        if fol_frame is not None:
            self._save_png(fol_frame, self.fol_dir / f"frame_{self.n:06d}.png")
        self._wr.writerow([self.n, f"{t:.4f}",
                           f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                           f"{yaw:.4f}", f"{pitch:.4f}", f"{roll:.4f}",
                           f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}"])
        self.n += 1

    def _save_png(self, frame, path):
        try:
            import PIL.Image
            PIL.Image.fromarray(frame).save(str(path))
        except ImportError:
            h, w = frame.shape[:2]
            with open(str(path).replace(".png", ".ppm"), "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(frame.tobytes())

    def finish(self):
        self._csv.close()
        dur = self.n / RECORD_FPS
        # Write metadata
        meta = {"fps": RECORD_FPS, "frames": self.n,
                 "duration_sec": round(dur, 2),
                 "resolution": [CAM_W, CAM_H],
                 "timestamp": self.ts}
        with open(self.dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[REC] {self.n} frames ({dur:.1f}s) saved to {self.dir}")
        self._compile_video("fpv")
        self._compile_video("follow")

    def _compile_video(self, cam_name):
        frames_dir = self.dir / "frames" / cam_name
        out_mp4    = self.dir / f"{cam_name}_{self.ts}.mp4"
        pattern    = str(frames_dir / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(RECORD_FPS),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"[REC] {cam_name} video → {out_mp4}")
        else:
            print(f"[REC] ffmpeg failed for {cam_name}: {result.stderr.decode()[:200]}")


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
stage = get_current_stage()

add_lighting(stage)
add_reference_to_stage(usd_path=WH_USD,  prim_path="/World/Warehouse")
add_reference_to_stage(usd_path=IRIS_USD, prim_path="/World/iris")
make_kinematic("/World/iris")

# Gate — orientation x=90, y=90, z=0 degrees
add_reference_to_stage(usd_path=GATE_USD, prim_path=GATE_PATH)
gate_prim = stage.GetPrimAtPath(GATE_PATH)
gate_xf   = UsdGeom.Xformable(gate_prim)
gate_xf.ClearXformOpOrder()
gate_translate_op = gate_xf.AddTranslateOp()
gate_translate_op.Set(Gf.Vec3d(*GATE_POS.tolist()))
qw, qx, qy, qz = euler_to_quat_wxyz(*GATE_ROT_DEG)
gate_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
make_kinematic(GATE_PATH)

def move_gate(pos):
    """Reposition the gate to a new XYZ (Z=0 keeps bottom at floor)."""
    gate_translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

setup_cameras()
world.reset()
world.play()

# Warmup renderer
print("Warming up renderer (40 frames)...", flush=True)
for _ in range(40):
    world.step(render=True)
# Prime replicator
print("Priming replicator...", flush=True)
for _ in range(6):
    rep.orchestrator.step(rt_subframes=4)
    world.step(render=True)
print("Scene ready. Fly the drone and press F to record!", flush=True)

# Capture FPV reference screenshot before any recording
rep.orchestrator.step(rt_subframes=4)
world.step(render=True)
_ref_frame = capture_frame(_fpv_rgb)
if _ref_frame is not None:
    _ref_path = REC_ROOT / "fpv_reference.jpg"
    try:
        import PIL.Image
        PIL.Image.fromarray(_ref_frame).save(str(_ref_path))
    except ImportError:
        h, w = _ref_frame.shape[:2]
        with open(str(_ref_path).replace(".jpg", ".ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(_ref_frame.tobytes())
    print(f"[REF] FPV reference screenshot → {_ref_path}", flush=True)

# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._held = set()
    def on_event(self, ev, *a, **kw):
        if ev.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._held.add(ev.input)
        elif ev.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(ev.input)
        return True
    def held(self, k):
        return k in self._held

K           = carb.input.KeyboardInput
appwindow   = omni.appwindow.get_default_app_window()
keyboard_hw = appwindow.get_keyboard()
input_iface = carb.input.acquire_input_interface()
ks          = KeyState()
_sub        = input_iface.subscribe_to_keyboard_events(keyboard_hw, ks.on_event)

# ---------------------------------------------------------------------------
# Flight state
# ---------------------------------------------------------------------------
pos   = START_POS.copy().astype(float)
vel   = np.zeros(3)
pitch = 0.0
roll  = 0.0
yaw   = 0.0

def reset():
    global pos, vel, pitch, roll, yaw
    pos[:] = START_POS; vel[:] = 0.0
    pitch = roll = yaw = 0.0

# Place drone at start
move_body(stage, pos, yaw)
update_fpv_camera(stage, pos, yaw)
update_follow_camera(stage, pos, yaw)

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
print("\n" + "="*52, flush=True)
print("  Drone Recorder  —  Warehouse + Gate + Dual Cam", flush=True)
print("="*52, flush=True)
print("  W/S  pitch fwd/back   A/D  roll left/right", flush=True)
print("  Q/E  yaw              Z/X  throttle up/down", flush=True)
print("  F    toggle recording (compiles MP4 on stop)", flush=True)
print("  T    auto-pilot: fly through gate + auto-record", flush=True)
print("  Space  toggle hover hold (also auto-activates after gate)", flush=True)
print("  R    reset   |   Esc  quit", flush=True)
print("="*52 + "\n", flush=True)

# ---------------------------------------------------------------------------
# Auto-pilot trajectory
# ---------------------------------------------------------------------------
# Waypoints: (x, y, z) — drone flies through gate center at (6, 0, 1.15+1.0=1.15 inner center)
# Gate inner opening center Z = 0.15 + 1.0 = 1.15m, but HOVER_HEIGHT=1.5 is fine (within 2m opening)
# ---------------------------------------------------------------------------
# 3 scenarios: drone always at (0,0,1.5), GATE moves left/right/ahead
# ---------------------------------------------------------------------------
SCENARIOS = {
    0: {   # Gate ahead at (5, 0, 0)
        "name":      "GATE AHEAD",
        "label":     "hospital_normal",
        "gate_pos":  np.array([5.0,  0.0,  0.0]),
        "start":     np.array([0.0,  0.0,  1.5]),
        "start_yaw": math.atan2(0.0, 5.0),   # 0° — faces +X
        "waypoints": [
            np.array([0.0,  0.0,  1.5]),   # hover at origin
            np.array([3.0,  0.0,  1.5]),   # approach
            np.array([5.0,  0.0,  1.5]),   # gate center ← mid image
            np.array([7.0,  0.0,  1.5]),   # 2m past gate ← stop
        ],
        "gate_wp": 2,
    },
    1: {   # Gate to the RIGHT at (5, 3, 0) — drone angles right
        "name":      "GATE RIGHT",
        "label":     "hospital_right",
        "gate_pos":  np.array([5.0,  3.0,  0.0]),
        "start":     np.array([0.0,  0.0,  1.5]),
        "start_yaw": math.atan2(3.0, 5.0),   # ~31° — faces gate
        "waypoints": [
            np.array([0.0,  0.0,  1.5]),   # hover at origin
            np.array([3.0,  1.5,  1.5]),   # angle toward gate
            np.array([5.0,  3.0,  1.5]),   # gate center ← mid image
            np.array([7.0,  3.0,  1.5]),   # 2m past gate ← stop
        ],
        "gate_wp": 2,
    },
    2: {   # Gate to the LEFT at (5, -3, 0) — drone angles left
        "name":      "GATE LEFT",
        "label":     "hospital_left",
        "gate_pos":  np.array([5.0, -3.0,  0.0]),
        "start":     np.array([0.0,  0.0,  1.5]),
        "start_yaw": math.atan2(-3.0, 5.0),  # ~-31° — faces gate
        "waypoints": [
            np.array([0.0,  0.0,  1.5]),   # hover at origin
            np.array([3.0, -1.5,  1.5]),   # angle toward gate
            np.array([5.0, -3.0,  1.5]),   # gate center ← mid image
            np.array([7.0, -3.0,  1.5]),   # 2m past gate ← stop
        ],
        "gate_wp": 2,
    },
}
AUTO_SPEED     = 2.0   # slower for more frames
AUTO_SPEED_MAX = 3.5   # gate pass speed
AUTO_WP_RADIUS = 0.5
_mid_saved     = False
auto_scenario  = 0   # cycles 0→1→2→0 each T press

# Active waypoints (set when T is pressed)
AUTO_WAYPOINTS = SCENARIOS[0]["waypoints"]
AUTO_GATE_WP   = SCENARIOS[0]["gate_wp"]

auto_mode         = False
auto_wp_idx       = 0
_t_prev_auto      = False
hover_mode        = False
hover_pos         = None
_sp_prev          = False

# Auto-start autopilot when running headless
if HEADLESS:
    sc = SCENARIOS[0]
    AUTO_WAYPOINTS[:] = sc["waypoints"]
    AUTO_GATE_WP = sc["gate_wp"]
    move_gate(sc["gate_pos"])
    pos[:] = sc["start"]; vel[:] = 0.0
    pitch = roll = 0.0; yaw = sc["start_yaw"]
    move_body(stage, pos, yaw)
    update_fpv_camera(stage, pos, yaw)
    update_follow_camera(stage, pos, yaw)
    auto_mode         = False   # warmup block needs auto_mode=False + hover_mode=True
    auto_wp_idx       = 0
    auto_hover_frames = 0
    auto_end_hover    = -1
    hover_mode        = True
    hover_pos         = pos.copy()
    print(f">> [HEADLESS] Auto-start SCENARIO 0: {sc['name']} — gate at {sc['gate_pos'][:2]}", flush=True)
auto_hover_frames = 0
AUTO_HOVER_WARMUP = SIM_HZ * 2   # 2s hover before recording starts
auto_end_hover    = -1            # countdown after gate pass (-1 = inactive)
AUTO_END_HOVER    = SIM_HZ * 2   # 2s hover past gate before ending recording

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
recording  = False
session    = None
_f_prev    = False
sim_step   = 0
prev_t     = time.time()

while simulation_app.is_running():
    world.step(render=True)

    if ks.held(K.ESCAPE):
        break

    now = time.time()
    dt  = min(now - prev_t, 0.05)
    prev_t = now

    # --- F: toggle recording ---
    f_now = ks.held(K.F)
    if f_now and not _f_prev:
        if not recording:
            session   = Session()
            recording = True
            print(">> REC ON  (press F to stop)", flush=True)
        else:
            if session:
                session.finish()
                session = None
            recording = False
            print(">> REC OFF — compiling videos...", flush=True)
    _f_prev = f_now

    # --- Space: toggle hover hold ---
    sp_now = ks.held(K.SPACE)
    if sp_now and not _sp_prev:
        hover_mode = not hover_mode
        if hover_mode:
            hover_pos = pos.copy()
            vel[:] = 0.0
            print(f">> HOVER HOLD at {hover_pos}", flush=True)
        else:
            print(">> HOVER HOLD released", flush=True)
    _sp_prev = sp_now

    # --- R: reset ---
    if ks.held(K.R):
        reset()
        auto_mode  = False
        hover_mode = False

    # --- T: toggle auto-pilot ---
    t_now = ks.held(K.T)
    if t_now and not _t_prev_auto:
        auto_mode   = not auto_mode
        auto_wp_idx = 0
        if auto_mode:
            sc = SCENARIOS[auto_scenario]
            AUTO_WAYPOINTS[:] = sc["waypoints"]
            AUTO_GATE_WP = sc["gate_wp"]
            # Move gate to scenario position
            move_gate(sc["gate_pos"])
            # Move drone to scenario start, facing the gate
            pos[:] = sc["start"]; vel[:] = 0.0
            pitch = roll = 0.0; yaw = sc["start_yaw"]
            move_body(stage, pos, yaw)
            update_fpv_camera(stage, pos, yaw)
            update_follow_camera(stage, pos, yaw)
            _mid_saved        = False
            auto_hover_frames = 0
            auto_end_hover    = -1
            hover_mode        = True
            hover_pos         = pos.copy()
            print(f">> SCENARIO {auto_scenario}: {sc['name']} — gate at {sc['gate_pos'][:2]} — hovering 2s...", flush=True)
        else:
            print(">> AUTO-PILOT OFF", flush=True)
    _t_prev_auto = t_now

    if auto_mode:
        # Advance to next waypoint if close enough
        wp = AUTO_WAYPOINTS[auto_wp_idx]
        if np.linalg.norm(wp - pos) < AUTO_WP_RADIUS and auto_wp_idx < len(AUTO_WAYPOINTS) - 1:
            auto_wp_idx += 1
            wp = AUTO_WAYPOINTS[auto_wp_idx]
        dist = np.linalg.norm(wp - pos)   # distance to CURRENT (possibly advanced) waypoint

        # Direction toward waypoint
        to_wp   = wp - pos
        to_wp_dist = np.linalg.norm(to_wp)
        if to_wp_dist > 0.01:
            dir_n = to_wp / to_wp_dist
        else:
            dir_n = np.zeros(3)

        # Variable speed
        is_last = (auto_wp_idx == len(AUTO_WAYPOINTS) - 1)
        is_gate = (auto_wp_idx == len(AUTO_WAYPOINTS) - 2)  # waypoint just before last = gate
        if is_last:
            t_speed = max(0.5, AUTO_SPEED * min(1.0, to_wp_dist / 1.5))
        elif is_gate:
            t_speed = AUTO_SPEED_MAX
        else:
            t_speed = AUTO_SPEED

        target_vel = dir_n * t_speed
        vel += (target_vel - vel) * min(1.0, 6.0 * dt)
        vel[2] -= GRAVITY * dt * 0.15

        spd_h = np.linalg.norm(vel[:2])
        pitch = max(-MAX_PITCH, min(MAX_PITCH, math.radians(-spd_h * 2.8)))
        roll  = max(-MAX_ROLL,  min(MAX_ROLL,  math.radians(vel[1] * 3.2)))
        if spd_h > 0.3:
            yaw = math.atan2(vel[1], vel[0])

        # Save mid image when passing gate center (wp AUTO_GATE_WP)
        if (not _mid_saved and auto_wp_idx >= AUTO_GATE_WP
                and recording and session):
            rep.orchestrator.step(rt_subframes=4)
            mid_frame = capture_frame(_fpv_rgb)
            if mid_frame is not None:
                session._save_png(mid_frame, session.dir / "mid_image.png")
                print("[REC] mid_image saved (gate passage)", flush=True)
            _mid_saved = True

        # Past gate — stop and hover
        if is_last and dist < AUTO_WP_RADIUS and auto_end_hover == -1:
            auto_mode      = False
            auto_end_hover = 0
            hover_mode     = True
            hover_pos      = pos.copy()
            vel[:] = 0.0
            print(f">> Past gate — hovering {AUTO_END_HOVER//SIM_HZ}s before end...", flush=True)

    # --- Post-gate hover countdown ---
    if auto_end_hover >= 0:
        auto_end_hover += 1
        if auto_end_hover >= AUTO_END_HOVER:
            auto_end_hover = -1
            hover_mode     = False
            print(f">> SCENARIO {auto_scenario} complete", flush=True)
            if recording and session:
                rep.orchestrator.step(rt_subframes=4)
                end_frame = capture_frame(_fpv_rgb)   # FPV POV at stop point
                if end_frame is not None:
                    session._save_png(end_frame, session.dir / "end_image.png")
                    print("[REC] end_image saved (follow cam)", flush=True)
                session.finish()
                session = None
                recording = False
                print(">> REC OFF — compiling videos...", flush=True)
            # Auto-chain: advance to next scenario
            auto_scenario = (auto_scenario + 1) % len(SCENARIOS)
            if auto_scenario != 0:
                # Start next scenario automatically
                sc = SCENARIOS[auto_scenario]
                AUTO_WAYPOINTS[:] = sc["waypoints"]
                AUTO_GATE_WP = sc["gate_wp"]
                # Move gate to new position
                move_gate(sc["gate_pos"])
                pos[:] = sc["start"]; vel[:] = 0.0
                pitch = roll = 0.0; yaw = sc["start_yaw"]
                move_body(stage, pos, yaw)
                update_fpv_camera(stage, pos, yaw)
                update_follow_camera(stage, pos, yaw)
                _mid_saved        = False
                auto_hover_frames = 0
                hover_mode        = True
                hover_pos         = pos.copy()
                auto_mode         = False
                print(f">> AUTO-CHAIN → SCENARIO {auto_scenario}: {sc['name']} — gate at {sc['gate_pos'][:2]}", flush=True)
            else:
                print(">> ALL SCENARIOS COMPLETE", flush=True)
                hover_mode = True
                hover_pos  = pos.copy()

    elif hover_mode and hover_pos is not None:
        # --- Hover hold: PD controller toward locked position ---
        err = hover_pos - pos
        vel += err * 6.0 * dt
        vel *= (0.85 ** dt)
        pitch = max(-MAX_PITCH, min(MAX_PITCH, math.radians(-vel[0] * 1.5)))
        roll  = max(-MAX_ROLL,  min(MAX_ROLL,  math.radians( vel[1] * 1.5)))

        # Warmup hover before auto-pilot starts recording
        if auto_mode is False and not recording and auto_hover_frames >= 0:
            auto_hover_frames += 1
            if auto_hover_frames >= AUTO_HOVER_WARMUP:
                # Hover stabilized — start recording + save start image
                rep.orchestrator.step(rt_subframes=4)
                world.step(render=True)
                start_frame = capture_frame(_fpv_rgb)
                session   = Session(label=SCENARIOS[auto_scenario]["label"])
                recording = True
                if start_frame is not None:
                    session._save_png(start_frame, session.dir / "start_image.png")
                    print("[REC] start_image saved from hover", flush=True)
                print(">> REC ON — launching autopilot", flush=True)
                auto_mode        = True
                auto_wp_idx      = 0
                hover_mode       = False
                auto_hover_frames = -1  # disable warmup countdown

    else:
        # --- Manual: Yaw ---
        if ks.held(K.Q): yaw += YAW_RATE * dt
        if ks.held(K.E): yaw -= YAW_RATE * dt
        yaw = (yaw + math.pi) % (2*math.pi) - math.pi

        # --- Pitch (W/S), self-levels ---
        if ks.held(K.W):
            pitch = min(MAX_PITCH, pitch + PITCH_RATE * dt)
        elif ks.held(K.S):
            pitch = max(-MAX_PITCH, pitch - PITCH_RATE * dt)
        else:
            pitch *= math.exp(-LEVEL_RATE * dt)

        # --- Roll (A/D), self-levels ---
        if ks.held(K.A):
            roll = max(-MAX_ROLL, roll - ROLL_RATE * dt)
        elif ks.held(K.D):
            roll = min(MAX_ROLL, roll + ROLL_RATE * dt)
        else:
            roll *= math.exp(-LEVEL_RATE * dt)

        # --- Throttle (gravity always pulls down) ---
        if ks.held(K.Z):
            vel[2] += THROTTLE_ACCEL * dt
        elif ks.held(K.X):
            vel[2] -= THROTTLE_ACCEL * dt
        else:
            vel[2] -= GRAVITY * dt          # gravity drags altitude down
            vel[2] *= (THROTTLE_DAMP ** dt) # damping keeps hover manageable

        # --- Horizontal from tilt with motor lag ---
        any_move = ks.held(K.W) or ks.held(K.S) or ks.held(K.A) or ks.held(K.D)
        cy, sy = math.cos(yaw), math.sin(yaw)
        tvx    =  math.sin(pitch) * MAX_SPEED
        tvy    = -math.sin(roll)  * MAX_SPEED
        tvx_w  = cy*tvx - sy*tvy
        tvy_w  = sy*tvx + cy*tvy
        lag    = 1.0 - math.exp(-dt / MOTOR_LAG)
        if any_move:
            drag_alpha = 1.0 - DRAG ** dt
            vel[0] += (tvx_w - vel[0]) * drag_alpha * lag
            vel[1] += (tvy_w - vel[1]) * drag_alpha * lag
        else:
            vel[0] *= (DRAG ** dt)
            vel[1] *= (DRAG ** dt)

    pos   += vel * dt
    pos[2] = max(0.05, pos[2])

    # --- Move drone + cameras ---
    move_body(stage, pos, yaw, pitch, roll)
    update_fpv_camera(stage, pos, yaw, pitch, speed=float(np.linalg.norm(vel)))
    update_follow_camera(stage, pos, yaw)

    # --- Capture frames at RECORD_FPS ---
    if recording and session and (sim_step % CAP_INTERVAL == 0):
        rep.orchestrator.step(rt_subframes=4)
        fpv_frame = capture_frame(_fpv_rgb)
        fol_frame = capture_frame(_fol_rgb)
        if fpv_frame is not None or fol_frame is not None:
            session.record(fpv_frame, fol_frame, pos, yaw, pitch, roll, vel)

    sim_step += 1

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
if session:
    session.finish()
input_iface.unsubscribe_to_keyboard_events(keyboard_hw, _sub)
world.stop()
simulation_app.close()
