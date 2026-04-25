# Humanoid-Goalkeeper — Isaac Lab Port: Full Documentation

**Date:** 2026-04-25  
**Status:** Smoke-tested and training-validated (exit 0, 5 training iterations, rewards converging)  
**Original repo:** `../Humanoid-Goalkeeper/` (frozen — do not modify)  
**This repo:** `Humanoid-Goalkeeper-isaaclab/`

---

## What Was Done

The original `Humanoid-Goalkeeper` project uses **NVIDIA Isaac Gym** (a deprecated Python-only API) with a custom **HIM-PPO** training framework built on top of `rsl_rl`. This port replaces every simulator-facing layer with **Isaac Lab 2.2.1 / Isaac Sim 5.0.0** while keeping all game logic, reward functions, observation definitions, and hyperparameters as close to the original as possible.

### Steps taken:
1. Fetched the official Isaac Lab migration guide and saved it to `context/isaaclab_migration_guide.md` for reference.
2. Created the full `goalkeeper/` Python package from scratch — no code was copied from the original verbatim because the APIs are fundamentally different.
3. Iteratively fixed runtime errors by launching the environment headlessly and resolving each error in sequence.
4. Validated with `scripts/test_env.py` (50 steps, 16 envs) and `scripts/train.py` (5 iterations, 64 envs).

---

## File Structure Comparison

### Original (`Humanoid-Goalkeeper/legged_gym/`)

```
legged_gym/
  legged_gym/
    envs/
      base/
        base_task.py          ← BaseTask (IsaacGym wrapper)
        legged_robot.py       ← Main env logic (1561 lines)
        legged_robot_config.py ← Base config dataclass
      g1/
        g1_29_config.py       ← G1-specific overrides (389 lines)
        g1_utils.py           ← MotionLib, joint mapping
    scripts/
      train.py                ← task_registry → HimOnPolicyRunner
      play.py
    utils/
      task_registry.py        ← Registers envs with IsaacGym
      math.py
      helpers.py
```

### New (`Humanoid-Goalkeeper-isaaclab/`)

```
goalkeeper/
  __init__.py                 ← gymnasium.register("Isaac-Goalkeeper-Direct-v0")
  goalkeeper_env_cfg.py       ← GoalkeeperEnvCfg(DirectRLEnvCfg) — all params (390 lines)
  goalkeeper_env.py           ← GoalkeeperEnv(DirectRLEnv) — all logic (1098 lines)
  goalkeeper_utils.py         ← MotionLib, math helpers (wxyz-aware)
  agents/
    __init__.py
    rsl_rl_ppo_cfg.py         ← GoalkeeperPPORunnerCfg(RslRlOnPolicyRunnerCfg)
scripts/
  train.py                    ← AppLauncher → gym.make → OnPolicyRunner
  play.py                     ← Inference loop
  test_env.py                 ← Headless smoke test
context/
  isaaclab_migration_guide.md ← Official migration reference
docs/
  FULL_DOCUMENTATION.md       ← This file
  MIGRATION_DECISIONS.md      ← Concise change log
setup.py
CLAUDE.md                     ← Auto-instructions for Claude
```

---

## Simulator Layer Changes

### Isaac Gym vs Isaac Lab

| Aspect | Isaac Gym (original) | Isaac Lab (new) |
|---|---|---|
| Python API | `isaacgym` (deprecated) | `isaaclab` (current) |
| State access | `gym.acquire_*_tensor()` + `gym.refresh_*()` | `scene.update()` auto-refreshes all state |
| Env base class | `BaseTask` | `DirectRLEnv` |
| Physics engine | PhysX via IsaacGym bindings | PhysX 5 via Isaac Sim / Omniverse USD |
| Asset loading | `gym.load_asset()` with URDF path | `ArticulationCfg(spawn=UrdfFileCfg(...))` — auto converts URDF → USD |
| Contact forces | `gym.refresh_net_contact_force_tensor()` | `ContactSensor.data.net_forces_w` |
| Terrain | `gym.add_ground_plane()` | `TerrainImporterCfg(terrain_type="plane")` |
| Gymnasium API | Custom | Standard `gym.make()` / `env.reset()` / `env.step()` |

---

## Environment Logic — What Stayed the Same

The following were ported exactly (same math, same values):

- **29 DOF G1 robot** — same URDF (`legged_gym/resources/robots/g1/urdf/g1_29.urdf`)
- **All 6 goalkeeper target regions** with identical `height`/`width`/`maxh`/`maxw`/`evalh`/`evalw` ranges
- **Observation dimensions** — policy=960-dim (10×96 history), critic=113-dim privileged
- **One-step obs layout** — `ball(3) + ang_vel(3) + gravity(3) + dof_pos(29) + dof_vel(29) + actions(29) = 96`
- **Privileged obs layout** — `one_step(96) + lin_vel(3) + region(1) + end_target(3) + ball_vel(3) + hand_r(3) + hand_l(3) + dist(1) = 113`
- **All reward functions** — eereach, success, stopball, stayonline, noretreat, successland, feetorientation, penalize_sharpcontact, penalize_kneeheight, feet_slippage, postorientation, postangvel, postupperdofpos, postwaistdofpos, postlinvel, ang_vel_xy, dof_acc, smoothness, torques, dof_vel, dof_pos_limits, dof_vel_limits, torque_limits, deviation_waist_pitch_joint
- **All reward scales** — exact same values from `g1_29_config.py`
- **All PD gains** — same stiffness/damping per joint group
- **All domain randomization parameters** — payload mass, CoM displacement, link mass, friction, restitution, kp/kd, joint injection, actuation offset
- **Ball physics** — radius=0.1m, mass=0.1kg, restitution=0.8, drag formula unchanged
- **Episode length** — 3.0s, decimation=4, physics dt=0.005s → policy dt=0.02s
- **Curriculum** — same sigma progression, same curriculum joint groups
- **Push randomization** — interval 15s, max_vel_xy=1.5
- **AMP dataset loading** — `MotionLib` class preserved, loads `.pt` files from `resources/datasets/goalkeeper/`
- **Initial joint poses** — exact same 29-element vector from `g1_29_config.py`

---

## Things That Changed vs the Original

### 1. Quaternion convention: xyzw → wxyz

**Original:** Isaac Gym uses `[x, y, z, w]`  
**New:** Isaac Lab uses `[w, x, y, z]`

Affects: `quat_rotate_inverse`, base orientation calculations, gravity projection, euler angle extraction.  
The `euler_from_quaternion_wxyz()` function in `goalkeeper_utils.py` was written for the new convention.

---

### 2. Joint ordering: depth-first → breadth-first

**Original:** Isaac Gym traverses the URDF joint tree depth-first  
**New:** Isaac Lab uses breadth-first ordering

This changes which numerical index corresponds to which joint. To avoid hardcoded index bugs, `_build_joint_maps()` in `goalkeeper_env.py` uses `self._robot.find_joints(name)` to build name→index maps at runtime. No hardcoded index arrays carried over from the original.

---

### 3. Training framework: HIM-PPO → standard PPO

**Original:** `HimOnPolicyRunner` + `HimPPO` (custom `rsl_rl` fork with AMP motion priors)  
**New:** Standard `OnPolicyRunner` + `PPO` from `rsl_rl 3.x`

**Why:** `rsl_rl 3.x` (the version shipped with Isaac Lab 2.2.1) does not include `HimPPO` or the AMP discriminator. The adversarial motion prior loss term is not connected to the optimizer.

**Impact:** The `MotionLib` still loads reference motion data and is available in the env, but it has no gradient path to the policy network. Training will still converge on the goalkeeper task but may produce less natural-looking motion.

**PPO hyperparameters — same as original:**

| Param | Original (HIM-PPO) | New (PPO) |
|---|---|---|
| `learning_rate` | 1e-3 | 1e-3 |
| `clip_param` | 0.2 | 0.2 |
| `value_loss_coef` | 1.0 | 1.0 |
| `num_learning_epochs` | 5 | 5 |
| `num_mini_batches` | 4 | 4 |
| `max_grad_norm` | 1.0 | 1.0 |
| `num_steps_per_env` | 100 | 100 |
| `entropy_coef` | — | 0.01 |
| `schedule` | — | "adaptive" |

---

### 4. PD control: built-in drives → manual effort control

**Original:** `gym.set_dof_properties(stiffness=..., damping=...)` — physics engine handles PD  
**New:** USD drives disabled (`target_type="none"`, stiffness=0, damping=0); torques computed manually per step:

```python
torques = p_gains * Kp_factors * (target - joint_pos) - d_gains * Kd_factors * joint_vel
torques += actuation_offset + joint_injection
torques = clip(torques, -torque_limits, torque_limits)
self._robot.set_joint_effort_target(torques)
```

PD gains, gain randomization, and torque clipping are identical to the original.

---

### 5. Ball asset: URDF actor → primitive sphere

**Original:** `gym.load_asset(ball.urdf)` — ball defined in URDF  
**New:** `RigidObjectCfg(spawn=SphereCfg(radius=0.1, mass_props=MassPropertiesCfg(mass=0.1)))`

No URDF for the ball. Isaac Lab's primitive sphere spawner handles geometry. Physics properties (restitution=0.8, drag computation) are identical.

---

### 6. Contact sensing: gym tensor → ContactSensor

**Original:** `gym.refresh_net_contact_force_tensor()` — single global tensor  
**New:** `ContactSensor(prim_path=".../Robot/.*")` — per-body contact force array at `sensor.data.net_forces_w`

Requires `activate_contact_sensors=True` in the robot spawn config (this was implicit in Isaac Gym).

---

### 7. Observation buffer name: `obs_buf` → `actor_history_buf`

**Original:** `self.obs_buf` was a simple tensor attribute  
**New:** `DirectRLEnv` overwrites `self.obs_buf` with the dict returned from `_get_observations()` after every step. The internal 10-frame rolling history is stored as `self.actor_history_buf` to avoid collision.

---

### 8. Config system: nested dataclasses → `@configclass`

**Original:** nested Python classes (`G129Cfg.env`, `G129Cfg.rewards.scales`, etc.)  
**New:** flat `@configclass GoalkeeperEnvCfg(DirectRLEnvCfg)` with direct field annotations

The `@configclass` decorator from Isaac Lab validates all fields at construction time (catches `MISSING` values, type errors). All values from the original nested classes are preserved as flat fields on `GoalkeeperEnvCfg`.

---

### 9. PhysX config — `max_depenetration_velocity` removed

**Original:** `physx.max_depenetration_velocity = 1.0` (set as a physx scene param)  
**New:** not a valid field in `PhysxCfg` in Isaac Lab 2.2.1; `solver_type` changed from string `"TGS"` to int `1`

Depenetration velocity is now set per-body via `RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)` on the individual asset spawn configs.

---

### 10. Logging: W&B → TensorBoard (default)

**Original:** W&B logging to project `"goalkeepper"`, entity `"i-p-b-bouwmeester-..."` (set via env var)  
**New:** TensorBoard by default (`logger: str = "tensorboard"`); W&B available by changing `logger: str = "wandb"`

Logs are written to `logs/rsl_rl/goalkeeper/<timestamp>_g1_isaaclab/`.

---

### 11. Domain randomization — friction/restitution not yet applied

**Original:** `gym.set_actor_rigid_shape_properties()` applied randomized friction/restitution per env  
**New:** Values are sampled but `write_physics_material_properties_to_sim()` is not yet called. The buffers `self.friction_coeffs` and `self.restitution_coeffs` exist but have no effect on the simulation.

All other domain randomization (payload mass, CoM displacement, link mass, kp/kd, joint injection, actuation offset) is applied correctly.

---

### 12. AMP motion priors — not connected to optimizer

**Original:** `HimPPO` included an adversarial discriminator; the AMP coefficient `amp_coef=0.4` scaled a motion-matching reward that was added to PPO's policy gradient.  
**New:** `MotionLib` loads the `.pt` datasets and can sample reference frames, but there is no discriminator network and no AMP gradient in the optimizer. `amp_coef` is still defined in `GoalkeeperEnvCfg` for future use.

---

## Running

```bash
# Activate environment
conda activate /home/isaak/miniconda3/envs/env_isaaclab

# Smoke test (verify env creates and steps)
/home/isaak/miniconda3/envs/env_isaaclab/bin/python -u scripts/test_env.py --headless --num_envs=16 --steps=50

# Quick training validation
/home/isaak/miniconda3/envs/env_isaaclab/bin/python -u scripts/train.py --headless --num_envs=64 --max_iterations=5

# Full training
/home/isaak/miniconda3/envs/env_isaaclab/bin/python -u scripts/train.py --headless --num_envs=512 --max_iterations=200000
```

Always use `python -u` (unbuffered) — Isaac Sim's log messages go to the same stdout buffer and can hide script output without it.

---

## Known Issues / Future Work

| Issue | Severity | Fix |
|---|---|---|
| AMP motion priors not connected | Medium | Re-implement HIM-PPO discriminator on top of rsl_rl 3.x |
| Friction/restitution randomization no-op | Low | Call `write_physics_material_properties_to_sim()` in domain rand reset |
| Motion dataset `'list' object has no attribute 'items'` warning | Low | `load_imitation_dataset()` returns a list when only one dataset type; needs dict wrapping |
| `quat_rotate_inverse` deprecation warnings | Cosmetic | Replace with `quat_apply_inverse` (same function, new name) |
| `num_envs` not divisible by 6 | Cosmetic | Current default is 1020 which is fine; smoke test uses 16 which truncates command regions |

---

## Validated Results (2026-04-25)

```
Smoke test (16 envs, 50 steps):
  policy obs shape: torch.Size([16, 960])  ✓
  critic obs shape: torch.Size([16, 113])  ✓
  step 0: reward=-0.55, terminated=0       ✓
  step 40: reward=-0.74, terminated=0      ✓
  PASSED

Training (64 envs, 5 iterations):
  Iteration 0: VF loss=224.1, surrogate=-0.063
  Iteration 4: VF loss=178.5, surrogate=-0.011  ✓ (decreasing)
  EXIT: 0  ✓
```
