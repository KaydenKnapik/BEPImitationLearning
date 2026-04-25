# Humanoid-Goalkeeper-isaaclab — Claude Instructions

## RULE #1: Always check the original before changing any code

Before modifying ANY logic, reward, observation, config value, or behaviour in this repo,
you MUST first read the corresponding file in the original upstream project:

```
/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/legged_gym/envs/
  base/legged_robot.py        ← all env logic (rewards, resets, observations, PD control)
  base/legged_robot_config.py ← base config defaults (sim.dt, physx, PPO params)
  g1/g1_29_config.py          ← G1-specific overrides (gains, reward scales, obs dims)
  g1/g1_utils.py              ← MotionLib, joint mapping, AMP dataset loading
```

The goal of this port is to stay **as close as possible to the original behaviour**.
Any divergence must be justified by an Isaac Lab API constraint — not preference.
If the original does X, this port must also do X (adapted for the new API, not reimagined).

## RULE #2: Read migration guide before any API changes

Always read `context/isaaclab_migration_guide.md` before editing Isaac Lab API calls.
It contains critical mappings, quaternion convention changes, and architectural decisions.

## RULE #3: Document every divergence

Any substantive difference from the original must be noted in `docs/FULL_DOCUMENTATION.md`
under "Things That Changed vs the Original".

---

## Project Goal
Isaac Lab 2.2.1 / Isaac Sim 5.0.0 port of the Humanoid-Goalkeeper project (Unitree G1).
- **Source (frozen, read-only):** `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/`
- **This folder:** all Isaac Lab-specific code — do NOT touch the source

## Running
```bash
conda activate /home/isaak/miniconda3/envs/env_isaaclab

# With GUI (small env count)
python -u scripts/train.py --num_envs=2 --max_iterations=200000

# Headless (full scale)
python -u scripts/train.py --headless --num_envs=512 --max_iterations=200000

# Smoke test
python -u scripts/test_env.py --headless --num_envs=16 --steps=50
```

Always use `python -u` (unbuffered) so print output isn't hidden behind Isaac Sim logs.

## Key Architecture

| File | Corresponds to original |
|---|---|
| `goalkeeper/goalkeeper_env_cfg.py` | `g1/g1_29_config.py` + `base/legged_robot_config.py` |
| `goalkeeper/goalkeeper_env.py` | `base/legged_robot.py` |
| `goalkeeper/goalkeeper_utils.py` | `g1/g1_utils.py` |
| `goalkeeper/agents/rsl_rl_ppo_cfg.py` | `g1/g1_29_config.py` → `G129CfgPPO` |
| `scripts/train.py` | `legged_gym/scripts/train.py` |

## Critical API Differences (Isaac Lab vs Isaac Gym)

1. Quaternions are **wxyz** in IsaacLab (was xyzw in IsaacGym)
2. Joint ordering is **breadth-first** (was depth-first in IsaacGym)
3. `_get_observations()` must return `{"policy": ..., "critic": ...}`
4. Contact forces come from `ContactSensor.data.net_forces_w` (not gym tensors)
5. Ball is a `RigidObject` (SphereCfg), not a URDF actor
6. Manual PD control: stiffness=0, damping=0, apply torques via `set_joint_effort_target()`
7. `self.actor_history_buf` = rolling obs history (renamed from `obs_buf` to avoid parent collision)
8. Use `quat_apply_inverse` not `quat_rotate_inverse` (same function, new name)
9. `joint_pos_limits` / `joint_vel_limits` (not `joint_limits` / `joint_velocity_limits`)

## License
CC BY-NC-SA 4.0 — non-commercial research only
