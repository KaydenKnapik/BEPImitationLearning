# Isaac Lab Migration Guide: IsaacGymEnvs → Isaac Lab
> Source: https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html
> IsaacLab version in this project: 2.2.1 | IsaacSim: 5.0.0

## CRITICAL: Always read this file before editing any code in this folder.

---

## Overview

This project migrates the Humanoid-Goalkeeper (IsaacGym + legged_gym) to Isaac Lab 2.2.1 using the **DirectRLEnv** workflow. Every file in `goalkeeper/` reflects decisions from this guide.

---

## 1. Configuration System

### Old (IsaacGym / legged_gym)
```python
class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 6144
        episode_length_s = 3
    class sim:
        dt = 0.005
        substeps = 1
```

### New (Isaac Lab @configclass)
```python
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class GoalkeeperEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1/200)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1020, env_spacing=5.0)
    decimation: int = 4
    episode_length_s: float = 3.0
    action_space: int = 29
    observation_space: int = 960   # 10 * 96 (actor history)
    state_space: int = 113         # privileged/critic obs
```

**Key**: `episode_length_s` is in **seconds** (not steps).
Convert: `episode_length_s = dt * decimation * num_steps`

---

## 2. Simulation Configuration

| Parameter | IsaacGymEnvs | Isaac Lab |
|-----------|-------------|-----------|
| Time step | `sim.dt = 0.005` | `SimulationCfg(dt=1/200)` |
| Control freq | `decimation = 4` | `DirectRLEnvCfg.decimation = 4` |
| Substeps | `substeps = 1` | handled by PhysX internally |
| GPU pipeline | `use_gpu_pipeline=True` | automatic |
| Physics solver | `solver_type = 1 (TGS)` | `PhysxCfg(solver_type="TGS")` |

---

## 3. Asset Spawning

### Robot (URDF → USD auto-conversion)
```python
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

G1_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path="/path/to/g1_29.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),  # effort mode
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={".*": 0.0},  # overridden per-joint below
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=400.0,
        ),
    },
)
```

### Ball (sphere primitive)
```python
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=0.1,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.8),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.5)),
)
```

---

## 4. Scene Setup (_setup_scene)

```python
def _setup_scene(self):
    self._robot = Articulation(self.cfg.robot)
    self._ball = RigidObject(self.cfg.ball)
    # Contact sensor for feet/body
    self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
    # Terrain
    self.cfg.terrain.num_envs = self.scene.cfg.num_envs
    self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
    self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
    # Clone environments
    self.scene.clone_environments(copy_from_source=False)
    self.scene.filter_collisions(global_prim_paths=[self._terrain.prim_path])
    # Register with scene
    self.scene.articulations["robot"] = self._robot
    self.scene.rigid_objects["ball"] = self._ball
    self.scene.sensors["contact_sensor"] = self._contact_sensor
```

---

## 5. State Access API (CRITICAL CHANGES)

### Old (IsaacGym)
```python
# Acquire tensors (xyzw quaternions)
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
all_states = gymtorch.wrap_tensor(actor_root_state).view(num_envs, 2, 13)
root_states, ball_states = all_states[:, 0], all_states[:, 1]
base_quat = root_states[:, 3:7]  # xyzw format
gym.refresh_actor_root_state_tensor(sim)  # must refresh manually

rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
all_body_states = gymtorch.wrap_tensor(rigid_body_state)
dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
```

### New (Isaac Lab) - auto-refreshed via scene.update()
```python
# Robot state (wxyz quaternions!)
joint_pos = self._robot.data.joint_pos      # (num_envs, num_dof)
joint_vel = self._robot.data.joint_vel      # (num_envs, num_dof)
root_pos_w = self._robot.data.root_pos_w    # (num_envs, 3)
root_quat_w = self._robot.data.root_quat_w  # (num_envs, 4) -- WXYZ!
root_lin_vel_w = self._robot.data.root_lin_vel_w  # (num_envs, 3)
body_pos_w = self._robot.data.body_pos_w    # (num_envs, num_bodies, 3)
body_quat_w = self._robot.data.body_quat_w  # (num_envs, num_bodies, 4) WXYZ!
body_lin_vel_w = self._robot.data.body_lin_vel_w
body_ang_vel_w = self._robot.data.body_ang_vel_w

# Ball state
ball_pos = self._ball.data.root_pos_w       # (num_envs, 3)
ball_quat = self._ball.data.root_quat_w     # (num_envs, 4) WXYZ!
ball_vel = self._ball.data.root_lin_vel_w   # (num_envs, 3)

# Contact forces (via ContactSensor)
net_contact_forces = self._contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
```

---

## 6. CRITICAL: Quaternion Convention Change

**IsaacGym**: `xyzw` — index 3 is w
**Isaac Lab**: `wxyz` — index 0 is w

```python
# Convert if needed (but prefer using isaaclab.utils.math directly)
from isaaclab.utils.math import convert_quat
quat_wxyz = convert_quat(quat_xyzw, to="wxyz")

# isaaclab.utils.math functions expect wxyz:
from isaaclab.utils.math import quat_rotate_inverse, quat_apply
projected_gravity = quat_rotate_inverse(root_quat_w, gravity_vec)  # root_quat_w is wxyz
```

---

## 7. Joint Ordering Change

IsaacGym uses **depth-first** joint ordering.
Isaac Lab uses **breadth-first** joint ordering.

Always retrieve joint names from the articulation:
```python
joint_names = self._robot.data.joint_names
# Build index mapping by name lookup, not assumed position
idx, _ = self._robot.find_joints("left_hip_pitch_joint")
```

---

## 8. DirectRLEnv Method Mapping

| IsaacGym (legged_gym) | Isaac Lab (DirectRLEnv) | Notes |
|-----------------------|------------------------|-------|
| `create_sim()` | `_setup_scene()` | One-time scene init |
| `pre_physics_step(a)` | `_pre_physics_step(a)` | Called once before decimation |
| `gym.set_dof_actuation_force_tensor()` | `_apply_action()` → `robot.set_joint_effort_target()` | Called decimation times |
| `gym.simulate(sim)` | handled by base class | After `_apply_action()` |
| `gym.refresh_*()` | `scene.update()` by base class | Automatic |
| `compute_observations()` | `_get_observations()` | Return `{"policy": obs, "critic": priv}` |
| `compute_reward()` | `_get_rewards()` | Return scalar tensor |
| `check_termination()` + `reset_idx()` | `_get_dones()` + `_reset_idx()` | Split into two methods |

---

## 9. Writing State Back to Sim

```python
# Reset robot joints
self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
# Reset robot root (pose = pos(3) + quat_wxyz(4))
root_pose = torch.cat([pos, quat_wxyz], dim=-1)  # (N, 7)
root_vel = torch.cat([lin_vel, ang_vel], dim=-1)  # (N, 6)
self._robot.write_root_state_to_sim(
    torch.cat([root_pose, root_vel], dim=-1), env_ids
)
# Reset ball
ball_pose = torch.cat([ball_pos, ball_quat_wxyz], dim=-1)
self._ball.write_root_state_to_sim(
    torch.cat([ball_pose, ball_vel_zero], dim=-1), env_ids
)
# Apply external forces on ball (drag)
self._ball.set_external_force_and_torque(forces, torques)
```

---

## 10. Execution Flow

```
env.step(actions)
├── _pre_physics_step(actions)   # clip actions, compute delayed actions
├── for _ in range(decimation):
│   ├── _apply_action()          # compute PD torques, set_joint_effort_target
│   ├── scene.write_data_to_sim()  # propagate targets (automatic)
│   ├── sim.step()               # physics step (automatic)
│   └── scene.update(dt)         # refresh data (automatic)
├── _get_dones()                 # returns (terminated, timeout)
├── _get_rewards()               # returns reward tensor
├── _reset_idx(env_ids)          # reset terminated envs
└── _get_observations()          # returns {"policy": obs, "critic": priv}
```

---

## 11. Training

```bash
# Activate conda env
conda activate /home/isaak/miniconda3/envs/env_isaaclab

# Run training headless
python scripts/train.py --headless --num_envs=512 --max_iterations=200000
```

---

## 12. Physics Parameter Differences

| Parameter | Isaac Sim | Isaac Gym |
|-----------|-----------|-----------|
| Angular Damping | 0.05 | 0.0 |
| Max Angular Velocity | inf | 1000 |
| Damping/Stiffness units | 1/deg | 1/rad |
| Quaternion convention | wxyz | xyzw |
| Joint ordering | breadth-first | depth-first |

---

## 13. Contact Sensor Setup

```python
from isaaclab.sensors import ContactSensorCfg

contact_sensor: ContactSensorCfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/.*",
    update_period=0.0,
    history_length=3,
    debug_vis=False,
)
```

Access: `self._contact_sensor.data.net_forces_w`  shape: `(num_envs, num_bodies, 3)`

---

## 14. RSL-RL Integration

IsaacLab 2.2.1 requires `rsl-rl-lib >= 3.0.1` (standard PPO, not HIM-PPO).

```python
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

# Wrap DirectRLEnv for rsl_rl
env = RslRlVecEnvWrapper(env, clip_actions=100.0)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

# obs_groups maps env obs dict keys to rsl_rl internal keys
obs_groups = {
    "policy": ["policy"],   # actor receives "policy" obs (history)
    "critic": ["critic"],   # critic receives "critic" obs (privileged)
}
```
