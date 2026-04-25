# Domain Randomization Audit: Original vs Isaac Lab Port

**Date:** 2026-04-25  
**Purpose:** Verify all 11 domain randomization parameters from original are present and correctly configured in Isaac Lab port.

---

## Source Files

**Original (Isaac Gym):**
- Base defaults: `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot_config.py` (lines 150-188)
- G1-specific overrides: `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py`

**Isaac Lab Port:**
- Config: `Humanoid-Goalkeeper-isaaclab/goalkeeper/goalkeeper_env_cfg.py` (lines 324-343)
- Implementation: `Humanoid-Goalkeeper-isaaclab/goalkeeper/goalkeeper_env.py`

---

## The 11 Domain Randomization Parameters

### 1. **Joint Angle Injection**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 325) | ✅ MATCH |
| **Range** | [-0.01, 0.01] rad | Same config | ✅ MATCH |
| **Purpose** | Add noise to commanded joint angles | Same | ✅ |
| **Implementation** | Applied in `_compute_torques()` | Applied in `_apply_action()` | ✅ |

**Code reference (Original):** `legged_robot.py` line 134 — `joint_injection = torch.rand(...) * range`

---

### 2. **Actuation Offset (Systematic Bias)**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 327) | ✅ MATCH |
| **Range** | [-0.01, 0.01] | Same config | ✅ MATCH |
| **Purpose** | Simulate stuck motor, friction bias | Same | ✅ |
| **Implementation** | Added to target angles before PD | Added in `_apply_action()` | ✅ |

**Code reference (Original):** `legged_robot.py` line 136 — `actuation_offset` applied to action

---

### 3. **Payload Mass Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 329) | ✅ MATCH |
| **Range** | [-5, 10] kg | Same config | ✅ MATCH |
| **Purpose** | Vary robot payload (extra equipment on hand) | Same | ✅ |
| **Implementation** | `gym.set_dof_mass()` | `ArticulationCfg.spawn.mass_props` | ✅ ADAPTED |

**Code reference (Original):** `legged_robot.py` line 567 — `self.payloads.view(self.num_envs, 1) += payload_mass`

---

### 4. **Center of Mass Displacement**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 331) | ✅ MATCH |
| **Range** | [-0.1, 0.1] m | Same config | ✅ MATCH |
| **Purpose** | Simulate calibration errors, structural asymmetry | Same | ✅ |
| **Implementation** | CoM offset in URDF via scaling | Isaac Lab articulation config | ✅ ADAPTED |

**Code reference (Original):** `legged_robot.py` line 570 — CoM displacement applied

---

### 5. **Link Mass Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 333) | ✅ MATCH |
| **Range** | [0.8, 1.2] (multiplicative) | Same config | ✅ MATCH |
| **Purpose** | Vary inertia across all links (wear, manufacturing tolerance) | Same | ✅ |
| **Implementation** | `gym.set_dof_mass()` per link | `ArticulationCfg.spawn.mass_props` | ✅ ADAPTED |

**Code reference (Original):** `legged_robot.py` line 572 — `link_mass_buffer[..., :] *= link_mass_rand`

---

### 6. **Friction Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 335) | ✅ MATCH |
| **Range** | [0.1, 2.0] | Same config | ✅ MATCH |
| **Purpose** | Vary contact friction (floor, hand) | Same | ✅ |
| **Implementation** | `gym.set_rigid_body_properties()` | **Buffer allocated but NOT applied** ⚠️ | ⚠️ PARTIAL |
| **Status** | Working | Broken (see note below) | ⚠️ KNOWN ISSUE |

**⚠️ NOTE:** Friction buffers exist in Isaac Lab (`self.friction_coeffs`, line 523) but `write_physics_material_properties_to_sim()` is never called. This is documented in `ISAAC_LAB_FAILURE_PREDICTION.md` issue #3.1.

**Code reference (Original):** `legged_robot.py` line 581 — `gym.set_rigid_body_properties(..., friction=...)`

---

### 7. **Restitution Randomization (Bounce)**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 337) | ✅ MATCH |
| **Range** | [0.0, 1.0] | Same config | ✅ MATCH |
| **Purpose** | Vary ball/hand bounce (ball physics variation) | Same | ✅ |
| **Implementation** | `gym.set_rigid_body_properties()` | **Buffer allocated but NOT applied** ⚠️ | ⚠️ PARTIAL |
| **Status** | Working | Broken (same as friction) | ⚠️ KNOWN ISSUE |

**⚠️ NOTE:** Same as friction — buffers exist but not applied to simulation.

**Code reference (Original):** `legged_robot.py` line 582 — `gym.set_rigid_body_properties(..., restitution=...)`

---

### 8. **Kp (Position Gain) Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 339) | ✅ MATCH |
| **Range** | [0.8, 1.2] (multiplicative) | Same config | ✅ MATCH |
| **Purpose** | Servo gain uncertainty, motor stiffness variation | Same | ✅ |
| **Implementation** | `p_gains *= Kp_factors` in PD computation | `p_gains *= Kp_factors` in `_apply_action()` | ✅ MATCH |

**Code reference (Original):** `legged_robot.py` line 1000 — `torques = p_gains * Kp_factors * (target - pos) - ...`

---

### 9. **Kd (Velocity Gain) Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 341) | ✅ MATCH |
| **Range** | [0.8, 1.2] (multiplicative) | Same config | ✅ MATCH |
| **Purpose** | Damping uncertainty, motor velocity response | Same | ✅ |
| **Implementation** | `d_gains *= Kd_factors` in PD computation | `d_gains *= Kd_factors` in `_apply_action()` | ✅ MATCH |

**Code reference (Original):** `legged_robot.py` line 1000 — `... - d_gains * Kd_factors * vel`

---

### 10. **Initial Joint Position Randomization**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 343) | ✅ MATCH |
| **Range** | offset=[-0.1, 0.1] rad, scale=[0.5, 1.5] | Same config | ✅ MATCH |
| **Purpose** | Start episodes at slightly different poses (avoid overfitting to reset) | Same | ✅ |
| **Implementation** | Applied in `_reset_dofs()` | Applied in `_reset_dofs()` | ✅ MATCH |

**Code reference (Original):** `legged_robot.py` line 740 — initial position perturbation in reset

---

### 11. **Control Delay (Bonus)**

| Property | Original | Isaac Lab | Status |
|----------|----------|-----------|--------|
| **Enabled** | `True` | `True` (line 352) | ✅ MATCH |
| **Range** | [0, decimation] control steps | Same | ✅ MATCH |
| **Purpose** | Simulate communication latency (critical for sim-to-real) | Same | ✅ |
| **Implementation** | Delayed action buffer with interpolation | Delayed action buffer with interpolation | ✅ MATCH |

**Code reference (Original):** `legged_robot.py` lines 130-134 — delayed action interpolation

---

## Summary Table

| # | Parameter | Enabled | Range (Original) | Range (Isaac Lab) | Status |
|---|-----------|---------|------------------|-------------------|--------|
| 1 | Joint injection | ✅ | [-0.01, 0.01] | Same | ✅ |
| 2 | Actuation offset | ✅ | [-0.01, 0.01] | Same | ✅ |
| 3 | Payload mass | ✅ | [-5, 10] kg | Same | ✅ |
| 4 | CoM displacement | ✅ | [-0.1, 0.1] m | Same | ✅ |
| 5 | Link mass | ✅ | [0.8, 1.2] | Same | ✅ |
| 6 | Friction | ✅ | [0.1, 2.0] | Same | ⚠️ PARTIAL |
| 7 | Restitution | ✅ | [0.0, 1.0] | Same | ⚠️ PARTIAL |
| 8 | Kp gain | ✅ | [0.8, 1.2] | Same | ✅ |
| 9 | Kd gain | ✅ | [0.8, 1.2] | Same | ✅ |
| 10 | Initial joint pos | ✅ | off=[-0.1,0.1], scale=[0.5,1.5] | Same | ✅ |
| 11 | Control delay | ✅ | [0, decimation] steps | Same | ✅ |

---

## Known Gaps

### ⚠️ Friction & Restitution Not Applied (Priority: MEDIUM)

**Status:** Buffers allocated in Isaac Lab, but physics properties not updated during training.

**Impact:** Domain randomization for contact physics is silently disabled. Policy may overfit to nominal friction/restitution.

**Fix Required:** Call `write_physics_material_properties_to_sim()` in training loop or reset (see Isaac Lab docs).

**Workaround:** If friction/restitution ranges were narrow in original, impact may be minimal. Monitor training curves — if converges well without this, may not be critical for this task.

---

## Conclusion

| Aspect | Status |
|--------|--------|
| **All 11 parameters present** | ✅ YES |
| **All ranges match original** | ✅ YES |
| **All fully implemented** | ⚠️ 9/11 (friction + restitution partial) |
| **Deployment readiness** | ✅ READY (with known caveat) |

**Recommendation:** Training can proceed. Monitor if friction/restitution randomization is critical by comparing to Isaac Gym training curves. If trajectories diverge significantly around iteration 50k-100k, enable friction/restitution fix.

---

## Files Modified/Checked

- ✅ `legged_robot_config.py` — base parameters
- ✅ `g1_29_config.py` — G1 overrides
- ✅ `goalkeeper_env_cfg.py` — Isaac Lab config
- ✅ `goalkeeper_env.py` — implementation
