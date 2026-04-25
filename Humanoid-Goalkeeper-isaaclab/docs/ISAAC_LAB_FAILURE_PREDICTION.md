# Isaac Lab Training: Failure Prediction vs Isaac Gym Baseline

**Date:** 2026-04-25  
**Purpose:** Document failure modes where Isaac Lab training could diverge from the 100% working Isaac Gym baseline.

**Context:** Isaac Gym has ~18 months production use on this task. Isaac Lab port is 0 days production. This document identifies high-risk areas where the new implementation could fail silently or degrade training quality.

---

## Summary: High-Risk Failure Categories

| Category | Risk Level | Impact | Detection |
|---|---|---|---|
| PhysX 5.x numerical differences | MEDIUM | Subtle reward/observation changes | Compare trajectories to Isaac Gym logs |
| AMP motion priors disconnected | HIGH | Unnatural motion, training slow | Training curves plateau; visual inspection |
| Friction/restitution randomization no-op | MEDIUM | Reduced domain robustness | Domain rand ablation test |
| Standard PPO vs HIM-PPO | MEDIUM | Different convergence rate | Compare loss curves / wall-clock time |
| Quaternion convention silent divergence | HIGH | Wrong reward computation | Unit tests on rotation operations |
| Joint gain randomization bugs | HIGH | Training instability | Loss spikes; action saturation |
| Contact force tensor misalignment | MEDIUM | Wrong ball interaction rewards | Ball velocity tracking |
| Observation history buffer corruption | HIGH | Silent training failure | Verify rolling window correctness |
| GPU memory scaling | MEDIUM | OOM or silently reduced batch size | Monitor GPU utilization |
| API deprecation breakage | LOW | Future maintenance burden | Check warnings during training |

---

## 1. Physics Engine: PhysX 4.x (Isaac Gym) vs PhysX 5.x (Isaac Lab)

### Specific Risks

**1.1 Collision geometry precision**
- Isaac Gym: PhysX 4.1 built into isaacgym bindings
- Isaac Lab: PhysX 5.0 via Isaac Sim / Omniverse USD
- **Risk:** Sphere-plane collision and sphere-capsule (joint collision) detection may have different precision or contact point calculation
- **Impact:** Ball bouncing behavior, ground contact detection, ball-hand interaction rewards all depend on collision accuracy
- **Detection:**
  - Compare ball trajectories: simple drop test, measure bounce height and lateral roll
  - Compare ground contact forces: record `net_forces_w` on robot feet during running and compare magnitude
  - Flag if ball velocity (obs) diverges from commanded velocity after N steps

**1.2 Joint friction and damping**
- Isaac Gym: PhysX config via `gym.set_dof_properties(damping=...)`
- Isaac Lab: Implicit damping in physics engine, may differ in formulation
- **Risk:** Different joint damping behavior → different natural oscillation frequency
- **Impact:** PD control stability, gait oscillation behavior, torque requirements
- **Detection:**
  - Record joint velocities under manual torque application (no external forces)
  - Measure time-to-settle after impulse; compare to Isaac Gym baseline

**1.3 Depenetration and constraint solving**
- Isaac Gym: `max_depenetration_velocity = 1.0` (scene-level setting)
- Isaac Lab: No scene-level depenetration velocity; set per-body via RigidBodyPropertiesCfg (not yet implemented)
- **Risk:** Bodies may clip through terrain or each other under high-speed impacts
- **Impact:** Ball penetration into ground, hand interpenetration with ball, numerical instability
- **Detection:**
  - Monitor `net_forces_w` for NaN or Inf
  - Check trajectory logs for sudden position jumps > 1cm/frame

---

## 2. Numerical Convergence: Observation and Reward Computation

### Specific Risks

**2.1 Quaternion order: wxyz vs xyzw**
- Isaac Gym: [x, y, z, w]
- Isaac Lab: [w, x, y, z]
- **Risk:** If quaternion order is used wrong anywhere, rotations are backwards/inverted/wrong
- **Impact:** Base orientation wrong → gravity projection wrong → orientation rewards wrong → entire reward signal corrupted
- **Detection:**
  - Unit test: rotate [1, 0, 0] by known quaternion, verify result matches expected rotation
  - Sanity check: base_quat_xy should be [0, 0] when upright; check against logs
  - Training check: orientation reward should be positive (low error) at start, not negative

**2.2 Euler angle extraction**
- `euler_from_quaternion_wxyz()` custom implementation
- **Risk:** Wrong sign convention, angle wrapping, or gimbal lock handling
- **Impact:** `ang_vel` observation component used in reward; if wrong, drives training in opposite direction
- **Detection:**
  - Compare first 100 observation vectors to Isaac Gym logs (same seed reset)
  - Flag if `ang_vel` flips sign compared to baseline

**2.3 Contact force accumulation**
- Isaac Gym: `gym.refresh_net_contact_force_tensor()` is well-understood, hardware-tested over 18 months
- Isaac Lab: `ContactSensor.data.net_forces_w` (new code path, less battle-tested)
- **Risk:** Contact forces may be in different frame (world vs local), sign-flipped, or missing some contact types
- **Impact:** Ball stopping reward (`stopball` scales contact forces), foot contact detection for `feetorientation` reward
- **Detection:**
  - Unit test: drop ball on ground, verify `net_forces_w` on ball is upward (correct frame)
  - Training test: if `stopball` reward is zero or negative during ball contact, force computation is wrong

**2.4 Rolling window history computation**
- Original: `obs_buf` filled by env.step(), policy reads latest
- New: `actor_history_buf` rolled and concatenated into `{"policy": [N, 960]}`
- **Risk:** Off-by-one indexing, wrong history order (newest/oldest), or overwriting future frames
- **Impact:** Policy sees corrupted history → training diverges wildly
- **Detection:**
  - Check if observation at t=1 contains observation from t=0 at correct position
  - Verify all 10 history frames are unique (not all same or corrupted)
  - Flag if loss doesn't decrease at all in first iteration

---

## 3. Domain Randomization: Partially Implemented

### Specific Risks

**3.1 Friction and restitution randomization (NOT YET APPLIED)**
- Buffers exist: `self.friction_coeffs`, `self.restitution_coeffs`
- Sampling code exists: `torch_rand_float(self.cfg.friction_range, ...)`
- **But:** `write_physics_material_properties_to_sim()` is never called
- **Risk:** Domain randomization for physics robustness is silently disabled
- **Impact:** Policy trained only on nominal friction/restitution; may overfit and fail on real hardware or different sim settings
- **Detection:**
  - Check `self.friction_coeffs[0, :].unique()` in debugger — if all values are sampled (not all identical), randomization code ran but had no effect
  - Compare training curves: if similar to Isaac Gym despite disabled randomization, either (a) friction range was narrow, or (b) task doesn't depend on it much

**3.2 Mass and COM randomization (IMPLEMENTED)**
- Payload mass, link mass, CoM displacement applied via `ArticulationCfg.spawn.mass_props`
- **Risk:** Isaac Lab's mass application may not match Isaac Gym's method exactly
- **Impact:** Different inertia response, altered joint torque requirements
- **Detection:**
  - Compare torque magnitudes: plot distribution of applied torques vs Isaac Gym baseline
  - Flag if 90th percentile torque exceeds training limits

**3.3 Joint angle and actuation injection (IMPLEMENTED)**
- `joint_injection` and `actuation_offset` applied in `_apply_action()`
- **Risk:** If domain rand scales are wrong or applied at wrong time, training becomes unstable
- **Impact:** Training collapse or divergence
- **Detection:**
  - Check if loss spikes on iterations where rand parameters change significantly
  - Verify `actuation_offset` magnitude stays within [-1, 1] (normalized torque space)

---

## 4. Training Framework: PPO vs HIM-PPO

### Specific Risks

**4.1 AMP motion priors disconnected**
- Original: `HimPPO` includes adversarial discriminator; AMP loss term scales motion-matching reward at `amp_coef=0.4`
- New: Standard `PPO` from rsl_rl 3.x (no discriminator, no AMP gradient)
- **Risk:** Policy trained on task reward only, not motion naturalness
- **Impact:** Learned gait may be jerky, inefficient, or unstable on hardware despite achieving task rewards
- **Detection:**
  - Compare convergence speed: Isaac Gym (with AMP) converges faster due to motion prior?
  - Visual inspection: watch policy at convergence — does it move naturally or stiffly?
  - Training plateau: if AMP loss was significant, removing it should increase policy loss or increase episode reward variance

**4.2 Standard PPO hyperparameter drift**
- Values copied from original HIM-PPO config
- **Risk:** HIM-PPO and PPO may have different optimal learning rates, entropy coefficients, or clip ranges
- **Impact:** Slower convergence, higher variance, or failure to converge on task
- **Detection:**
  - Compare loss curves: Isaac Gym vs Isaac Lab over first 1000 iterations
  - Flag if PPO loss increases or plateaus while Isaac Gym loss decreases steadily

**4.3 Value function loss divergence**
- Original: `HimPPO` may have custom value function weight or scaling
- New: Standard PPO value weight = 1.0
- **Risk:** Value function learns slower or faster, affecting critic gradient quality
- **Impact:** Higher variance in rewards, less stable learning
- **Detection:**
  - Monitor `value_loss` and `surrogate_loss` separately; if value_loss >> surrogate_loss, critic is struggling
  - Flag if advantage estimates have very high variance (std >> mean)

---

## 5. Silent Failures: APIs That Work But Produce Wrong Results

### Specific Risks

**5.1 Joint ordering assumption**
- Isaac Lab uses breadth-first joint ordering
- **Risk:** If `self._robot.find_joints()` returns joints in unexpected order, gain indices are off
- **Impact:** Gains applied to wrong joints; right leg gets left leg gains; training is chaotic
- **Detection:**
  - Print joint order at init: `print([j.name for j in find_joints(...)])`
  - Verify order matches expected (FL, FR, BL, BR for quadrupeds; hand, arm, waist, torso for humanoids)
  - Test: zero out one joint's stiffness in config, verify it doesn't move with torque commands

**5.2 Action clipping before vs after domain randomization**
- Current code: `action = torch.clamp(action, -1, 1)` in `_pre_physics_step()`
- **Risk:** If actuation_offset or joint_injection added later, final torque may exceed limits
- **Impact:** Unused torque capacity, suboptimal control
- **Detection:**
  - Measure max applied torque over training; flag if it clips against torque_limits frequently
  - Compare to Isaac Gym logs: if Isaac Gym has unused headroom but Isaac Lab clips, implementation differs

**5.3 Observation concatenation order**
- Policy obs = 10 frames of [ball(3) + ang_vel(3) + gravity(3) + dof_pos(29) + dof_vel(29) + actions(29) = 96], concatenated as [frame_0, frame_1, ..., frame_9] = 960
- **Risk:** If concatenation is [frame_9, frame_8, ..., frame_0] (reversed history), policy learns inverted temporal pattern
- **Impact:** Policy trained on time-reversed trajectory; may not generalize to real episodes
- **Detection:**
  - Unit test: verify `obs_buf[:, 0:96]` is oldest frame (lowest t), `obs_buf[:, 864:960]` is newest frame (highest t)
  - Sanity check: position should change smoothly across frames, not jump around

---

## 6. Joint PD Control: Manual Torque Computation

### Specific Risks

**6.1 Gain randomization order**
- Kp, Kd factors sampled, then applied: `torques = p_gains * Kp_factors * (target - pos) - d_gains * Kd_factors * vel`
- **Risk:** If `Kp_factors` or `Kd_factors` not applied correctly (off-by-one, indexing error), some joints have nominal gains while others are randomized
- **Impact:** Training instability; some joints move too fast/slow compared to intended
- **Detection:**
  - Check `Kp_factors.unique()` — should contain multiple values across envs, not all 1.0
  - Measure joint acceleration distribution; flag if it's bimodal (two populations of fast/slow)

**6.2 Clipping after torque computation**
- Torques clipped to `torque_limits` after offset and injection applied
- **Risk:** If clipping is too aggressive, reachable action space shrinks unpredictably
- **Impact:** Policy can't execute high-torque motions; may get stuck
- **Detection:**
  - Measure percentage of timesteps where torque hits limit; flag if > 5%
  - Check if loss plateaus despite exploration — sign of constrained action space

**6.3 Target angle saturation**
- Action from policy: [-1, 1], scaled to joint angle range before PD computation
- **Risk:** If joint angle range is wrong (off-by-one in config, or hardcoded), target angle goes out of bounds
- **Impact:** PD gains compute against physically impossible targets; control breaks
- **Detection:**
  - Verify joint angle targets stay within joint limits: `assert target_angle in joint_limits`
  - Plot target vs actual angles; flag if target is saturated at limits

---

## 7. Contact Sensing: Per-Body Sensor

### Specific Risks

**7.1 Contact sensor prim path mismatch**
- Config: `ContactSensorCfg(prim_path=".../Robot/.*")`
- **Risk:** Regex doesn't match actual body names in URDF; ContactSensor.data stays zero
- **Impact:** All contact-based rewards zero; training has no signal
- **Detection:**
  - Print `self._robot._asset.root_prim_path`; manually check if contact sensor regex matches
  - Verify `self._contact_sensor.data.net_forces_w[0, :, :]` is nonzero during ground contact

**7.2 Contact force in wrong frame**
- Isaac Gym: contact forces in world frame
- Isaac Lab: `ContactSensor.data.net_forces_w` also world frame
- **Risk:** If assumption is wrong and forces are in local frame, magnitudes don't match gravity (should see ~mg on upright robot)
- **Impact:** Contact rewards scaled wrong; ball stopping reward wrong
- **Detection:**
  - Unit test: drop robot on ground, measure contact force z-component
  - Should be approximately `mass * g * num_feet` in upright position
  - Flag if order of magnitude is wrong

---

## 8. Observation Buffer Corruption

### Specific Risks

**8.1 Rolling window index wraparound**
- 10-frame history, each observation added at index `(step % 10) * 96`
- **Risk:** If modulo or index computation is off, frames overwrite the wrong positions
- **Impact:** Policy sees corrupted history (e.g., future frames where past should be)
- **Detection:**
  - Manually inspect history buffer: `actor_history_buf[env_id, 0:96]` should be oldest, `[864:960]` newest
  - Verify `obs_buf_time = torch.arange(100) % 10` gives expected patterns

**8.2 Critic observation not updated**
- Critic obs includes current state + privileged info; must be synchronized with policy obs history
- **Risk:** If critic obs is stale (from previous step), value function trains on wrong state
- **Impact:** Value function diverges from actual returns; high variance
- **Detection:**
  - Verify `critic_obs` is constructed AFTER policy obs history is updated
  - Check if value loss >> policy loss; sign of misaligned supervision

---

## 9. GPU Memory and Scaling

### Specific Risks

**9.1 VRAM exhaustion under scaling**
- Smoke test: 16 envs passes
- Full training: 512 envs
- **Risk:** At 512 envs, VRAM may fill unexpectedly; PyTorch silently reduces batch size or fails
- **Impact:** Training silently runs at reduced scale; metrics look OK but wall-clock training is much slower than expected
- **Detection:**
  - Monitor GPU memory during training: should stabilize at ~80% of available VRAM
  - Flag if VRAM keeps growing (memory leak)
  - Measure wall-clock time per iteration; compare to Isaac Gym baseline

**9.2 Tensor shapes incompatible at scale**
- Code tested at 16 and 64 envs; 512 envs may expose new broadcasting bugs
- **Risk:** An index operation that works at small N fails at large N due to RAM layout or cache effects
- **Impact:** Training crashes after many iterations
- **Detection:**
  - Run smoke test, then run with 256 envs for 100 iterations, then 512 envs
  - Catch OOM or shape errors early

---

## 10. Configuration Drift

### Specific Risks

**10.1 Someone reverts config back to Isaac Gym values**
- Configs like `wait_for_textures=False`, `solver_type=1`, `activate_contact_sensors=True` are Isaac Lab-specific
- **Risk:** If accidentally reverted, code breaks silently (hangs on texture load, wrong solver, zero contact forces)
- **Impact:** Hours of debugging for a simple config typo
- **Detection:**
  - Add comments to CLAUDE.md: "Do NOT change X — it's Isaac Lab-specific"
  - Run smoke test as first validation step (catches config errors immediately)

**10.2 Reward scale drift**
- Original: reward scales in `g1_29_config.py`
- New: reward scales in `goalkeeper_env_cfg.py`
- **Risk:** Someone updates one but not the other, breaking parity
- **Impact:** Training converges to different policy; not comparable to baseline
- **Detection:**
  - Diff `goal_height_reward_scale` (or any reward scale) between both configs monthly
  - Add to CLAUDE.md: "Always update BOTH original and port when changing reward scales"

---

## 11. Recommended Validation Checklist Before Long Runs

### Pre-Training Validation (Before Running Full 200k Iterations)

- [ ] **Smoke test passed:** 16 envs, 50 steps, exit code 0
- [ ] **Observations sane:** policy shape [16, 960], critic [16, 113]
- [ ] **Rewards in range:** step 0 reward ∈ [-1.0, 0.0]
- [ ] **No NaN/Inf:** check after 10 steps
- [ ] **Joint order correct:** print joint names, verify humanoid structure
- [ ] **Quaternion test:** rotate base by known quat, verify result
- [ ] **Contact test:** drop robot on ground, verify contact forces nonzero
- [ ] **Action clipping:** apply max torque, verify not hitting limits > 5% of time
- [ ] **Training validation:** 64 envs, 5 iterations, verify loss decreasing
- [ ] **No memory leak:** GPU memory stable over 100 iterations
- [ ] **Config checked:** `wait_for_textures=False`, `solver_type=1`, `activate_contact_sensors=True` all present

### Periodic Checks During Long Training (Every 10k Iterations)

- [ ] **Loss still decreasing** (or at expected plateau)
- [ ] **Reward mean increasing** (or at expected plateau)
- [ ] **No loss spikes** (spikes indicate instability)
- [ ] **GPU memory stable** (not growing)
- [ ] **No silent NaN** (verify last 100 steps have finite values)
- [ ] **Wall-clock time reasonable** (compare iterations/sec to baseline)

### Comparison to Isaac Gym Baseline

- [ ] **Loss curve shape similar** — same general trend, may differ in scale
- [ ] **Convergence speed comparable** — if Isaac Lab is 10x slower, something is wrong
- [ ] **Episode reward similar at convergence** — within 5% of original
- [ ] **Visual gait plausible** — not jerky, not impossibly fast (watch videos)

---

## 12. Future Work: Closing Failure Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| Implement friction/restitution randomization | LOW | 2 hours | Domain robustness |
| Re-integrate HIM-PPO / AMP discriminator | HIGH | 20 hours | Motion naturalness |
| Add per-body depenetration velocity | MEDIUM | 1 hour | Collision stability |
| Validate PhysX 5.x numerical equivalence | MEDIUM | 4 hours | Confidence in physics |
| Profile vs Isaac Gym on same hardware | MEDIUM | 2 hours | Identify scaling gaps |

---

## Summary

Isaac Lab port is functionally complete and smoke-tested. High-confidence on:
- Observation structure and shapes
- Reward computation structure (same math)
- Environment creation and stepping
- Training loop integration

Medium-confidence on:
- Physics engine exact equivalence (PhysX version difference)
- AMP motion priors (not connected; training may be less natural)
- Friction/restitution randomization (not applied; domain robustness may degrade)
- Quaternion operations (custom-implemented; needs ongoing validation)

Low-risk areas (well-tested, standard Isaac Lab APIs):
- Articulation and joint control
- Terrain and gravity
- Scene management

**Recommendation:** Run full training with monitoring checklist. Expect 100-200k iterations to converge similarly to Isaac Gym baseline. If training stalls or diverges around 50k iterations, debug the "Medium-confidence" areas first.
