# Domain Randomization: Complete Explanation

**Purpose:** Vary physics and control parameters during training so the learned policy works on real hardware despite sim-to-real gaps.

Without domain randomization, the policy overfits to the exact simulator parameters and fails on real robots.

---

## 1. Joint Angle Injection

**Range:** [-0.01, 0.01] radians (~0.6 degrees)

**What it does:**
- Adds random noise directly to the commanded joint angles before PD control
- Example: You command joint to 0.5 rad, but sim randomly applies 0.501 or 0.499 instead

**Why it matters:**
- Real motors have sensor noise and measurement jitter
- Joint encoders have finite resolution and backlash
- Trains policy to be robust to noisy state feedback

**Physics:**
```
target_angle = action * range + noise
target_angle = action * range + uniform(-0.01, 0.01)
```

**Real-world analogy:** Your joint sensor reads "45.0°" but it's really "45.0° ± 0.6°" due to noise.

---

## 2. Actuation Offset (Systematic Bias)

**Range:** [-0.01, 0.01] (normalized torque space)

**What it does:**
- Adds a constant offset to motor commands
- Example: You command torque = 10 Nm, but motor actually applies 10.1 Nm every step
- Stays constant for entire episode (not random per step)

**Why it matters:**
- Real motors have systematic bias from friction, calibration errors, servo offset
- Different motors on the same robot may have different biases
- Policy must learn to compensate for persistent offsets

**Physics:**
```
applied_torque = computed_torque + offset
offset = uniform(-0.01, 0.01) per environment (sampled once per reset)
```

**Real-world analogy:** Your leg actuator always applies 2% more torque than commanded due to mechanical wear.

---

## 3. Payload Mass Randomization

**Range:** [-5, 10] kg

**What it does:**
- Simulates carrying different loads (ball, tools, sensor package)
- Adds extra mass to the robot (increases inertia)
- Makes the humanoid heavier or lighter dynamically

**Why it matters:**
- Real robots carry different payloads (sensor packages, grippers, tools)
- Heavier payloads require more torque for same motion
- Lighter payloads are faster but less stable
- Policy must work across payload range

**Physics:**
```
total_mass = base_mass + payload_mass
payload_mass = uniform(-5, 10) kg per environment
```

**Real-world analogy:** Your humanoid sometimes catches a 5 kg ball, sometimes a 10 kg object, sometimes nothing.

---

## 4. Center of Mass (CoM) Displacement

**Range:** [-0.1, 0.1] meters

**What it does:**
- Shifts the center of mass location randomly in 3D space
- Simulates asymmetric wear, structural damage, or internal shifts
- Makes the robot feel "off-balance" in different directions

**Why it matters:**
- Real robots have manufacturing tolerances and wear
- Internal components may shift during use
- CoM displacement changes balance point and torque requirements
- Policy must adapt to different balance points

**Physics:**
```
CoM_offset = uniform(-0.1, 0.1) m per axis
body_CoM_actual = body_CoM_nominal + CoM_offset
```

**Real-world analogy:** Due to manufacturing, your robot's CoM is 5 cm higher than designed, affecting balance.

---

## 5. Link Mass Randomization

**Range:** [0.8, 1.2] multiplicative factor

**What it does:**
- Scales mass of each body link independently
- Example: Upper arm becomes 20% lighter (0.8×), lower arm becomes 20% heavier (1.2×)
- Different scale for each link AND each environment

**Why it matters:**
- Real robots have component tolerances
- Materials vary (aluminum vs composite)
- Joints wear and add/lose mass over time
- Different links wear at different rates
- Policy must handle variable link inertias

**Physics:**
```
link_mass_actual = link_mass_nominal * factor
factor = uniform(0.8, 1.2) per link
```

**Real-world analogy:** Over time, your arm actuators accumulate wear — some links get heavier (metal dust), others lighter (corrosion).

---

## 6. Friction Randomization

**Range:** [0.1, 2.0]

**What it does:**
- Varies contact friction coefficient between robot and ground/ball
- Higher friction = more grip (harder to slide)
- Lower friction = slippery surface

**Why it matters:**
- Real ground has different friction (wet floor, carpet, concrete)
- Ball surface has different friction (smooth vs rough)
- Robot hands have different grip coefficients
- Policy must work on slippery AND grippy surfaces

**Physics:**
```
friction_coefficient = uniform(0.1, 2.0)
friction_force = friction_coefficient * normal_force
```

**Real-world analogy:** Same task on ice (friction=0.1) vs rough concrete (friction=2.0) requires different strategy.

---

## 7. Restitution Randomization (Bounce)

**Range:** [0.0, 1.0]

**What it does:**
- Varies how much the ball/objects bounce
- 0.0 = dead ball (no bounce, absorbs impact)
- 1.0 = super-bouncy ball (bounces back at same speed)

**Why it matters:**
- Real balls have variable bounce depending on:
  - Temperature (cold balls are less bouncy)
  - Material (rubber vs plastic)
  - Wear (old balls are deader)
- Policy must track and catch balls with different bounce characteristics

**Physics:**
```
velocity_after_bounce = velocity_before_bounce * restitution_coefficient
restitution = uniform(0.0, 1.0)
```

**Real-world analogy:** In summer (hot), the ball bounces more (restitution=0.8). In winter (cold), it bounces less (restitution=0.5).

---

## 8. Kp (Position Gain) Randomization

**Range:** [0.8, 1.2] multiplicative

**What it does:**
- Varies the proportional gain in PD controller
- Kp determines how stiff the joint is
- Higher Kp = stiffer (tracks harder, more oscillation risk)
- Lower Kp = softer (smoother, less responsive)

**Why it matters:**
- Real servos have gain variations due to:
  - Manufacturing tolerances
  - Temperature drift
  - Wear and calibration drift
  - Different motor models
- Policy must work with stiffer AND softer joints

**PD Control Formula:**
```
torque = Kp * (target_pos - actual_pos) - Kd * actual_vel
torque = (Kp * factor) * (target - pos) - Kd * vel
factor = uniform(0.8, 1.2)

Example:
  Nominal Kp = 100 N·m/rad
  Random factor = 0.9
  Actual Kp = 100 × 0.9 = 90 N·m/rad (softer)
```

**Real-world analogy:** Servo A has stiff response (Kp=100), Servo B is worn (Kp=80) — same policy must work on both.

---

## 9. Kd (Velocity Gain) Randomization

**Range:** [0.8, 1.2] multiplicative

**What it does:**
- Varies the derivative (damping) gain in PD controller
- Kd determines how much damping (friction-like resistance to motion)
- Higher Kd = more damped (slower, oscillation suppression)
- Lower Kd = less damped (faster response, more oscillation risk)

**Why it matters:**
- Real servos have damping variations due to:
  - Oil viscosity in joints (temperature dependent)
  - Bearing wear
  - Manufacturing tolerances
- Policy must handle joints that are over-damped AND under-damped

**PD Control Formula:**
```
torque = Kp * (target - pos) - (Kd * factor) * vel
factor = uniform(0.8, 1.2)

Example:
  Nominal Kd = 10 N·m·s/rad
  Random factor = 1.1
  Actual Kd = 10 × 1.1 = 11 N·m·s/rad (more damped)
```

**Real-world analogy:** Old joint with stiff bearings (Kd=12) vs newly serviced joint (Kd=8) — same policy adapts.

---

## 10. Initial Joint Position Randomization

**Range:** 
- Offset: [-0.1, 0.1] rad
- Scale: [0.5, 1.5]

**What it does:**
- Adds random perturbation to starting pose at episode reset
- Scale: multiplies default pose by random factor [0.5, 1.5]
- Offset: adds random angle on top
- Each episode starts from slightly different position

**Why it matters:**
- Real robots don't reset to exact same pose
- Each episode starts from slightly different configuration
- Prevents overfitting to a specific reset pose
- Trains policy to work from diverse starting positions

**Reset Mechanics:**
```
reset_pos = default_pos
reset_pos *= uniform(0.5, 1.5)  # Scale first
reset_pos += uniform(-0.1, 0.1) # Then offset
```

**Real-world analogy:** After each catch attempt, the humanoid doesn't perfectly reset — it's always slightly off-pose. Policy must handle this.

---

## BONUS: Control Delay (Latency)

**Range:** [0, decimation] control steps

**What it does:**
- Simulates communication latency between controller and robot
- Example: You send command at t=0, but robot doesn't receive until t=10ms
- Random delay per environment: some have 0ms delay, others have 50ms

**Why it matters:**
- Real robot control has latency from:
  - Network communication (~20-100ms)
  - Processing time (~5-50ms)
  - Sensor-to-actuator loop delay
- Policy must handle delayed feedback and be stable with lag

**Implementation:**
```
# Store last N actions
delayed_actions[0] = action from 50ms ago
delayed_actions[1] = action from 40ms ago
...
delayed_actions[N] = action now

# Apply action based on random delay
delay_steps = random(0, decimation)
applied_action = delayed_actions[delay_steps]
```

**Real-world analogy:** 
- Sim world: You command motion, robot responds instantly (ideal)
- Real world: You command motion, robot responds 50ms later (delayed)
- Without this training, learned policy is unstable on real hardware

---

## Summary: Why All 11 Matter for Real-World Deployment

| Parameter | Sim Assumption | Real-World Reality | Training Goal |
|-----------|---------------|-------------------|--------------|
| Joint Injection | Perfect sensors | Noisy encoders | Robust to noise |
| Actuation Offset | Perfect motors | Biased servos | Compensate for bias |
| Payload Mass | Fixed mass | Variable load | Works with different payloads |
| CoM Displacement | Perfect balance | Off-center CoM | Handles imbalance |
| Link Mass | Exact values | Manufacturing tolerance | Adapts to variations |
| Friction | Fixed grip | Variable ground | Works on slippery/grippy |
| Restitution | Fixed bounce | Variable ball | Catches different balls |
| Kp Gain | Fixed stiffness | Worn servos | Works with soft/stiff joints |
| Kd Gain | Fixed damping | Temperature/wear | Handles over/under-damping |
| Initial Pose | Perfect reset | Imperfect reset | Recovers from off-pose |
| Control Delay | Instant response | Network latency | Stable with communication lag |

---

## Training Impact

**Without domain randomization:**
- Policy learns perfect strategy for nominal parameters
- Fails immediately on real robot (different friction, delay, mass, etc.)
- Cannot transfer from sim to real

**With domain randomization:**
- Policy learns strategy that works for ANY combination of parameters
- Works on real robot despite sim-to-real gap
- More robust to hardware variations and wear over time

**Cost:** Takes longer to train (more diversity = harder problem), but final policy is production-ready.

---

## How They Work Together

Imagine training without domain randomization:
```
Episode 1: Perfect motors, perfect friction, perfect sensors → Optimal motion
Episode 2: Same as Episode 1 → Policy memorizes this exact scenario
Episode 1000: Still same scenario → Converges to "perfect solution"
Deploy on real robot: Failure! (Different friction, delayed response, motor bias)
```

With domain randomization:
```
Episode 1: Kp=90, friction=0.8, delay=20ms → Motion style A
Episode 2: Kp=110, friction=1.8, delay=5ms → Motion style B
Episode 3: Kp=100, friction=1.2, delay=15ms → Motion style C
...
Episode 1000: Different params every episode
Converges to: "Motion strategy that works for any Kp, friction, delay"
Deploy on real robot: Success! ✅ (Works despite hardware differences)
```

---

## Original Paper Reference

This concept comes from **"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"** (OpenAI, 2016).

Key idea: **Maximize diversity in training → minimize gap between simulation and reality.**
